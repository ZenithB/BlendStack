#!/usr/bin/env python3
"""BlendStack — GIMP 3 plugin (project brief §6).

Blends all VISIBLE layers of the active image with the Canon-faithful
Comparative Bright / Comparative Dark engine and inserts the result as a
new layer at the top of the stack, named ``BlendStack <mode label>``.
Originals are untouched; the whole operation is one undo step.

Layer order assumption: ``Gimp.Image.get_layers()`` returns the layer
stack **top-first** in GIMP 3.  Per brief §6 the top layer is the
first/base image of the fold, so the list is used exactly as returned.
(Top-level layer groups are folded as their composited projection, since
``get_buffer()`` on a group yields the group's rendered content.)

Pixel access: float data is read and written through GEGL buffers using
the Babl format "R'G'B' float" (non-linear sRGB float).  That is correct
for the v1 modes because they are defined in gamma space and max/min are
monotone-invariant (brief §1), and it makes 8-, 16- and 32-bit images all
work identically.  Layers with alpha are read as "R'G'B'A float" and
flattened against black (rgb × a) to match core behaviour.

Backends: the numeric work runs through one of two interchangeable
backends, chosen at load time.  When NumPy imports, ``blend_logic.py``
(the frozen core vendored by ``gimp_plugin/sync_core.py``) does the maths.
When NumPy cannot load — the usual case inside GIMP on Apple Silicon,
where the hardened-runtime interpreter's library validation rejects
NumPy's ad-hoc-signed C extensions — the stdlib-only ``fold_purepy.py``
backend runs instead (brief §6 mitigation 3): slower, but dependency-free
and bit-exact with the engine at default settings.  Either way the plugin
always registers and always runs.
"""

import os
import sys

# The vendored core (./blendstack/), blend_logic.py and fold_purepy.py live
# next to this file; GIMP does not put the plugin folder on sys.path itself.
_PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

import gi

gi.require_version("Gimp", "3.0")   # 3.0 API is stable across the 3.x series
gi.require_version("GimpUi", "3.0")
gi.require_version("Gegl", "0.4")
gi.require_version("Gtk", "3.0")    # GIMP 3.2 UI is GTK3
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import (  # noqa: E402
    GdkPixbuf, Gegl, Gimp, GimpUi, GLib, GObject, Gtk,
)

# fold_purepy is the stdlib-only fallback backend and the source of the mode
# metadata + layer limits. It has no dependencies, so it always imports and
# the plugin always registers and runs.
import fold_purepy  # noqa: E402

# NumPy is the fast path. On Apple Silicon macOS it usually CANNOT load
# inside GIMP: GIMP's bundled Python runs under a hardened runtime that
# enforces library validation, which rejects NumPy's ad-hoc-signed C
# extensions ("Library Validation failed ... different Team IDs"). There is
# no way to make it load without re-signing GIMP itself, so we do NOT ship a
# NumPy wheel (it can't load) — we fall back to fold_purepy (brief §6
# mitigation 3). Where NumPy *does* load (Linux/Windows/Intel, or a GIMP
# built without library validation), we use it automatically for speed.
_NUMPY_ERROR = None
try:
    import numpy as np
    import blend_logic
except Exception as exc:  # pragma: no cover - path depends on the host
    np = None
    blend_logic = None
    _NUMPY_ERROR = str(exc)

USE_NUMPY = np is not None and blend_logic is not None

PROC_NAME = "plug-in-blendstack"

# Layer limits + mode metadata come from fold_purepy so they are available
# whether or not NumPy loaded.
MIN_LAYERS = fold_purepy.MIN_LAYERS
MAX_LAYERS = fold_purepy.MAX_LAYERS


def _mode_choices():
    return fold_purepy.mode_choices()


def _mode_label(mode):
    return fold_purepy.mode_label(mode)


def _error(procedure, message):
    """Build an EXECUTION_ERROR return; GIMP shows the message to the user."""
    return procedure.new_return_values(
        Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error(message)
    )


def _layer_float_bytes(layer):
    """(raw float32 bytes, w, h, has_alpha, off_x, off_y) for a layer."""
    layer_w, layer_h = layer.get_width(), layer.get_height()
    buffer = layer.get_buffer()
    rect = Gegl.Rectangle.new(0, 0, layer_w, layer_h)
    has_alpha = layer.has_alpha()
    fmt = "R'G'B'A float" if has_alpha else "R'G'B' float"
    data = buffer.get(rect, 1.0, fmt, Gegl.AbyssPolicy.NONE)
    offsets = layer.get_offsets()
    # GIMP 3 returns (success, offset_x, offset_y); tolerate bindings that
    # drop the leading boolean.
    off_x, off_y = int(offsets[-2]), int(offsets[-1])
    return data, layer_w, layer_h, has_alpha, off_x, off_y


# --------------------------------------------------------------------------
# Live preview (added post-v1 on user request; brief §6 originally shipped
# without one). The preview is a small, downscaled rendering shown inside
# the dialog and updated as the blend controls change.
# --------------------------------------------------------------------------

PREVIEW_MAX_EDGE = 360      # longest edge of the preview, in pixels
PREVIEW_DEBOUNCE_MS = 120   # coalesce rapid slider changes


class _PreviewController:
    """Owns the dialog's Preview checkbox + image and keeps them in sync.

    Ticking the box builds a one-time downscaled cache of the visible layers
    (via ``image.duplicate()`` + ``scale()`` so GIMP does the resampling),
    then every blend-control change re-folds just that small cache in pure
    Python (milliseconds) and repaints. The whole preview path is wrapped so
    any failure disables the preview quietly and never affects the real
    blend, which runs at full resolution when the dialog is accepted.
    """

    def __init__(self, image, config, check, container, image_widget):
        self._image = image
        self._config = config
        self._check = check
        self._container = container      # frame shown/hidden with the toggle
        self._image_widget = image_widget
        self._cache = None               # (list[array('f')], sw, sh) or None
        self._pending = 0                # debounce timeout source id
        self._handlers = []              # (obj, handler_id) to disconnect
        self._handlers.append((check, check.connect("toggled", self._on_toggle)))
        for prop in ("mode", "softness", "bias", "basis"):
            hid = config.connect("notify::" + prop, self._on_param)
            self._handlers.append((config, hid))

    def disconnect(self):
        """Drop all signal connections and any pending redraw (call when the
        dialog closes so repeat runs don't accumulate stale handlers)."""
        if self._pending:
            GLib.source_remove(self._pending)
            self._pending = 0
        for obj, hid in self._handlers:
            try:
                obj.disconnect(hid)
            except Exception:  # noqa: BLE001 - object may already be gone
                pass
        self._handlers = []

    # -- signal handlers ---------------------------------------------------

    def _on_toggle(self, _btn):
        if self._check.get_active():
            self._cache = None           # rebuilt on next recompute
            self._container.show()
            self._schedule()
        else:
            self._container.hide()

    def _on_param(self, *_args):
        if self._check.get_active():
            self._schedule()

    # -- work --------------------------------------------------------------

    def _schedule(self):
        if self._pending:
            GLib.source_remove(self._pending)
        self._pending = GLib.timeout_add(PREVIEW_DEBOUNCE_MS, self._recompute)

    def _build_cache(self):
        """Downscale the image once and read its visible layers to canvases."""
        self._cache = None
        dup = self._image.duplicate()
        try:
            cw, ch = dup.get_width(), dup.get_height()
            sw, sh = fold_purepy.preview_target_size(cw, ch, PREVIEW_MAX_EDGE)
            if (sw, sh) != (cw, ch):
                dup.scale(sw, sh)
                sw, sh = dup.get_width(), dup.get_height()
            visible = [l for l in dup.get_layers() if l.get_visible()]
            layers = []
            for layer in visible[:MAX_LAYERS]:
                data, w, h, has_alpha, ox, oy = _layer_float_bytes(layer)
                layers.append(
                    fold_purepy.layer_to_canvas(data, w, h, has_alpha, ox, oy, sw, sh)
                )
            self._cache = (layers, sw, sh)
        finally:
            dup.delete()  # off-screen duplicate; never shown, no undo impact

    def _recompute(self):
        self._pending = 0
        if not self._check.get_active():
            return False
        try:
            if self._cache is None:
                self._build_cache()
            layers, sw, sh = self._cache
            if len(layers) < MIN_LAYERS:
                self._image_widget.clear()
                return False
            mode = self._config.get_property("mode")
            softness = self._config.get_property("softness")
            bias = self._config.get_property("bias")
            basis = self._config.get_property("basis")
            result = fold_purepy.fold(layers, mode, softness, bias, basis)
            data = GLib.Bytes.new(fold_purepy.canvas_to_rgb8(result))
            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
                data, GdkPixbuf.Colorspace.RGB, False, 8, sw, sh, sw * 3
            )
            self._image_widget.set_from_pixbuf(pixbuf)
        except Exception as exc:  # never let preview break the dialog
            print(f"BlendStack preview error: {exc}", file=sys.stderr)
            self._cache = None
        return False  # one-shot timeout


class BlendStack(Gimp.PlugIn):
    """GObject-introspected plugin class (brief §6)."""

    # -- registration ------------------------------------------------------

    def do_query_procedures(self):
        return [PROC_NAME]

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.run, None
        )
        procedure.set_image_types("*")
        # Requires an open image; indifferent to how many drawables are
        # selected (the plugin works on the layer stack, not the selection).
        procedure.set_sensitivity_mask(
            Gimp.ProcedureSensitivityMask.DRAWABLE
            | Gimp.ProcedureSensitivityMask.DRAWABLES
            | Gimp.ProcedureSensitivityMask.NO_DRAWABLES
        )
        procedure.set_menu_label("BlendStack…")
        procedure.add_menu_path("<Image>/Filters/Combine/")
        procedure.set_documentation(
            "Blend all visible layers with Canon comparative modes",
            "Recreates the Canon EOS R5 Comparative Bright / Comparative "
            "Dark multiple-exposure blend modes over all visible layers "
            "(top layer = base). The result is inserted as a new top "
            "layer; the originals are untouched.",
            name,
        )
        procedure.set_attribution("BlendStack", "BlendStack project", "2026")

        # Procedure arguments (also scriptable non-interactively).
        mode_choice = Gimp.Choice.new()
        for idx, (mode_name, mode_lbl) in enumerate(_mode_choices()):
            mode_choice.add(mode_name, idx, mode_lbl, "")
        procedure.add_choice_argument(
            "mode", "_Mode", "Blend mode",
            mode_choice, "canon_bright", GObject.ParamFlags.READWRITE,
        )
        procedure.add_double_argument(
            "softness", "_Softness",
            "0 = hard select (pixel-exact Canon); 1-100 feathers the "
            "winner-takes-all boundary",
            0.0, 100.0, 0.0, GObject.ParamFlags.READWRITE,
        )
        procedure.add_double_argument(
            "bias", "_Bias",
            "Comparison offset -100..+100; positive lets the incoming "
            "(lower) layer win near-ties more often (0 = Canon)",
            -100.0, 100.0, 0.0, GObject.ParamFlags.READWRITE,
        )
        basis_choice = Gimp.Choice.new()
        basis_choice.add("per_channel", 0, "Per channel (Canon)", "")
        basis_choice.add("luminance", 1, "Luminance", "")
        procedure.add_choice_argument(
            "basis", "Comparison _basis",
            "per_channel = Canon-faithful (colours may mix); luminance = "
            "whole-pixel winner keeps its colour",
            basis_choice, "per_channel", GObject.ParamFlags.READWRITE,
        )
        return procedure

    # -- execution ---------------------------------------------------------

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        # GIMP 3.0.x ImageProcedure run signature:
        # (procedure, run_mode, image, drawables, config, run_data).
        # NumPy is optional: when it can't load (the usual case inside GIMP
        # on Apple Silicon), the stdlib fold_purepy backend runs instead, so
        # there is no hard failure here.
        if run_mode == Gimp.RunMode.INTERACTIVE:
            GimpUi.init("blendstack-blend")
            dialog = GimpUi.ProcedureDialog.new(procedure, config, "BlendStack")
            dialog.fill(None)  # auto-render all registered arguments
            # Attach the live preview (checkbox + downscaled image) below the
            # auto-filled controls. Wrapped so a preview/GTK failure can never
            # stop the blend controls from working.
            preview_ctrl = None
            try:
                preview_ctrl = self._attach_preview(dialog, image, config)
            except Exception as exc:  # noqa: BLE001
                print(f"BlendStack: preview unavailable ({exc})", file=sys.stderr)
            accepted = dialog.run()
            if preview_ctrl is not None:
                preview_ctrl.disconnect()
            if not accepted:
                dialog.destroy()
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CANCEL, GLib.Error()
                )
            dialog.destroy()

        mode = config.get_property("mode")
        softness = config.get_property("softness")
        bias = config.get_property("bias")
        basis = config.get_property("basis")

        try:
            return self._blend(procedure, image, mode, softness, bias, basis)
        except Exception as exc:  # keep GIMP responsive on unexpected errors
            return _error(procedure, f"BlendStack failed: {exc}")

    @staticmethod
    def _attach_preview(dialog, image, config):
        """Pack a Preview checkbox + image into the dialog; return its
        controller (kept alive for the dialog's lifetime)."""
        content = dialog.get_content_area()

        check = Gtk.CheckButton.new_with_label("Preview")
        frame = Gtk.Frame()
        preview_image = Gtk.Image()
        preview_image.set_margin_top(4)
        preview_image.set_margin_bottom(4)
        preview_image.set_margin_start(4)
        preview_image.set_margin_end(4)
        frame.add(preview_image)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.set_margin_top(6)
        box.set_margin_bottom(6)
        box.pack_start(check, False, False, 0)
        box.pack_start(frame, False, False, 0)
        content.pack_start(box, False, False, 0)

        box.show_all()
        frame.hide()  # preview area appears only once ticked

        return _PreviewController(image, config, check, frame, preview_image)

    def _blend(self, procedure, image, mode, softness, bias, basis):
        if image.get_base_type() != Gimp.ImageBaseType.RGB:
            return _error(
                procedure,
                "BlendStack works on RGB images — convert via "
                "Image > Mode > RGB and re-run.",
            )

        # get_layers() is top-first in GIMP 3; brief §6: top layer = base.
        visible = [layer for layer in image.get_layers() if layer.get_visible()]
        if len(visible) < MIN_LAYERS:
            return _error(
                procedure,
                f"BlendStack needs at least {MIN_LAYERS} visible "
                f"layers to blend; this image has {len(visible)}.",
            )
        if len(visible) > MAX_LAYERS:
            return _error(
                procedure,
                f"BlendStack blends at most {MAX_LAYERS} layers "
                f"(engine limit); this image has {len(visible)} visible — "
                "hide some layers and re-run.",
            )

        canvas_w, canvas_h = image.get_width(), image.get_height()

        # Read every visible layer as float via its GEGL buffer, flatten
        # alpha against black, and composite onto a black canvas-sized
        # background at the layer's offsets (brief §6). Both backends produce
        # canvas-sized layers; the fold then combines them.
        if USE_NUMPY:
            layers = [self._read_layer_np(l, canvas_w, canvas_h) for l in visible]
            result_bytes = blend_logic.fold_visible(
                layers, mode, softness, bias, basis
            ).tobytes()
        else:
            layers = [self._read_layer_pp(l, canvas_w, canvas_h) for l in visible]
            result_bytes = fold_purepy.to_bytes(
                fold_purepy.fold(layers, mode, softness, bias, basis)
            )

        # Insert the composite as a new top layer inside one undo group.
        label = _mode_label(mode)
        image.undo_group_start()
        try:
            new_layer = Gimp.Layer.new(
                image,
                f"BlendStack {label}",
                canvas_w,
                canvas_h,
                Gimp.ImageType.RGB_IMAGE,
                100.0,
                Gimp.LayerMode.NORMAL,
            )
            image.insert_layer(new_layer, None, 0)  # position 0 = top
            buffer = new_layer.get_buffer()
            rect = Gegl.Rectangle.new(0, 0, canvas_w, canvas_h)
            buffer.set(rect, "R'G'B' float", result_bytes)
            buffer.flush()
            new_layer.update(0, 0, canvas_w, canvas_h)
        finally:
            image.undo_group_end()
        Gimp.displays_flush()

        return procedure.new_return_values(
            Gimp.PDBStatusType.SUCCESS, GLib.Error()
        )

    @classmethod
    def _read_layer_np(cls, layer, canvas_w, canvas_h):
        """NumPy path: layer -> float32 (canvas_h, canvas_w, 3) on black."""
        data, w, h, has_alpha, off_x, off_y = _layer_float_bytes(layer)
        if has_alpha:
            rgba = np.frombuffer(data, dtype=np.float32).reshape(h, w, 4)
            rgb = blend_logic.flatten_alpha(rgba)
        else:
            rgb = np.frombuffer(data, dtype=np.float32).reshape(h, w, 3)
        return blend_logic.composite_at_canvas(
            (canvas_w, canvas_h), rgb, (off_x, off_y)
        )

    @classmethod
    def _read_layer_pp(cls, layer, canvas_w, canvas_h):
        """Pure-Python path: layer -> array('f') canvas on black background."""
        data, w, h, has_alpha, off_x, off_y = _layer_float_bytes(layer)
        return fold_purepy.layer_to_canvas(
            data, w, h, has_alpha, off_x, off_y, canvas_w, canvas_h
        )


Gimp.main(BlendStack.__gtype__, sys.argv)
