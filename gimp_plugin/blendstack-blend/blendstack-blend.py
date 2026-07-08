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

All numeric work lives in ``blend_logic.py`` (GIMP-independent, tested
outside GIMP bit-exact against the core engine).  The frozen core package
is vendored alongside this file by ``gimp_plugin/sync_core.py``.
"""

import os
import sys

# The vendored core (./blendstack/) and blend_logic.py live next to this
# file; GIMP does not put the plugin folder on sys.path automatically.
_PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

# NumPy is also vendored (./vendor/numpy, see vendor_numpy.py) because on
# current Apple Silicon macOS, GIMP's bundled python3.10 cannot be invoked
# directly from a shell at all -- macOS Launch Constraints (AMFI) refuse
# it, so the "pip install into GIMP's interpreter" one-liner some older
# guides suggest does not work. Falling back to that instruction (below)
# only if this vendored copy is somehow missing/incompatible.
_VENDOR_DIR = os.path.join(_PLUGIN_DIR, "vendor")
if os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

import gi

gi.require_version("Gimp", "3.0")   # 3.0 API is stable across the 3.x series
gi.require_version("GimpUi", "3.0")
gi.require_version("Gegl", "0.4")
from gi.repository import Gegl, Gimp, GimpUi, GLib, GObject  # noqa: E402

# NumPy may be missing from GIMP's bundled Python (brief §6 known risk).
# Register the procedure regardless; the run() handler reports the fix.
_IMPORT_ERROR = None
try:
    import numpy as np
    import blend_logic
except ImportError as exc:  # pragma: no cover - exercised only inside GIMP
    np = None
    blend_logic = None
    _IMPORT_ERROR = str(exc)

PROC_NAME = "plug-in-blendstack"

NUMPY_INSTALL_CMD = (
    "/Applications/GIMP.app/Contents/MacOS/python3 -m pip install numpy"
)
NUMPY_HELP = (
    "BlendStack needs NumPy, and the copy normally bundled with this "
    "plugin (vendor/numpy) is missing or failed to load — reinstall the "
    "plugin, or run `python3 gimp_plugin/vendor_numpy.py` from the "
    "BlendStack repo and copy the result back in.\n\n"
    "If you're on an older GIMP/macOS combination where GIMP's own Python "
    "can be invoked directly from a terminal, you can alternatively run "
    "this once and restart GIMP:\n\n    " + NUMPY_INSTALL_CMD +
    "\n\n(On current Apple Silicon macOS this command is blocked by "
    "macOS Launch Constraints and will not work — use the vendored copy "
    "instead.)"
)

# Fallback mode list used only if the vendored core failed to import
# (e.g. NumPy missing) so the procedure still registers with sane args.
_FALLBACK_MODES = (("canon_bright", "Canon Bright"), ("canon_dark", "Canon Dark"))


def _mode_choices():
    if blend_logic is not None:
        try:
            return blend_logic.mode_choices()
        except Exception:
            pass
    return list(_FALLBACK_MODES)


def _error(procedure, message):
    """Build an EXECUTION_ERROR return; GIMP shows the message to the user."""
    return procedure.new_return_values(
        Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error(message)
    )


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
        if blend_logic is None or np is None:
            message = f"{NUMPY_HELP}\n\n(Python import error: {_IMPORT_ERROR})"
            if run_mode == Gimp.RunMode.INTERACTIVE:
                GimpUi.init("blendstack-blend")
                Gimp.message(message)
            return _error(procedure, message)

        if run_mode == Gimp.RunMode.INTERACTIVE:
            GimpUi.init("blendstack-blend")
            dialog = GimpUi.ProcedureDialog.new(procedure, config, "BlendStack")
            dialog.fill(None)  # auto-render all registered arguments
            if not dialog.run():
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

    def _blend(self, procedure, image, mode, softness, bias, basis):
        if image.get_base_type() != Gimp.ImageBaseType.RGB:
            return _error(
                procedure,
                "BlendStack works on RGB images — convert via "
                "Image > Mode > RGB and re-run.",
            )

        # get_layers() is top-first in GIMP 3; brief §6: top layer = base.
        visible = [layer for layer in image.get_layers() if layer.get_visible()]
        if len(visible) < blend_logic.MIN_LAYERS:
            return _error(
                procedure,
                f"BlendStack needs at least {blend_logic.MIN_LAYERS} visible "
                f"layers to blend; this image has {len(visible)}.",
            )
        if len(visible) > blend_logic.MAX_LAYERS:
            return _error(
                procedure,
                f"BlendStack blends at most {blend_logic.MAX_LAYERS} layers "
                f"(engine limit); this image has {len(visible)} visible — "
                "hide some layers and re-run.",
            )

        canvas_w, canvas_h = image.get_width(), image.get_height()

        # Read every visible layer as float via its GEGL buffer, flatten
        # alpha against black, and composite onto a black canvas-sized
        # background at the layer's offsets (brief §6).
        arrays = []
        for layer in visible:
            arrays.append(self._read_layer(layer, canvas_w, canvas_h))

        result = blend_logic.fold_visible(arrays, mode, softness, bias, basis)

        # Insert the composite as a new top layer inside one undo group.
        label = blend_logic.mode_label(mode)
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
            buffer.set(rect, "R'G'B' float", result.tobytes())
            buffer.flush()
            new_layer.update(0, 0, canvas_w, canvas_h)
        finally:
            image.undo_group_end()
        Gimp.displays_flush()

        return procedure.new_return_values(
            Gimp.PDBStatusType.SUCCESS, GLib.Error()
        )

    @staticmethod
    def _read_layer(layer, canvas_w, canvas_h):
        """Layer -> float32 (canvas_h, canvas_w, 3) on black background."""
        layer_w, layer_h = layer.get_width(), layer.get_height()
        buffer = layer.get_buffer()
        rect = Gegl.Rectangle.new(0, 0, layer_w, layer_h)
        if layer.has_alpha():
            data = buffer.get(rect, 1.0, "R'G'B'A float", Gegl.AbyssPolicy.NONE)
            rgba = np.frombuffer(data, dtype=np.float32).reshape(layer_h, layer_w, 4)
            rgb = blend_logic.flatten_alpha(rgba)
        else:
            data = buffer.get(rect, 1.0, "R'G'B' float", Gegl.AbyssPolicy.NONE)
            rgb = np.frombuffer(data, dtype=np.float32).reshape(layer_h, layer_w, 3)

        offsets = layer.get_offsets()
        # GIMP 3 returns (success, offset_x, offset_y); be tolerant of
        # bindings that drop the boolean.
        off_x, off_y = (offsets[-2], offsets[-1])
        return blend_logic.composite_at_canvas(
            (canvas_w, canvas_h), rgb, (int(off_x), int(off_y))
        )


Gimp.main(BlendStack.__gtype__, sys.argv)
