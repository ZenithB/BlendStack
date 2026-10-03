"""Offscreen smoke test for the standalone app (brief §8 Phase-2 checks).

Run with::

    QT_QPA_PLATFORM=offscreen python -m blendstack.app.selftest

Exercises, against a real (offscreen) MainWindow:

1. add 3 generated images of different sizes/formats → strip + preview;
2. preview render completes; pixmap non-null; histogram has data;
3. softness + one image's exposure changes trigger a re-render
   (generation counter advances);
4. drag-style reorder changes strip and document order;
4b. dynamic blend controls: switching to a param-less mode (multiply,
   average) hides softness/bias/basis and still renders; switching back to
   canon_bright shows them again; per-mode params reset on switch;
4c. preset round-trip for a param-less mode (multiply) restores the mode
   and its empty params;
5. preset save → clear → load round-trip restores images and settings;
6. the 21st image is refused with a clear message;
7. full-resolution export to 16-bit TIFF (dims + dtype verified);
8. per-image Solo (state + strip button click + S key; isolated preview with
   adjustments, banner, cleared on remove/clear, works with 1 image);
9. Move / Crop tool: layer move via state and via simulated mouse drag
   (anchored; ignored with the tool off / nothing selected; adjustments are
   NOT recomputed during a move), crop mode (double-click, handles, Apply,
   Cancel, Reset, tool-off cancels), export with crop + moved layer, preset
   round-trip of crop / move / denoise / curve / order (and an old preset),
   small window minimum size.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from blendstack.core import engine, geometry  # noqa: E402
from blendstack.core import io as bs_io  # noqa: E402
from blendstack.core.adjustments import DEFAULT_ORDER, Adjustments  # noqa: E402
from blendstack.app.main_window import MainWindow  # noqa: E402

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))


def wait_until(condition, timeout_ms: int = 8000) -> bool:
    """Spin the event loop until ``condition()`` or timeout."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        if condition():
            return True
        QTest.qWait(20)
    return bool(condition())


def make_image(path: Path, width: int, height: int, fmt: str, seed: int) -> Path:
    """Write a deterministic gradient+noise test image."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    base = np.concatenate(
        [x * np.ones((height, 1, 1), np.float32),
         y * np.ones((1, width, 1), np.float32),
         0.5 * (x + y)],
        axis=2,
    ).astype(np.float32)
    noise = rng.random((height, width, 3), dtype=np.float32) * 0.1
    arr = np.clip(base * 0.9 + noise, 0.0, 1.0)
    return bs_io.save_image(arr, path, format=fmt)


def render_settled(window: MainWindow) -> bool:
    p = window.preview
    return (
        p.requested_generation > 0
        and p.completed_generation == p.requested_generation
    )


def settle(window: MainWindow, timeout_ms: int = 10000) -> bool:
    """Wait until the debounce timer is idle and the last render completed."""
    return wait_until(
        lambda: not window.preview._timer.isActive() and render_settled(window),
        timeout_ms,
    )


def mouse_event(widget, kind, pos, button=Qt.LeftButton, buttons=Qt.LeftButton):
    """Deliver a synthetic mouse event with explicit buttons (QTest's
    mouseMove cannot carry a pressed button)."""
    local = QPointF(pos)
    event = QMouseEvent(
        kind, local, QPointF(widget.mapToGlobal(local)),
        button, buttons, Qt.NoModifier,
    )
    QApplication.sendEvent(widget, event)


def drag(widget, start, end, steps: int = 4) -> None:
    """Press at ``start``, move in ``steps`` increments, release at ``end``."""
    start, end = QPointF(start), QPointF(end)
    mouse_event(widget, QEvent.MouseButtonPress, start)
    for i in range(1, steps + 1):
        t = i / steps
        mouse_event(
            widget, QEvent.MouseMove,
            QPointF(start.x() + (end.x() - start.x()) * t,
                    start.y() + (end.y() - start.y()) * t),
            button=Qt.NoButton,
        )
    mouse_event(widget, QEvent.MouseButtonRelease, end,
                buttons=Qt.NoButton)


def expected_single(entry, entries):
    """Reference solo frame: cover_scale -> adjust_image -> place (offset)."""
    target = geometry.target_dimensions(
        [(e.proxy.shape[1], e.proxy.shape[0]) for e in entries]
    )
    adjusted = engine.adjust_image(
        geometry.cover_scale(entry.proxy, target), entry.adjustments
    )
    dx, dy = geometry.offset_px(
        entry.adjustments.move_x, entry.adjustments.move_y, target
    )
    return geometry.place(adjusted, target, dx, dy)[0]


def reload_images(window: MainWindow, paths) -> list[int]:
    window.move_crop_action.setChecked(False)
    window.state.clear()
    settle(window)
    window.add_files(paths)
    window.strip.setCurrentRow(0)
    settle(window)
    return window.state.entry_ids()


def check_solo(window: MainWindow, paths) -> None:
    """Section 8: per-image solo."""
    state, strip, canvas = window.state, window.strip, window.canvas
    ids = reload_images(window, paths)
    entries = state.entries
    blend_frame = canvas.composite().copy()

    # -- via the state -----------------------------------------------------------
    state.set_solo(ids[1])
    settle(window)
    check("solo set via state", state.solo_id == ids[1]
          and strip.solo_id() == ids[1])
    check("solo banner shows file name",
          canvas.solo_label() == "SOLO — two.jpg",
          f"label={canvas.solo_label()}")
    img = canvas.grab().toImage()
    rect = canvas.display_rect()
    px = img.pixelColor(int(rect.left()) + 11, int(rect.top()) + 22)
    check("solo banner is painted on the canvas",
          px.red() > 200 and 120 < px.green() < 220 and px.blue() < 80,
          f"pixel={px.getRgb()}")
    check("solo frame is the single image (no adjustments)",
          np.allclose(canvas.composite(),
                      expected_single(entries[1], entries), atol=1e-6)
          and not np.allclose(canvas.composite(), blend_frame, atol=1e-3))
    check("histogram has data while soloed", window.histogram.has_data())

    # exposure applied to the soloed image shows in isolation
    adj = dataclasses.replace(state.entry(ids[1]).adjustments, exposure=1.0)
    state.set_adjustments(ids[1], adj)
    settle(window)
    ref = engine.adjust_image(
        geometry.cover_scale(state.entry(ids[1]).proxy, (512, 384)), adj
    )
    base = geometry.cover_scale(state.entry(ids[1]).proxy, (512, 384))
    check("solo shows the adjusted image (exposure)",
          np.allclose(canvas.composite(), ref, atol=1e-6)
          and abs(float(canvas.composite().mean()) - float(ref.mean())) < 1e-6
          and float(canvas.composite().mean()) > float(base.mean()) + 0.01,
          f"mean={canvas.composite().mean():.4f} ref={ref.mean():.4f}")
    check("solo differs from the blend",
          abs(float(canvas.composite().mean()) - float(blend_frame.mean())) > 1e-3)
    state.set_adjustments(ids[1], dataclasses.replace(adj, exposure=0.0))
    state.set_solo(None)
    settle(window)
    check("un-solo returns to the blend",
          state.solo_id is None and canvas.solo_label() is None
          and np.allclose(canvas.composite(), blend_frame, atol=1e-6))

    # -- via a simulated click on the strip button -------------------------------
    sel_before = strip.current_entry_id()
    order_before = strip.current_ids()
    button = strip.solo_button_rect(2)
    check("solo button rect is inside the row",
          not button.isEmpty() and strip.viewport().rect().contains(button))
    QTest.mouseClick(strip.viewport(), Qt.LeftButton, pos=button.center())
    settle(window)
    check("clicking the S button solos that row",
          state.solo_id == ids[2] and strip.solo_id() == ids[2])
    check("S click kept order and selection",
          strip.current_ids() == order_before == state.entry_ids()
          and strip.current_entry_id() == sel_before,
          f"order={strip.current_ids()} sel={strip.current_entry_id()}")
    QTest.mouseClick(strip.viewport(), Qt.LeftButton,
                     pos=strip.solo_button_rect(1).center())
    check("soloing another row switches (exclusive)", state.solo_id == ids[1])
    QTest.mouseClick(strip.viewport(), Qt.LeftButton,
                     pos=strip.solo_button_rect(1).center())
    check("clicking the active S button un-solos", state.solo_id is None
          and strip.solo_id() is None)
    # A press+drag starting on the button must not start a reorder.
    mouse_event(strip.viewport(), QEvent.MouseButtonPress,
                strip.solo_button_rect(0).center())
    mouse_event(strip.viewport(), QEvent.MouseMove,
                strip.solo_button_rect(2).center(), button=Qt.NoButton)
    mouse_event(strip.viewport(), QEvent.MouseButtonRelease,
                strip.solo_button_rect(2).center(), buttons=Qt.NoButton)
    check("dragging from the S button does not reorder",
          state.entry_ids() == ids and strip.current_ids() == ids)
    state.set_solo(None)
    strip.setCurrentRow(0)
    strip.setFocus()
    QTest.keyClick(strip, Qt.Key_S)
    check("S key solos the selected row", state.solo_id == ids[0])
    QTest.keyClick(strip, Qt.Key_S)
    check("S key again un-solos", state.solo_id is None)
    # selection still works normally by clicking the row body
    QTest.mouseClick(strip.viewport(), Qt.LeftButton,
                     pos=strip.visualItemRect(strip.item(1)).center())
    check("row body click still selects", strip.current_entry_id() == ids[1])
    strip.setCurrentRow(0)

    # -- cleared on remove / clear ------------------------------------------------
    state.set_solo(ids[2])
    state.remove([ids[2]])
    settle(window)
    check("solo cleared when its image is removed",
          state.solo_id is None and strip.solo_id() is None
          and canvas.solo_label() is None)
    ids = reload_images(window, paths)
    state.set_solo(ids[0])
    state.clear()
    settle(window)
    check("solo cleared when the list is cleared",
          state.solo_id is None and strip.solo_id() is None
          and canvas.solo_label() is None)

    # -- solo with a single image ---------------------------------------------------
    window.add_files([paths[0]])
    settle(window)
    only = state.entry_ids()[0]
    check("1 image: placeholder (no blend possible)", not canvas.has_image())
    state.set_solo(only)
    settle(window)
    check("1 image: solo renders",
          canvas.has_image() and canvas.composite().shape == (480, 640, 3),
          f"shape={None if canvas.composite() is None else canvas.composite().shape}")
    state.set_solo(None)
    settle(window)
    check("1 image: un-solo returns to the placeholder", not canvas.has_image())


def check_move(window: MainWindow, paths) -> None:
    """Section 9a: layer move."""
    state, strip, canvas = window.state, window.strip, window.canvas
    ids = reload_images(window, paths)
    worker = window.preview.worker

    # -- via the state ---------------------------------------------------------------
    frame0 = canvas.composite().copy()
    state.set_placement(ids[0], 0.1, -0.05)
    settle(window)
    adj = state.entry(ids[0]).adjustments
    check("set_placement reaches the entry",
          adj.move_x == 0.1 and adj.move_y == -0.05)
    check("moving a layer changes the blend",
          not np.allclose(canvas.composite(), frame0, atol=1e-4))
    state.reset_placement(ids[0])
    settle(window)
    check("reset_placement recentres",
          state.entry(ids[0]).adjustments.move_x == 0.0
          and np.allclose(canvas.composite(), frame0, atol=1e-6))
    # panel edits cannot clobber the placement
    state.set_placement(ids[0], 0.2, 0.1)
    window.adjustments_panel.exposure_row.slider.setValue(50)
    a = state.entry(ids[0]).adjustments
    check("panel edit preserves placement",
          a.move_x == 0.2 and a.move_y == 0.1 and a.exposure == 0.5,
          f"{a.move_x} {a.move_y} {a.exposure}")
    window.adjustments_panel.exposure_row.slider.setValue(0)
    state.reset_placement(ids[0])
    settle(window)

    # -- via the mouse -------------------------------------------------------------------
    window.move_crop_action.setChecked(True)
    settle(window)
    check("tool on: full frame displayed, canvas interactive",
          canvas.interactive() and canvas.tool_enabled())
    rect = canvas.display_rect()
    start = rect.center()
    dx_px, dy_px = 40.0, -30.0
    mouse_event(canvas, QEvent.MouseButtonPress, start)
    check("open hand before / closed hand while dragging",
          canvas.cursor().shape() == Qt.ClosedHandCursor)
    mouse_event(canvas, QEvent.MouseMove,
                QPointF(start.x() + dx_px / 2, start.y() + dy_px / 2),
                button=Qt.NoButton)
    mid = state.entry(ids[0]).adjustments
    check("layer follows the mouse live",
          abs(mid.move_x - dx_px / 2 / rect.width()) < 1e-6
          and abs(mid.move_y - dy_px / 2 / rect.height()) < 1e-6,
          f"{mid.move_x} {mid.move_y}")
    mouse_event(canvas, QEvent.MouseMove,
                QPointF(start.x() + dx_px, start.y() + dy_px),
                button=Qt.NoButton)
    mouse_event(canvas, QEvent.MouseButtonRelease,
                QPointF(start.x() + dx_px, start.y() + dy_px),
                buttons=Qt.NoButton)
    fx, fy = dx_px / rect.width(), dy_px / rect.height()
    adj = state.entry(ids[0]).adjustments
    check("drag moved the selected image by the dragged fraction",
          abs(adj.move_x - fx) < 1e-3 and abs(adj.move_y - fy) < 1e-3,
          f"got ({adj.move_x:.4f},{adj.move_y:.4f}) want ({fx:.4f},{fy:.4f})")
    check("only the selected image moved",
          all(state.entry(i).adjustments.move_x == 0.0 for i in ids[1:]))
    check("open hand after release",
          canvas.cursor().shape() == Qt.OpenHandCursor)
    mouse_event(canvas, QEvent.MouseMove, QPointF(start.x() + 200, start.y()),
                button=Qt.NoButton, buttons=Qt.NoButton)
    settle(window)
    after = state.entry(ids[0]).adjustments
    check("released layer stays anchored",
          after.move_x == adj.move_x and after.move_y == adj.move_y)

    # a second drag continues from the anchored position
    drag(canvas, start, QPointF(start.x() + 10, start.y()))
    again = state.entry(ids[0]).adjustments
    check("second drag is relative to the anchored position",
          abs(again.move_x - (adj.move_x + 10 / rect.width())) < 1e-3)
    state.set_placement(ids[0], adj.move_x, adj.move_y)

    # -- no adjust recompute while moving --------------------------------------------------
    calls = {"n": 0}
    real_adjust = engine.adjust_image

    def counting(image, adjustments=None):
        calls["n"] += 1
        return real_adjust(image, adjustments)

    engine.adjust_image = counting
    try:
        settle(window)
        a0, s0, c0 = worker.adjust_calls, worker.scale_calls, calls["n"]
        gen0 = window.preview.completed_generation
        mouse_event(canvas, QEvent.MouseButtonPress, start)
        for i in range(1, 7):
            mouse_event(canvas, QEvent.MouseMove,
                        QPointF(start.x() + 12 * i, start.y() + 5 * i),
                        button=Qt.NoButton)
            QTest.qWait(110)  # let the debounced renders run mid-drag
        mouse_event(canvas, QEvent.MouseButtonRelease,
                    QPointF(start.x() + 72, start.y() + 30), buttons=Qt.NoButton)
        settle(window)
        renders = window.preview.completed_generation - gen0
        check("move drag triggered several re-renders", renders >= 3,
              f"renders={renders}")
        check("adjust_image NOT recomputed during a move",
              worker.adjust_calls == a0 and calls["n"] == c0,
              f"worker {a0}->{worker.adjust_calls}, wrapped {c0}->{calls['n']}")
        check("cover_scale NOT recomputed during a move",
              worker.scale_calls == s0)
        # opacity is a fold-step setting: also no recompute
        state.set_adjustments(
            ids[1], dataclasses.replace(state.entry(ids[1]).adjustments,
                                        opacity=60.0))
        settle(window)
        check("opacity change does not recompute adjustments",
              worker.adjust_calls == a0 and calls["n"] == c0)
        # positive control: a real adjustment edit recomputes exactly one image
        state.set_adjustments(
            ids[2], dataclasses.replace(state.entry(ids[2]).adjustments,
                                        contrast=25.0))
        settle(window)
        check("a pixel adjustment recomputes only that image (control)",
              worker.adjust_calls == a0 + 1 and calls["n"] == c0 + 1,
              f"worker {a0}->{worker.adjust_calls}, wrapped {c0}->{calls['n']}")
    finally:
        engine.adjust_image = real_adjust
    state.set_adjustments(ids[1], dataclasses.replace(
        state.entry(ids[1]).adjustments, opacity=100.0))
    state.set_adjustments(ids[2], dataclasses.replace(
        state.entry(ids[2]).adjustments, contrast=0.0))

    # -- ignored with no selection / with the tool off --------------------------------------
    for i in ids:
        state.reset_placement(i)
    settle(window)
    strip.setCurrentRow(-1)
    check("no row selected", strip.current_entry_id() is None)
    drag(canvas, start, QPointF(start.x() + 50, start.y() + 20))
    check("no selected image: nothing moves",
          all(state.entry(i).adjustments.move_x == 0.0
              and state.entry(i).adjustments.move_y == 0.0 for i in ids))
    check("no selected image: status hint shown",
          "Select an image" in window.statusBar().currentMessage(),
          window.statusBar().currentMessage())
    strip.setCurrentRow(0)
    window.move_crop_action.setChecked(False)
    settle(window)
    check("tool off: canvas not interactive", not canvas.interactive())
    drag(canvas, start, QPointF(start.x() + 50, start.y() + 20))
    check("tool off: drag does not move the layer",
          state.entry(ids[0]).adjustments.move_x == 0.0
          and state.entry(ids[0]).adjustments.move_y == 0.0)


def check_crop(window: MainWindow, paths) -> None:
    """Section 9b: canvas crop."""
    state, strip, canvas = window.state, window.strip, window.canvas
    ids = reload_images(window, paths)
    full_shape = (384, 512, 3)
    action = window.move_crop_action

    action.setChecked(True)
    settle(window)
    rect = canvas.display_rect()
    center = rect.center()
    check("crop: starts without a crop", state.crop is None)
    QTest.mouseDClick(canvas, Qt.LeftButton, pos=center.toPoint())
    check("double-click enters crop mode",
          canvas.crop_mode() and window.crop_bar.isVisible())
    check("crop bar: Reset hidden without a crop",
          not window.reset_crop_button.isVisible())
    c = canvas.crop_rect()
    check("initial crop is the canvas inset ~8%",
          all(abs(a - b) < 1e-9 for a, b in zip(c, (0.08, 0.08, 0.92, 0.92))),
          f"crop={c}")
    check("double-click did not move the layer",
          state.entry(ids[0]).adjustments.move_x == 0.0)
    hint = window.minimumSizeHint()
    check("crop bar keeps the window minimum small",
          hint.width() <= 900 and hint.height() <= 450,
          f"minimumSizeHint={hint.width()}x{hint.height()}")

    # drag the bottom-right handle inwards
    br = canvas.crop_handle_point("br")
    drag(canvas, br, br - QPointF(60, 40).toPoint())
    c1 = canvas.crop_rect()
    check("dragging the BR handle shrinks the rectangle",
          abs(c1[2] - (0.92 - 60 / rect.width())) < 2e-3
          and abs(c1[3] - (0.92 - 40 / rect.height())) < 2e-3
          and c1[0] == c[0] and c1[1] == c[1], f"crop={c1}")
    check("crop edit does not move layers",
          state.entry(ids[0]).adjustments.move_x == 0.0
          and state.entry(ids[0]).adjustments.move_y == 0.0)
    # edge handle
    left = canvas.crop_handle_point("l")
    drag(canvas, left, left + QPointF(30, 0).toPoint())
    c2 = canvas.crop_rect()
    check("dragging the left edge handle moves only x0",
          abs(c2[0] - (0.08 + 30 / rect.width())) < 2e-3
          and c2[1:] == c1[1:], f"crop={c2}")
    # minimum size: drag TL far past BR
    tl = canvas.crop_handle_point("tl")
    drag(canvas, tl, QPointF(rect.right() + 200, rect.bottom() + 200).toPoint())
    c3 = canvas.crop_rect()
    check("minimum crop size (~16 px) enforced",
          (c3[2] - c3[0]) * rect.width() >= 15.5
          and (c3[3] - c3[1]) * rect.height() >= 15.5,
          f"crop={c3}")
    # move inside, clamped to the canvas
    canvas.exit_crop_mode()
    check("exit_crop_mode leaves crop mode", not canvas.crop_mode()
          and not window.crop_bar.isVisible())
    QTest.mouseDClick(canvas, Qt.LeftButton, pos=center.toPoint())
    mid = canvas.crop_handle_point("move")
    drag(canvas, mid, mid + QPointF(5000, 5000).toPoint(), steps=2)
    c4 = canvas.crop_rect()
    check("moving the rectangle clamps to the canvas",
          abs(c4[2] - 1.0) < 1e-9 and abs(c4[3] - 1.0) < 1e-9
          and abs((c4[2] - c4[0]) - 0.84) < 1e-9, f"crop={c4}")
    # press outside the rectangle does nothing
    window.cancel_crop_button.click()
    check("Cancel leaves crop mode, no crop stored",
          not canvas.crop_mode() and state.crop is None
          and not window.crop_bar.isVisible())

    # Apply
    QTest.mouseDClick(canvas, Qt.LeftButton, pos=center.toPoint())
    br = canvas.crop_handle_point("br")
    drag(canvas, br, br - QPointF(80, 50).toPoint())
    tl = canvas.crop_handle_point("tl")
    drag(canvas, tl, tl + QPointF(40, 30).toPoint())
    want = canvas.crop_rect()
    window.apply_crop_button.click()
    check("Apply stores the crop and leaves crop mode",
          state.crop is not None and not canvas.crop_mode()
          and all(abs(a - b) < 1e-12 for a, b in zip(state.crop, want)),
          f"crop={state.crop} want={want}")
    applied = state.crop
    settle(window)
    check("tool on after Apply: still the full canvas",
          canvas.composite().shape == full_shape and canvas.frame_is_full())
    action.setChecked(False)
    settle(window)
    x0, y0, x1, y1 = engine.crop_box_px(applied, 512, 384)
    shape = canvas.composite().shape
    check("tool off: preview is the cropped result",
          shape == (y1 - y0, x1 - x0, 3), f"shape={shape} box={(x0, y0, x1, y1)}")
    pm = canvas.pixmap()
    check("tool off: preview aspect matches the crop",
          abs(pm.width() / pm.height() - (x1 - x0) / (y1 - y0)) < 1e-9
          and not canvas.frame_is_full())
    check("histogram follows the cropped frame", window.histogram.has_data())
    # solo respects the crop
    state.set_solo(ids[1])
    settle(window)
    check("solo respects the crop",
          canvas.composite().shape == (y1 - y0, x1 - x0, 3))
    state.set_solo(None)
    settle(window)

    # tool back on: old cropped frame must not accept mouse input
    action.setChecked(True)
    check("tool on with a cropped frame displayed: mouse ignored until the "
          "full frame arrives", not canvas.interactive())
    before = state.entry(ids[0]).adjustments
    drag(canvas, center, QPointF(center.x() + 30, center.y()))
    check("press on the stale cropped frame does not move the layer",
          state.entry(ids[0]).adjustments == before)
    settle(window)
    check("full frame arrived: canvas interactive again",
          canvas.interactive() and canvas.composite().shape == full_shape)
    check("applied crop is drawn as an overlay (stored on the canvas)",
          canvas._applied_crop == applied)

    # Cancel keeps the previous crop
    QTest.mouseDClick(canvas, Qt.LeftButton, pos=center.toPoint())
    check("crop mode starts from the applied crop",
          canvas.crop_rect() is not None
          and all(abs(a - b) < 1e-12 for a, b in zip(canvas.crop_rect(), applied)))
    check("crop bar: Reset visible when a crop exists",
          window.reset_crop_button.isVisible())
    br = canvas.crop_handle_point("br")
    drag(canvas, br, br - QPointF(50, 50).toPoint())
    window.cancel_crop_button.click()
    check("Cancel keeps the previous crop", state.crop == applied)

    # Esc / Enter shortcuts (QShortcut needs an active window, even offscreen)
    for box in QApplication.topLevelWidgets():  # dismiss modal notices (export)
        if isinstance(box, QMessageBox):
            box.close()
    QTest.qWait(50)
    window.activateWindow()
    wait_until(window.isActiveWindow, 2000)
    QTest.mouseDClick(canvas, Qt.LeftButton, pos=center.toPoint())
    QTest.keyClick(window, Qt.Key_Escape)
    check("Esc cancels the crop edit", not canvas.crop_mode()
          and state.crop == applied)
    QTest.mouseDClick(canvas, Qt.LeftButton, pos=center.toPoint())
    br = canvas.crop_handle_point("br")
    drag(canvas, br, br - QPointF(20, 20).toPoint())
    edited = canvas.crop_rect()
    QTest.keyClick(window, Qt.Key_Return)
    check("Enter applies the crop edit",
          not canvas.crop_mode() and state.crop is not None
          and all(abs(a - b) < 1e-12 for a, b in zip(state.crop, edited))
          and state.crop != applied, f"crop={state.crop}")
    applied = state.crop

    # toggling the tool off cancels an in-progress crop
    QTest.mouseDClick(canvas, Qt.LeftButton, pos=center.toPoint())
    br = canvas.crop_handle_point("br")
    drag(canvas, br, br - QPointF(70, 70).toPoint())
    action.setChecked(False)
    check("tool off while cropping = Cancel",
          not canvas.crop_mode() and not window.crop_bar.isVisible()
          and state.crop == applied)
    settle(window)

    # Reset crop
    action.setChecked(True)
    settle(window)
    QTest.mouseDClick(canvas, Qt.LeftButton, pos=center.toPoint())
    window.reset_crop_button.click()
    settle(window)
    check("Reset crop clears the crop", state.crop is None
          and not canvas.crop_mode() and not window.crop_bar.isVisible())
    action.setChecked(False)
    settle(window)
    check("after Reset the preview is the full canvas again",
          canvas.composite().shape == full_shape)


def check_export_and_presets(window: MainWindow, paths, tmp: Path) -> None:
    """Section 9c: export honours crop + move; preset round-trips."""
    import imageio.v3 as iio

    state = window.state
    ids = reload_images(window, paths)
    crop = (0.1, 0.2, 0.7, 0.9)
    state.set_crop(crop)
    state.set_placement(ids[0], 0.2, 0.0)
    state.set_placement(ids[1], -0.1, 0.05)
    out = tmp / "cropped.tif"
    done: list[tuple[bool, str]] = []
    window.export_done.connect(lambda ok, msg: done.append((ok, msg)))
    started = window.export_to(out, "tiff")
    wait_until(lambda: bool(done), timeout_ms=30000)
    check("export with crop finished", started and bool(done) and done[0][0],
          f"done={done}")
    arr = iio.imread(out)
    box = engine.crop_box_px(crop, 512, 384)  # canvas = smallest source
    check("export dims equal crop_box_px on the full-res canvas",
          arr.shape == (box[3] - box[1], box[2] - box[0], 3),
          f"shape={arr.shape} box={box}")
    arrays = [bs_io.load_image(p) for p in paths]
    adjs = [e.adjustments for e in state.entries]
    want = engine.blend_arrays(arrays, adjs, state.mode, state.params, crop=crop)
    got = arr.astype(np.float32) / 65535.0
    check("export honours the moved layers (matches blend_arrays)",
          got.shape == want.shape and float(np.abs(got - want).max()) < 2e-3,
          f"maxdiff={float(np.abs(got - want).max()):.5f}")
    unmoved = engine.blend_arrays(arrays, None, state.mode, state.params, crop=crop)
    check("moved export differs from the unmoved one",
          float(np.abs(got - unmoved).max()) > 0.01)

    # -- preset round trip ---------------------------------------------------------
    curve = ((0.0, 0.0), (0.5, 0.62), (1.0, 1.0))
    order = tuple(reversed(DEFAULT_ORDER))
    custom = dataclasses.replace(
        state.entry(ids[0]).adjustments,
        denoise=35.0, curve_master=curve, order=order, exposure=0.5,
    )
    state.set_adjustments(ids[0], custom)
    saved = {e.path: e.adjustments for e in state.entries}
    preset = tmp / "tools.bsp"
    window.save_preset_to(preset)
    document = json.loads(preset.read_text(encoding="utf-8"))
    check("preset schema 2 carries crop and placement",
          document["schema"] == 2 and document["crop"] == list(crop)
          and document["images"][0]["adjustments"]["move_x"] == 0.2)
    state.clear()
    settle(window)
    check("clear dropped the crop", state.crop is None)
    window.load_preset_from(preset)
    settle(window)
    loaded = {e.path: e.adjustments for e in state.entries}
    check("preset round-trips crop", state.crop == crop, f"crop={state.crop}")
    check("preset round-trips every adjustment incl. move/denoise/curve/order",
          loaded == saved, f"{loaded} != {saved}")
    first = state.entries[0].adjustments
    check("preset round-trips move + denoise + curve + order values",
          first.move_x == 0.2 and first.denoise == 35.0
          and first.curve_master == curve and first.order == order,
          f"{first}")
    check("loaded preset renders the cropped preview",
          window.canvas.composite().shape == (
              box[3] - box[1], box[2] - box[0], 3))

    # old-format preset (schema 1: no crop, six adjustment keys) still loads
    old = {
        "schema": 1, "app": "BlendStack", "mode": "canon_bright",
        "params": {"softness": 0.0, "bias": 0.0, "basis": "per_channel"},
        "output": {"format": "tiff"},
        "images": [
            {"path": str(p.resolve()),
             "adjustments": {"exposure": 0.25, "contrast": 0.0,
                             "saturation": 0.0, "sharpen_radius": 1.0,
                             "sharpen_amount": 0.0, "opacity": 100.0}}
            for p in paths
        ],
    }
    old_path = tmp / "old.bsp"
    old_path.write_text(json.dumps(old), encoding="utf-8")
    window.load_preset_from(old_path)
    settle(window)
    e0 = state.entries[0].adjustments
    check("old (schema 1) preset loads with no crop",
          len(state.entries) == 3 and state.crop is None
          and e0.exposure == 0.25 and e0.move_x == 0.0 and e0.denoise == 0.0)
    check("old preset renders the full canvas",
          window.canvas.composite().shape == (384, 512, 3))

    hint = window.minimumSizeHint()
    check("window minimumSizeHint stays small (<= 900 x 450)",
          hint.width() <= 900 and hint.height() <= 450,
          f"minimumSizeHint={hint.width()}x{hint.height()}")
    check("controls scroll area and pinned histogram still present",
          window.controls_scroll.widget() is not None
          and window.histogram.isVisible())
    state.set_crop(None)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()

    tmp = Path(tempfile.mkdtemp(prefix="blendstack-selftest-"))
    p1 = make_image(tmp / "one.png", 640, 480, "png", 1)
    p2 = make_image(tmp / "two.jpg", 800, 600, "jpeg", 2)
    p3 = make_image(tmp / "three.tif", 512, 384, "tiff", 3)

    # -- 1. add 3 images of mixed sizes/formats -------------------------------
    report = window.add_files([p1, p2, p3])
    check("add 3 images accepted", len(report.added) == 3 and report.ok,
          f"added={len(report.added)} errors={report.errors}")
    check("strip shows 3 items in drop order",
          window.strip.count() == 3
          and [window.strip.item(i).text() for i in range(3)]
          == ["one.png", "two.jpg", "three.tif"])

    # -- 2. preview renders; pixmap + histogram --------------------------------
    ok = wait_until(lambda: render_settled(window) and window.canvas.has_image())
    check("initial preview render completed", ok)
    check("preview pixmap is non-null", window.canvas.has_image())
    check("histogram has data", window.histogram.has_data())

    # -- 3. slider changes re-render (generation counter advances) -------------
    gen_before = window.preview.completed_generation
    window.blend_controls.softness_row.slider.setValue(30)  # user-style edit
    window.strip.setCurrentRow(0)
    window.adjustments_panel.exposure_row.slider.setValue(100)  # +1.00 EV
    ok = wait_until(
        lambda: render_settled(window)
        and window.preview.completed_generation > gen_before
    )
    check("re-render after softness+exposure change (generation advanced)",
          ok and window.preview.completed_generation > gen_before,
          f"before={gen_before} after={window.preview.completed_generation}")
    check("softness reached state", window.state.params["softness"] == 30.0)
    first_id = window.state.entry_ids()[0]
    entry0 = window.state.entry(first_id)
    check("exposure reached state", entry0 is not None
          and entry0.adjustments.exposure == 1.0,
          f"exposure={entry0.adjustments.exposure if entry0 else None}")

    # -- 4. reorder two images ---------------------------------------------------
    ids_before = window.state.entry_ids()
    names_before = [window.strip.item(i).text() for i in range(window.strip.count())]
    window.strip.move_item(0, 1)
    ids_after = window.state.entry_ids()
    names_after = [window.strip.item(i).text() for i in range(window.strip.count())]
    check("reorder changed strip order", names_after != names_before
          and names_after[0] == names_before[1])
    check("reorder changed document order",
          ids_after == [ids_before[1], ids_before[0], ids_before[2]],
          f"before={ids_before} after={ids_after}")
    wait_until(lambda: render_settled(window))

    # -- 4b. dynamic per-mode blend controls ----------------------------------------
    bc = window.blend_controls

    def _select_mode(name: str) -> None:
        idx = bc.mode_combo.findData(name)
        bc.mode_combo.setCurrentIndex(idx)  # user-style edit → mode_changed

    gen_before = window.preview.completed_generation
    _select_mode("multiply")
    ok = wait_until(
        lambda: render_settled(window)
        and window.preview.completed_generation > gen_before
        and window.canvas.has_image()
    )
    check("multiply selected reaches state", window.state.mode == "multiply",
          f"mode={window.state.mode}")
    check("multiply hides softness/bias/basis rows",
          not bc.softness_row.isVisible()
          and not bc.bias_row.isVisible()
          and not bc.basis_row.isVisible())
    check("multiply params reset to empty", window.state.params == {},
          f"params={window.state.params}")
    check("preview still renders in multiply", ok and window.canvas.has_image())

    gen_before = window.preview.completed_generation
    _select_mode("average")
    ok = wait_until(
        lambda: render_settled(window)
        and window.preview.completed_generation > gen_before
        and window.canvas.has_image()
    )
    check("average selected reaches state", window.state.mode == "average",
          f"mode={window.state.mode}")
    check("preview renders in average (linear-light)",
          ok and window.canvas.has_image())

    gen_before = window.preview.completed_generation
    _select_mode("canon_bright")
    wait_until(
        lambda: render_settled(window)
        and window.preview.completed_generation > gen_before
    )
    check("canon_bright shows softness/bias/basis rows again",
          bc.softness_row.isVisible()
          and bc.bias_row.isVisible()
          and bc.basis_row.isVisible())
    check("canon_bright params reset to defaults",
          set(window.state.params) == {"softness", "bias", "basis"},
          f"params={window.state.params}")

    # -- 4c. preset round-trip for a param-less mode (multiply) ---------------------
    _select_mode("multiply")
    wait_until(lambda: render_settled(window))
    multiply_preset = tmp / "multiply.bsp"
    window.save_preset_to(multiply_preset)
    check("multiply preset written", multiply_preset.exists())
    _select_mode("canon_bright")  # move away so reload must restore it
    wait_until(lambda: render_settled(window))
    window.load_preset_from(multiply_preset)
    wait_until(lambda: render_settled(window))
    check("multiply preset restored mode", window.state.mode == "multiply",
          f"mode={window.state.mode}")
    check("multiply preset restored empty params",
          window.state.params == {}, f"params={window.state.params}")

    # Restore a Canon mode + softness for the remaining round-trip checks.
    _select_mode("canon_bright")
    window.blend_controls.softness_row.slider.setValue(30)
    wait_until(lambda: render_settled(window))
    check("canon softness re-applied after preset detour",
          window.state.params.get("softness") == 30.0,
          f"params={window.state.params}")

    # -- 5. preset save / clear / load round-trip ---------------------------------
    preset_path = tmp / "roundtrip.bsp"
    window.save_preset_to(preset_path)
    check("preset file written", preset_path.exists())

    saved_ids_paths = [e.path for e in window.state.entries]
    window.state.clear()
    wait_until(lambda: not window.canvas.has_image())
    check("clear emptied the document",
          len(window.state.entries) == 0 and not window.canvas.has_image())

    load_report = window.load_preset_from(preset_path)
    ok = wait_until(lambda: render_settled(window) and window.canvas.has_image())
    check("preset restored 3 images",
          len(window.state.entries) == 3 and load_report.ok,
          f"n={len(window.state.entries)}")
    check("preset preserved image order",
          [e.path for e in window.state.entries] == saved_ids_paths)
    restored = next(
        (e for e in window.state.entries if e.path == p1.resolve()), None
    )
    check("preset preserved per-image exposure",
          restored is not None and restored.adjustments.exposure == 1.0,
          f"restored={restored.adjustments if restored else None}")
    check("preset preserved softness",
          window.state.params["softness"] == 30.0,
          f"params={window.state.params}")
    check("preview re-rendered after preset load", ok)

    # -- 7 (run before the cap fills the strip). full-res 16-bit TIFF export ------
    export_path = tmp / "export.tif"
    done: list[tuple[bool, str]] = []
    window.export_done.connect(lambda ok, msg: done.append((ok, msg)))
    started = window.export_to(export_path, "tiff")
    ok = wait_until(lambda: bool(done), timeout_ms=30000)
    check("export finished", started and ok and done and done[0][0],
          f"done={done}")
    check("export file exists", export_path.exists())
    import imageio.v3 as iio
    arr = iio.imread(export_path)
    # Smallest source by area is 512×384 → composite is (384, 512, 3).
    check("export dims match smallest source",
          arr.shape == (384, 512, 3), f"shape={arr.shape}")
    check("export is 16-bit", arr.dtype == np.uint16, f"dtype={arr.dtype}")

    # -- 8 / 9. solo, move, crop, export + presets with the new tools ------------
    check_solo(window, [p1, p2, p3])
    check_move(window, [p1, p2, p3])
    check_crop(window, [p1, p2, p3])
    check_export_and_presets(window, [p1, p2, p3], tmp)
    window.state.clear()
    settle(window)
    window.add_files([p1, p2, p3])
    settle(window)

    # -- 6. the 21st image is refused ------------------------------------------------
    extras = [
        make_image(tmp / f"extra_{i}.png", 64, 48, "png", 10 + i)
        for i in range(engine.MAX_IMAGES - 3)
    ]
    report = window.add_files(extras)
    check("filled to the 20-image cap",
          len(window.state.entries) == engine.MAX_IMAGES and report.ok,
          f"n={len(window.state.entries)}")
    twenty_first = make_image(tmp / "too_many.png", 64, 48, "png", 99)
    report = window.add_files([twenty_first])
    check("21st image refused",
          len(window.state.entries) == engine.MAX_IMAGES
          and len(report.refused_cap) == 1 and not report.added)
    check("refusal message is clear",
          window.last_notice is not None
          and str(engine.MAX_IMAGES) in window.last_notice[1]
          and "too_many.png" in window.last_notice[1],
          f"notice={window.last_notice}")
    wait_until(lambda: render_settled(window))

    window.close()
    QTest.qWait(50)

    failed = [name for name, ok, _ in _RESULTS if not ok]
    print("-" * 60)
    print(f"{len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks passed")
    if failed:
        print("FAILED: " + "; ".join(failed))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
