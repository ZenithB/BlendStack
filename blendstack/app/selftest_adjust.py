"""Offscreen checks for the knob, adjustments panel, blend controls, curve editor,
order panel and histogram.

Run with::

    QT_QPA_PLATFORM=offscreen python -m blendstack.app.selftest_adjust

Prints PASS/FAIL per check and exits non-zero if any check fails.
"""

from __future__ import annotations

import os
import sys
from dataclasses import fields, replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QModelIndex, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from blendstack.core.adjustments import (  # noqa: E402
    DEFAULT_ORDER,
    IDENTITY_CURVE,
    STAGES,
    Adjustments,
)
from blendstack.app import theme  # noqa: E402
from blendstack.app import adjustments_panel as _m_panel  # noqa: E402
from blendstack.app import blend_controls as _m_blend  # noqa: E402
from blendstack.app import curve_editor as _m_curve  # noqa: E402
from blendstack.app import histogram as _m_hist  # noqa: E402
from blendstack.app import knob as _m_knob  # noqa: E402
from blendstack.app import order_panel as _m_order  # noqa: E402
from blendstack.app.adjustments_panel import AdjustmentsPanel  # noqa: E402
from blendstack.app.blend_controls import BlendControls  # noqa: E402
from blendstack.app.curve_editor import HIT_RADIUS, MIN_X_GAP, CurveEditor  # noqa: E402
from blendstack.app.histogram import HistogramWidget  # noqa: E402
from blendstack.app.knob import Knob  # noqa: E402
from blendstack.app.order_panel import OrderPanel  # noqa: E402

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail and not condition else ""))


ID = tuple(tuple(p) for p in IDENTITY_CURVE)

# A deliberately "rich" Adjustments: every field differs from the default.
RICH = Adjustments(
    exposure=0.75,
    contrast=-20.0,
    saturation=35.0,
    sharpen_radius=12.5,
    sharpen_amount=80.0,
    opacity=60.0,
    denoise=42.0,
    curve_master=((0.0, 0.1), (0.5, 0.6), (1.0, 0.9)),
    curve_red=((0.0, 0.0), (0.3, 0.4), (1.0, 1.0)),
    curve_green=((0.0, 0.05), (1.0, 1.0)),
    curve_blue=((0.0, 0.0), (0.7, 0.5), (1.0, 1.0)),
    order=("sharpen", "denoise", "curves", "saturation", "contrast", "exposure"),
    move_x=0.125,
    move_y=-0.25,
    mute=True,
)


class Recorder:
    def __init__(self, signal) -> None:
        self.args: list = []
        signal.connect(lambda *a: self.args.append(a))

    @property
    def count(self) -> int:
        return len(self.args)

    def clear(self) -> None:
        self.args.clear()


def only_changed(a: Adjustments, b: Adjustments) -> set[str]:
    return {f.name for f in fields(Adjustments) if getattr(a, f.name) != getattr(b, f.name)}


# --- mouse helpers (send explicit QMouseEvents so button state is exact) ------

def _ev(kind, widget, pos: QPointF, button, buttons, mods=Qt.NoModifier):
    gp = widget.mapToGlobal(pos)
    return QMouseEvent(kind, pos, gp, button, buttons, mods)


def press(w, pos, button=Qt.LeftButton, mods=Qt.NoModifier):
    QApplication.sendEvent(w, _ev(QEvent.MouseButtonPress, w, pos, button, button, mods))


def move(w, pos, buttons=Qt.LeftButton, mods=Qt.NoModifier):
    QApplication.sendEvent(w, _ev(QEvent.MouseMove, w, pos, Qt.NoButton, buttons, mods))


def release(w, pos, button=Qt.LeftButton, mods=Qt.NoModifier):
    QApplication.sendEvent(w, _ev(QEvent.MouseButtonRelease, w, pos, button, Qt.NoButton, mods))


def dblclick(w, pos, button=Qt.LeftButton):
    QApplication.sendEvent(w, _ev(QEvent.MouseButtonDblClick, w, pos, button, button))


def wheel(w, dy: int, mods=Qt.NoModifier):
    pos = QPointF(w.width() / 2, w.height() / 2)
    ev = QWheelEvent(pos, w.mapToGlobal(pos), QPoint(0, 0), QPoint(0, dy),
                     Qt.NoButton, mods, Qt.NoScrollPhase, False)
    QApplication.sendEvent(w, ev)


def key(w, k, mods=Qt.NoModifier):
    QApplication.sendEvent(w, QKeyEvent(QEvent.KeyPress, k, mods))
    QApplication.sendEvent(w, QKeyEvent(QEvent.KeyRelease, k, mods))


def px(plot, x, y) -> QPointF:
    return plot.to_pixel(x, y)


# =============================================================================

def test_panel(app: QApplication) -> None:
    panel = AdjustmentsPanel()
    panel.setEnabled(True)
    panel.show()
    rec = Recorder(panel.adjustments_edited)

    # round trip of every field
    panel.set_values(RICH)
    got = panel.values()
    check("panel round-trip: values() == set_values()", got == RICH,
          f"diff={only_changed(got, RICH)}")
    for f in fields(Adjustments):
        check(f"panel round-trip field {f.name}",
              getattr(got, f.name) == getattr(RICH, f.name),
              f"{getattr(got, f.name)} != {getattr(RICH, f.name)}")
    check("set_values emits nothing", rec.count == 0, f"emitted {rec.count}")
    panel.set_values(Adjustments())
    panel.set_values(RICH)
    check("repeated set_values emits nothing", rec.count == 0)

    # off-grid values survive (not re-quantised by the knobs)
    odd = replace(RICH, exposure=0.123, contrast=-12.5, denoise=33.3)
    panel.set_values(odd)
    check("off-grid values preserved by values()",
          panel.values().exposure == 0.123 and panel.values().contrast == -12.5
          and panel.values().denoise == 33.3)
    panel.set_values(RICH)

    # each knob edit changes only its field, placement + mute preserved
    edits = {
        "denoise_row": ("denoise", 77, 77.0),
        "radius_row": ("sharpen_radius", 200, 20.0),
        "amount_row": ("sharpen_amount", 150, 150.0),
        "exposure_row": ("exposure", -150, -1.5),
        "contrast_row": ("contrast", 55, 55.0),
        "saturation_row": ("saturation", -60, -60.0),
        "opacity_row": ("opacity", 10, 10.0),
    }
    for row_name, (field, raw, expect) in edits.items():
        panel.set_values(RICH)
        rec.clear()
        getattr(panel, row_name).set_raw_from_user(raw)
        ok = rec.count == 1
        a = rec.args[-1][0] if rec.args else None
        check(f"{row_name} emits exactly once", ok, f"count={rec.count}")
        if a is not None:
            check(f"{row_name}: only '{field}' changed",
                  only_changed(a, RICH) == {field},
                  f"changed={only_changed(a, RICH)}")
            check(f"{row_name}: value {expect}",
                  abs(getattr(a, field) - expect) < 1e-9, f"{getattr(a, field)}")
            check(f"{row_name}: move_x/move_y/mute preserved",
                  (a.move_x, a.move_y, a.mute) == (RICH.move_x, RICH.move_y, RICH.mute))
    panel.set_values(RICH)
    check("sharpen radius knob min 0.5 / max 20.0",
          abs(panel.radius_row.minimum() * 0.1 - 0.5) < 1e-9
          and abs(panel.radius_row.maximum() * 0.1 - 20.0) < 1e-9)
    check("noise knob range 0..100",
          (panel.denoise_row.minimum(), panel.denoise_row.maximum()) == (0, 100))
    check("panel knobs are Knob instances",
          all(isinstance(getattr(panel, n), Knob) for n in edits))

    # placement survives a chain of edits and an emitted-state echo
    panel.set_values(RICH)
    rec.clear()
    panel.contrast_row.set_raw_from_user(10)
    panel.amount_row.set_raw_from_user(20)
    check("placement + mute preserved across consecutive edits",
          rec.args[-1][0].move_x == 0.125 and rec.args[-1][0].move_y == -0.25
          and rec.args[-1][0].mute is True)
    # mute / placement set from outside (state echo) survive later edits
    panel.set_values(replace(RICH, mute=False, move_x=0.5))
    rec.clear()
    panel.contrast_row.set_raw_from_user(-33)
    check("echoed mute/move survive an edit",
          rec.args[-1][0].mute is False and rec.args[-1][0].move_x == 0.5)

    # reset keeps placement, restores the rest
    panel.set_values(RICH)
    rec.clear()
    panel.reset()
    a = rec.args[-1][0] if rec.args else None
    check("reset() emits once", rec.count == 1)
    expect = replace(Adjustments(), move_x=0.125, move_y=-0.25, mute=True)
    check("reset() restores defaults, keeps move_x/move_y/mute", a == expect,
          f"diff={only_changed(a, expect) if a else None}")
    check("reset(): curves identity, order default, opacity 100, denoise 0",
          a is not None and a.curve_master == ID and a.curve_red == ID
          and a.curve_green == ID and a.curve_blue == ID
          and a.order == DEFAULT_ORDER and a.opacity == 100.0 and a.denoise == 0.0)
    check("reset(): widgets match",
          panel.values() == expect and panel.curve_editor.curves() == (ID, ID, ID, ID)
          and panel.order_panel.order() == DEFAULT_ORDER)
    rec.clear()
    panel.set_values(RICH)
    QTest.mouseClick(panel.reset_button, Qt.LeftButton)
    check("Reset button works", rec.count == 1 and rec.args[-1][0] == expect)

    # curve edits through the panel
    panel.set_values(RICH)
    rec.clear()
    panel.curve_editor.set_channel(1)
    idx = panel.curve_editor.add_point(1, 0.8, 0.9)
    a = rec.args[-1][0]
    check("curve edit emits Adjustments with only curve_red changed",
          rec.count == 1 and only_changed(a, RICH) == {"curve_red"}
          and (0.8, 0.9) in a.curve_red and idx >= 0)

    # order edit through the panel
    panel.set_values(RICH)
    rec.clear()
    panel.order_panel.move_item(0, 5)
    a = rec.args[-1][0] if rec.args else None
    exp_order = list(RICH.order)
    exp_order.append(exp_order.pop(0))
    check("panel emits Adjustments with the new order after move",
          rec.count == 1 and a is not None and a.order == tuple(exp_order)
          and only_changed(a, RICH) == {"order"},
          f"{a.order if a else None}")
    # via the model (what a real drop does)
    panel.set_values(RICH)
    rec.clear()
    lw = panel.order_panel.list
    lw.model().moveRow(QModelIndex(), 3, QModelIndex(), 0)
    a = rec.args[-1][0] if rec.args else None
    check("panel emits new order after a model row move (drop path)",
          rec.count == 1 and a is not None and set(a.order) == set(STAGES)
          and a.order[0] == RICH.order[3], f"{a.order if a else None}")

    # dimming follows values
    panel.set_values(Adjustments())
    op = panel.order_panel
    states = {op.row_item(i).data(Qt.UserRole): op.row_item(i).font().italic()
              for i in range(op.list.count())}
    check("identity adjustments: all stages dimmed", all(states.values()), str(states))
    panel.denoise_row.set_raw_from_user(30)
    states = {op.row_item(i).data(Qt.UserRole): op.row_item(i).font().italic()
              for i in range(op.list.count())}
    check("dimming updates after edit (denoise active, others dim)",
          states["denoise"] is False and all(v for k, v in states.items() if k != "denoise"),
          str(states))
    panel.set_values(RICH)
    states = {op.row_item(i).data(Qt.UserRole): op.row_item(i).font().italic()
              for i in range(op.list.count())}
    check("dimming after set_values: all stages active", not any(states.values()), str(states))

    # disabled panel stays rendered/readable; enabling works
    panel.setEnabled(False)
    check("panel can be disabled (before an image is selected)", not panel.exposure_row.isEnabled())
    panel.setEnabled(True)

    # compact mode shrinks the dials and the plot but keeps every contract
    h_norm = panel.sizeHint().height()
    panel.set_compact(True)
    check("compact mode: dials + plot shrink",
          panel.exposure_row.dial_size() == Knob.COMPACT
          and panel.curve_editor.plot.width() < 170 and panel.is_compact())
    panel.set_values(RICH)
    check("compact mode: round-trip still exact", panel.values() == RICH)
    panel.set_compact(False)
    check("normal mode restored", panel.exposure_row.dial_size() == Knob.NORMAL)
    # width-driven automatic compact mode (hysteresis-free: depends only on width)
    auto = AdjustmentsPanel()
    auto.show()
    auto.resize(auto.minimumSizeHint().width(), 250)
    app.processEvents()
    narrow = auto.is_compact()
    auto.resize(auto.sizeHint().width() + 20, 250)
    app.processEvents()
    wide = auto.is_compact()
    check("panel goes compact when narrower than its preferred width, normal when wide",
          narrow and not wide, f"narrow={narrow} wide={wide}")
    check("compact hint is shorter than the normal hint",
          auto.minimumSizeHint().height() < auto.sizeHint().height()
          and auto.minimumSizeHint().width() <= auto.sizeHint().width())
    auto.close()
    check("panel has no explicit min/max height set", panel.maximumHeight() > 10000
          and h_norm == panel.sizeHint().height())
    check("panel does not use a scroll area",
          not panel.findChildren(__import__("PySide6.QtWidgets", fromlist=["QScrollArea"]).QScrollArea))
    panel.close()


def test_curve_editor(app: QApplication) -> None:
    ed = CurveEditor()
    ed.show()
    app.processEvents()
    rec = Recorder(ed.curves_changed)
    plot = ed.plot
    check("plot is square", plot.width() == plot.height())
    check("plot is compact (160..176 px)", 160 <= plot.width() <= 176, f"{plot.width()}")
    check("plot drawing area ~160 px", 150 <= plot.plot_rect().width() <= 170,
          f"{plot.plot_rect().width()}")
    check("initial curves identity", ed.curves() == (ID, ID, ID, ID))

    # API: add / move / remove
    i = ed.add_point(0, 0.5, 0.7)
    check("add_point returns sorted index & emits", i == 1 and rec.count == 1
          and ed.curves()[0] == ((0.0, 0.0), (0.5, 0.7), (1.0, 1.0)))
    check("add_point too close in x is refused", ed.add_point(0, 0.5 + MIN_X_GAP / 2, 0.2) == -1
          and len(ed.curves()[0]) == 3)
    check("channel independence on add", ed.curves()[1:] == (ID, ID, ID))
    # move: clamp between neighbours
    ed.move_point(0, 1, 2.0, 5.0)
    p = ed.curves()[0][1]
    check("move_point clamps x below next neighbour & y to 1",
          p[0] < 1.0 and abs(p[0] - (1.0 - MIN_X_GAP)) < 1e-12 and p[1] == 1.0, str(p))
    ed.move_point(0, 1, -3.0, -1.0)
    p = ed.curves()[0][1]
    check("move_point clamps x above prev neighbour & y to 0",
          abs(p[0] - MIN_X_GAP) < 1e-12 and p[1] == 0.0, str(p))
    ed.move_point(0, 0, -1.0, 0.3)  # endpoint: x stays 0..next
    check("endpoint x stays >= 0, y free", ed.curves()[0][0] == (0.0, 0.3))
    ed.move_point(0, 0, 0.9, 0.3)   # endpoint cannot pass next point
    check("first endpoint cannot pass its neighbour",
          ed.curves()[0][0][0] < ed.curves()[0][1][0])
    ed.move_point(0, 2, 0.0, 0.3)   # last endpoint cannot go left of neighbour
    check("last endpoint cannot pass its neighbour",
          ed.curves()[0][2][0] > ed.curves()[0][1][0])
    check("move_point out-of-range index is a no-op", ed.move_point(0, 9, 0.5, 0.5) is False)
    # remove
    n0 = rec.count
    check("remove_point removes", ed.remove_point(0, 1) and len(ed.curves()[0]) == 2
          and rec.count == n0 + 1)
    n0 = rec.count
    check("cannot go below 2 points", not ed.remove_point(0, 0) and not ed.remove_point(0, 1)
          and len(ed.curves()[0]) == 2 and rec.count == n0)

    # set_curves silent; sanitises
    ed.reset_all()
    rec.clear()
    ed.set_curves(RICH.curve_master, RICH.curve_red, RICH.curve_green, RICH.curve_blue)
    check("set_curves is silent", rec.count == 0)
    check("curves() round-trips set_curves",
          ed.curves() == (RICH.curve_master, RICH.curve_red, RICH.curve_green, RICH.curve_blue))
    ed.set_curves([(0.5, 0.5)], [(0.2, 0.2), (0.2, 0.9), (0.8, 0.8)], [], [(1.5, 2.0), (-1, -1)])
    c = ed.curves()
    check("set_curves sanitises bad input into valid curves",
          c[0] == ID and len(c[1]) == 2 and c[2] == ID
          and c[3] == ((0.0, 0.0), (1.0, 1.0)), str(c))
    ed.reset_all()
    rec.clear()

    # --- mouse: click to add -------------------------------------------------
    ed.set_channel(0)
    target = px(plot, 0.5, 0.75)
    press(plot, target)
    release(plot, target)
    c = ed.curves()[0]
    check("click on empty plot adds a point", len(c) == 3 and rec.count >= 1, str(c))
    check("added point is near the click",
          abs(c[1][0] - 0.5) < 0.01 and abs(c[1][1] - 0.75) < 0.01, str(c[1]))

    # drag it
    start = px(plot, *c[1])
    press(plot, start)
    move(plot, px(plot, 0.45, 0.5))
    move(plot, px(plot, 0.30, 0.40))
    release(plot, px(plot, 0.30, 0.40))
    c = ed.curves()[0]
    check("drag moves the point", len(c) == 3 and abs(c[1][0] - 0.30) < 0.01
          and abs(c[1][1] - 0.40) < 0.01, str(c))
    check("dragging emitted continuously", rec.count >= 3, f"count={rec.count}")

    # drag beyond neighbour: ordering kept
    start = px(plot, *c[1])
    press(plot, start)
    move(plot, px(plot, 1.4, 0.4))   # far right, outside the plot
    release(plot, px(plot, 1.4, 0.4))
    c = ed.curves()[0]
    xs = [p[0] for p in c]
    check("drag past neighbour keeps x strictly increasing",
          all(b > a for a, b in zip(xs, xs[1:])) and len(c) == 3, str(c))

    # hit radius: a press just inside HIT_RADIUS drags, outside adds
    ed.reset_all()
    ed.add_point(0, 0.5, 0.5)
    pt = plot.to_pixel(0.5, 0.5)
    near = QPointF(pt.x() + HIT_RADIUS - 1.5, pt.y())
    press(plot, near)
    release(plot, near)
    check("press inside hit radius selects, does not add", len(ed.curves()[0]) == 3)
    far = QPointF(pt.x() + HIT_RADIUS + 12, pt.y())
    press(plot, far)
    release(plot, far)
    check("press outside hit radius adds a point", len(ed.curves()[0]) == 4)

    # right click removes
    ed.reset_all()
    ed.add_point(0, 0.5, 0.5)
    pt = px(plot, 0.5, 0.5)
    press(plot, pt, Qt.RightButton)
    release(plot, pt, Qt.RightButton)
    check("right-click on a point removes it", len(ed.curves()[0]) == 2)
    # right-click on endpoints with only 2 points never removes
    for e in (px(plot, 0, 0), px(plot, 1, 1)):
        press(plot, e, Qt.RightButton)
        release(plot, e, Qt.RightButton)
    check("right-click cannot go below 2 points", len(ed.curves()[0]) == 2)
    # right-click on empty plot does nothing
    e = px(plot, 0.3, 0.9)
    press(plot, e, Qt.RightButton)
    release(plot, e, Qt.RightButton)
    check("right-click on empty area does nothing", ed.curves()[0] == ID)

    # double click removes (QTest.mouseDClick: press, release, dblclick, release)
    ed.reset_all()
    ed.add_point(0, 0.5, 0.5)
    pt = plot.to_pixel(0.5, 0.5).toPoint()
    QTest.mouseDClick(plot, Qt.LeftButton, Qt.NoModifier, pt)
    check("double-click on a point removes it", len(ed.curves()[0]) == 2,
          str(ed.curves()[0]))
    pt = plot.to_pixel(0.0, 0.0).toPoint()
    QTest.mouseDClick(plot, Qt.LeftButton, Qt.NoModifier, pt)
    check("double-click cannot go below 2 points", len(ed.curves()[0]) == 2)

    # QTest click path as well
    ed.reset_all()
    QTest.mouseClick(plot, Qt.LeftButton, Qt.NoModifier, plot.to_pixel(0.25, 0.6).toPoint())
    check("QTest.mouseClick adds a point", len(ed.curves()[0]) == 3)

    # hover cursor + readout
    ed.reset_all()
    ed.add_point(0, 0.5, 0.5)
    pt = px(plot, 0.5, 0.5)
    move(plot, pt, Qt.NoButton)
    check("hover over a point -> pointing-hand cursor",
          plot.cursor().shape() == Qt.PointingHandCursor)
    check("hover readout shows in -> out in 0-255",
          ed._readout.text().count("128") == 2, ed._readout.text())
    move(plot, px(plot, 0.2, 0.8), Qt.NoButton)
    check("hover off a point -> cross cursor", plot.cursor().shape() == Qt.CrossCursor)

    # channel independence & selection via tab buttons
    ed.reset_all()
    rec.clear()
    QTest.mouseClick(ed._tabs[1], Qt.LeftButton)
    check("tab button selects channel R", ed.channel() == 1)
    t = px(plot, 0.4, 0.2)
    press(plot, t)
    release(plot, t)
    m, r, g, b = ed.curves()
    check("editing R does not change master/G/B",
          m == ID and g == ID and b == ID and len(r) == 3, str(ed.curves()))
    ed.set_channel(2)
    ed.add_point(2, 0.6, 0.6)
    m2, r2, g2, b2 = ed.curves()
    check("editing G leaves master/R/B alone", m2 == ID and r2 == r and b2 == ID and len(g2) == 3)
    ed.set_channel(3)
    ed.add_point(3, 0.1, 0.3)
    check("set_channel out of range clamps", (ed.set_channel(99), ed.channel())[1] == 3)

    # reset buttons
    ed.set_channel(1)
    rec.clear()
    QTest.mouseClick(ed.reset_channel_button, Qt.LeftButton)
    m, r, g, b = ed.curves()
    check("Reset channel resets only the selected channel",
          r == ID and len(g) == 3 and len(b) == 3 and rec.count == 1)
    QTest.mouseClick(ed.reset_all_button, Qt.LeftButton)
    check("Reset all curves", ed.curves() == (ID, ID, ID, ID))

    # validity of anything the editor can produce: fuzz sequence
    import random

    rng = random.Random(7)
    ed.reset_all()
    ok = True
    for step in range(400):
        ch = rng.randrange(4)
        ed.set_channel(ch)
        op = rng.choice(["add", "move", "move", "remove", "mouse", "mouse_drag", "rclick"])
        n = len(ed.curves()[ch])
        if op == "add":
            ed.add_point(ch, rng.uniform(-0.2, 1.2), rng.uniform(-0.2, 1.2))
        elif op == "move":
            ed.move_point(ch, rng.randrange(n + 1), rng.uniform(-0.5, 1.5), rng.uniform(-0.5, 1.5))
        elif op == "remove":
            ed.remove_point(ch, rng.randrange(n + 1))
        elif op == "mouse":
            q = px(plot, rng.uniform(0, 1), rng.uniform(0, 1))
            press(plot, q)
            release(plot, q)
        elif op == "mouse_drag":
            pts = ed.curves()[ch]
            q = px(plot, *rng.choice(pts))
            press(plot, q)
            for _ in range(3):
                move(plot, px(plot, rng.uniform(-0.2, 1.2), rng.uniform(-0.2, 1.2)))
            release(plot, q)
        else:
            q = px(plot, *rng.choice(ed.curves()[ch]))
            press(plot, q, Qt.RightButton)
            release(plot, q, Qt.RightButton)
        try:
            Adjustments(
                curve_master=ed.curves()[0], curve_red=ed.curves()[1],
                curve_green=ed.curves()[2], curve_blue=ed.curves()[3],
            )
        except ValueError as exc:
            ok = False
            print("  invalid curves after", op, exc, ed.curves())
            break
    check("400 random edits always produce core-valid curves", ok)
    check("every curve keeps >= 2 points", all(len(c) >= 2 for c in ed.curves()))
    ed.close()


def test_order_panel(app: QApplication) -> None:
    op = OrderPanel()
    op.show()
    rec = Recorder(op.order_changed)
    check("default order", op.order() == DEFAULT_ORDER)
    labels = [op.row_item(i).text() for i in range(op.list.count())]
    check("rows listed with their labels",
          labels == ["Exposure", "Contrast", "Curves", "Saturation", "Noise removal", "Sharpen"],
          str(labels))
    check("chip rows are ~20 px high", all(
        op.row_item(i).sizeHint().height() == 20 for i in range(6)))
    check("ids stored in UserRole",
          tuple(op.row_item(i).data(Qt.UserRole) for i in range(6)) == STAGES)
    check("hint text", "drag to reorder" in op.title_label.text())

    op.move_item(0, 3)
    exp = list(DEFAULT_ORDER)
    exp.insert(3, exp.pop(0))
    check("move_item(0,3) gives expected permutation", op.order() == tuple(exp)
          and rec.count == 1, str(op.order()))
    op.move_item(5, 0)
    exp.insert(0, exp.pop(5))
    check("move_item(5,0) gives expected permutation", op.order() == tuple(exp))
    n = rec.count
    op.move_item(2, 2)
    op.move_item(-1, 2)
    check("move_item no-ops emit nothing", rec.count == n)
    check("order() is always a permutation of STAGES",
          sorted(op.order()) == sorted(STAGES) and len(op.order()) == 6)

    # drop path: model moveRow (what InternalMove does) keeps the id mapping
    op.set_order(DEFAULT_ORDER)
    rec.clear()
    op.list.model().moveRow(QModelIndex(), 1, QModelIndex(), 5)
    # Qt moveRow(src, dstChild): moving row 1 before row 5 -> lands at index 4
    got = op.order()
    check("model row move (drop path) emits and keeps a valid permutation",
          rec.count == 1 and sorted(got) == sorted(STAGES) and got != DEFAULT_ORDER, str(got))
    check("row text and id stay paired after a move",
          all(op.row_item(i).text().endswith(
              {"exposure": "Exposure", "contrast": "Contrast", "curves": "Curves",
               "saturation": "Saturation", "denoise": "Noise removal",
               "sharpen": "Sharpen"}[op.row_item(i).data(Qt.UserRole)])
              for i in range(6)))

    # a corrupted list (duplicate row) is repaired, never reported as an order
    rec.clear()
    op.set_order(DEFAULT_ORDER)
    op.list.addItem("bogus")  # simulate a drop that duplicated/added a row
    op.list.item(op.list.count() - 1).setData(Qt.UserRole, "exposure")
    check("invalid list is not reported as an order", op.order() == DEFAULT_ORDER)
    op.set_order(DEFAULT_ORDER)
    check("set_order repairs the list to 6 rows", op.list.count() == 6)

    # set_order silent + normalising
    rec.clear()
    op.set_order(("sharpen", "bogus", "sharpen", "curves"))
    check("set_order is silent", rec.count == 0)
    check("set_order normalises to a full permutation",
          op.order() == ("sharpen", "curves", "exposure", "contrast", "saturation", "denoise"),
          str(op.order()))

    # reset
    rec.clear()
    op.reset_order()
    check("reset_order restores DEFAULT_ORDER and emits",
          op.order() == DEFAULT_ORDER and rec.count == 1)
    op.reset_order()
    check("reset_order when already default emits nothing", rec.count == 1)
    op.set_order(("sharpen",) + DEFAULT_ORDER[:-1])
    rec.clear()
    QTest.mouseClick(op.reset_button, Qt.LeftButton)
    check("Reset order button", op.order() == DEFAULT_ORDER and rec.count == 1)

    # dimming
    op.set_active_stages({"exposure", "sharpen"})
    dim = {op.row_item(i).data(Qt.UserRole): op.row_item(i).font().italic()
           for i in range(6)}
    check("dimming reflects set_active_stages",
          dim == {"exposure": False, "contrast": True, "curves": True,
                  "saturation": True, "denoise": True, "sharpen": False}, str(dim))
    check("dimmed rows get a foreground brush, active rows none",
          op.row_item(1).data(Qt.ForegroundRole) is not None
          and op.row_item(0).data(Qt.ForegroundRole) is None)
    op.move_item(0, 4)
    dim = {op.row_item(i).data(Qt.UserRole): op.row_item(i).font().italic()
           for i in range(6)}
    check("dimming survives reordering", dim["exposure"] is False and dim["curves"] is True)
    op.set_active_stages(set(STAGES))
    check("all active -> nothing dimmed",
          not any(op.row_item(i).font().italic() for i in range(6)))
    op.close()


def _knob_center(k: Knob) -> QPointF:
    return QPointF(k.width() / 2.0, k._TOP + k.dial_size() / 2.0)


def _drag(k: Knob, dy: float = 0.0, dx: float = 0.0, mods=Qt.NoModifier, steps: int = 5) -> None:
    """Press on the dial, move in ``steps`` increments, release."""
    c = _knob_center(k)
    press(k, c, mods=mods)
    for i in range(1, steps + 1):
        move(k, QPointF(c.x() + dx * i / steps, c.y() + dy * i / steps), mods=mods)
    release(k, QPointF(c.x() + dx, c.y() + dy), mods=mods)


def test_knob(app: QApplication) -> None:
    exp = Knob("EXPOSURE", -300, 300, scale=0.01, decimals=2, suffix=" EV")
    rad = Knob("SHARPEN RADIUS", 5, 200, scale=0.1, decimals=1, suffix=" px", default=1.0)
    opa = Knob("OPACITY", 0, 100, suffix=" %", default=100.0)
    soft = Knob("SOFTNESS", 0, 100)
    bias = Knob("BIAS", -100, 100)
    for k in (exp, rad, opa, soft, bias):
        k.show()
    app.processEvents()

    # raw <-> real mapping -----------------------------------------------------
    exp.set_raw(-300)
    check("exposure raw -300 -> -3.00 EV", exp.value() == -3.0 and exp.raw() == -300)
    exp.set_raw(300)
    check("exposure raw 300 -> +3.00 EV", exp.value() == 3.0)
    exp.set_value(0.57)
    check("exposure 0.57 -> raw 57 -> 0.57 (clean float)", exp.raw() == 57 and exp.value() == 0.57,
          f"{exp.raw()} {exp.value()!r}")
    check("exposure matches() lands on the step",
          exp.matches(0.57) and exp.matches(0.5700001) and not exp.matches(0.58))
    exp.set_value(9.0)
    check("exposure set_value clamps to +3", exp.value() == 3.0)
    exp.set_value(-9.0)
    check("exposure set_value clamps to -3", exp.value() == -3.0)
    exp.set_value(float("nan"))
    check("NaN is ignored", exp.value() == -3.0)
    exp.set_value(0.5)
    check("exposure readout '+0.50 EV'", exp.format_value() == "+0.50 EV", exp.format_value())
    exp.set_value(0.0)
    check("zero readout has no minus sign", exp.format_value() == "+0.00 EV", exp.format_value())
    check("radius range 0.5..20.0 (raw 5..200)",
          rad.minimum() == 5 and rad.maximum() == 200
          and abs(rad.minimum() * rad.scale() - 0.5) < 1e-9
          and abs(rad.maximum() * rad.scale() - 20.0) < 1e-9)
    rad.set_value(12.5)
    check("radius 12.5 -> raw 125", rad.raw() == 125 and rad.value() == 12.5)
    rad.set_raw(30)
    check("radius raw 30 -> exactly 3.0", rad.value() == 3.0, repr(rad.value()))
    rad.set_value(0.1)
    check("radius clamps to 0.5", rad.value() == 0.5)
    rad.set_value(99.0)
    check("radius clamps to 20.0", rad.value() == 20.0)
    check("radius readout '12.5 px'", (rad.set_value(12.5), rad.format_value())[1] == "12.5 px")
    opa.set_value(60.0)
    check("opacity 60 -> '60 %' unsigned", opa.value() == 60.0 and opa.format_value() == "60 %")
    bias.set_value(-20)
    check("bias readout signed '-20'", bias.format_value() == "-20", bias.format_value())

    # bipolar / unipolar / default ------------------------------------------------
    check("bipolar flag: exposure/bias yes; opacity/softness/radius no",
          exp.is_bipolar() and bias.is_bipolar()
          and not opa.is_bipolar() and not soft.is_bipolar() and not rad.is_bipolar())
    check("default: spanning range -> 0; unipolar -> minimum; explicit wins",
          exp.default() == 0.0 and bias.default() == 0.0 and soft.default() == 0.0
          and rad.default() == 1.0 and opa.default() == 100.0
          and Knob("R", 5, 200, scale=0.1).default() == 0.5)
    check("fresh knob starts at 0 / minimum",
          Knob("E", -300, 300, scale=0.01).value() == 0.0 and Knob("R", 5, 200, scale=0.1).value() == 0.5)

    # silent vs user ------------------------------------------------------------
    rec = Recorder(exp.valueChanged)
    exp.set_value(1.25)
    exp.set_raw(10)
    check("set_value / set_raw are silent", rec.count == 0)
    exp.set_value_from_user(1.5)
    check("set_value_from_user emits exactly once, in real units",
          rec.count == 1 and rec.args[0] == (1.5,) and exp.value() == 1.5, str(rec.args))
    exp.set_value_from_user(1.5)
    check("no change -> no emit", rec.count == 1)
    exp.set_raw_from_user(-20)
    check("set_raw_from_user emits real value", rec.count == 2 and rec.args[-1] == (-0.2,),
          str(rec.args))
    check("slider shim drives the knob like a user edit",
          (exp.slider.setValue(40), rec.count == 3 and exp.raw() == 40 and exp.slider.value() == 40
           and exp.slider.minimum() == -300 and exp.slider.maximum() == 300)[1])

    # tooltip -----------------------------------------------------------------------
    exp.set_value(0.5)
    check("tooltip: label, value, reset hint",
          "EXPOSURE" in exp.toolTip() and "+0.50 EV" in exp.toolTip()
          and "double-click to reset" in exp.toolTip(), exp.toolTip())

    # drag ---------------------------------------------------------------------------
    exp.set_value(0.0)
    rec = Recorder(exp.valueChanged)
    _drag(exp, dy=-50)  # 50 px up
    check("drag up raises the value (50 px ~ 1.5 EV)",
          abs(exp.value() - 1.5) < 0.04, f"{exp.value()}")
    check("drag emits continuously (>1 signal)", rec.count > 1, f"{rec.count}")
    check("not dragging after release", not exp.is_dragging())
    exp.set_value(0.0)
    _drag(exp, dy=+50)
    check("drag down lowers the value", abs(exp.value() + 1.5) < 0.04, f"{exp.value()}")
    exp.set_value(0.0)
    _drag(exp, dx=+30)
    check("drag right raises the value too", exp.value() > 0.8, f"{exp.value()}")
    exp.set_value(0.0)
    _drag(exp, dy=-900)
    check("drag clamps at the maximum", exp.value() == 3.0, f"{exp.value()}")
    # clamped accumulator: reversing takes effect immediately
    c = _knob_center(exp)
    exp.set_value(0.0)
    press(exp, c)
    move(exp, QPointF(c.x(), c.y() - 400))
    move(exp, QPointF(c.x(), c.y() - 390))   # 10 px back down
    release(exp, QPointF(c.x(), c.y() - 390))
    check("reversing after clamping responds immediately (no dead zone)",
          exp.value() < 3.0, f"{exp.value()}")
    exp.set_value(0.0)
    _drag(exp, dy=+900)
    check("drag clamps at the minimum", exp.value() == -3.0, f"{exp.value()}")
    # Shift = fine (x0.1)
    exp.set_value(0.0)
    _drag(exp, dy=-50, mods=Qt.ShiftModifier)
    check("Shift-drag is ~10x finer", abs(exp.value() - 0.15) < 0.02, f"{exp.value()}")
    # integer knob, 200 px for the full range
    soft.set_value(0)
    _drag(soft, dy=-100)
    check("opacity-style knob: 100 px = half range", abs(soft.value() - 50) <= 1, f"{soft.value()}")
    # right-button press does not drag
    soft.set_value(10)
    c = _knob_center(soft)
    press(soft, c, Qt.RightButton)
    move(soft, QPointF(c.x(), c.y() - 40), Qt.RightButton)
    release(soft, c, Qt.RightButton)
    check("right button does not change the value", soft.value() == 10)

    # wheel -----------------------------------------------------------------------------
    exp.set_value(0.0)
    rec = Recorder(exp.valueChanged)
    wheel(exp, 120)
    check("wheel up = +1 step", exp.raw() == 1 and rec.count == 1)
    wheel(exp, -120)
    wheel(exp, -120)
    check("wheel down = -1 step each", exp.raw() == -1)
    wheel(exp, 120, Qt.ShiftModifier)
    check("Shift+wheel = x10 step", exp.raw() == 9, f"{exp.raw()}")
    wheel(exp, 40)
    check("sub-notch wheel delta does not move yet", exp.raw() == 9)
    soft.set_value(100)
    wheel(soft, 120)
    check("wheel clamps at max", soft.value() == 100)

    # keys ---------------------------------------------------------------------------------
    exp.set_value(0.0)
    rec = Recorder(exp.valueChanged)
    key(exp, Qt.Key_Up)
    key(exp, Qt.Key_Right)
    check("Up / Right = +1 step", exp.raw() == 2 and rec.count == 2)
    key(exp, Qt.Key_Down)
    key(exp, Qt.Key_Left)
    key(exp, Qt.Key_Left)
    check("Down / Left = -1 step", exp.raw() == -1)
    key(exp, Qt.Key_PageUp)
    check("PageUp = +10 steps", exp.raw() == 9)
    key(exp, Qt.Key_PageDown)
    key(exp, Qt.Key_PageDown)
    check("PageDown = -10 steps", exp.raw() == -11)
    key(exp, Qt.Key_Home)
    check("Home = minimum", exp.value() == -3.0)
    key(exp, Qt.Key_End)
    check("End = maximum", exp.value() == 3.0)
    check("focus policy is strong (Tab / click)", exp.focusPolicy() == Qt.StrongFocus)

    # double-click reset ---------------------------------------------------------------------
    opa.set_value(40.0)
    rec = Recorder(opa.valueChanged)
    c = _knob_center(opa)
    press(opa, c)
    release(opa, c)
    dblclick(opa, c)
    release(opa, c)
    check("double-click resets to the default (opacity -> 100) and emits once",
          opa.value() == 100.0 and rec.count == 1 and rec.args[0] == (100.0,), str(rec.args))
    c = _knob_center(exp)
    exp.set_value(1.0)
    dblclick(exp, c)
    check("double-click resets exposure to 0", exp.value() == 0.0)
    rec = Recorder(exp.valueChanged)
    exp.reset()
    check("reset() at default emits nothing", rec.count == 0)
    rad.set_value(8.0)
    rad.reset()
    check("reset() radius -> 1.0", rad.value() == 1.0)

    # rendering never crashes, enabled / disabled / focused / dragging ------------------------
    ok = True
    for k in (exp, rad, opa, soft, bias):
        for state in ("normal", "disabled", "focus", "compact"):
            k.setEnabled(state != "disabled")
            if state == "focus":
                k.setFocus(Qt.TabFocusReason)
            if state == "compact":
                k.set_dial_size(Knob.COMPACT)
            ok &= not k.grab().isNull()
        k.set_dial_size(Knob.NORMAL)
        k.setEnabled(True)
    check("knobs render in every state", ok)
    sh = exp.sizeHint()
    check("knob sizeHint ~ (dial+14.., dial+34)",
          sh.height() == Knob.NORMAL + 34 and sh.width() >= Knob.NORMAL + 14, f"{sh}")
    exp.set_dial_size(Knob.COMPACT)
    check("compact knob is shorter", exp.sizeHint().height() == Knob.COMPACT + 34)
    check("Knob.COMPACT < Knob.NORMAL", Knob.COMPACT < Knob.NORMAL and Knob.NORMAL >= 52)
    for k in (exp, rad, opa, soft, bias):
        k.close()


def test_blend(app: QApplication) -> None:
    bc = BlendControls()
    bc.show()
    app.processEvents()
    modes = Recorder(bc.mode_changed)
    params = Recorder(bc.param_changed)

    bc.set_from_state("canon_bright", {"softness": 30.0, "bias": -20.0, "basis": "luminance"})
    check("set_from_state emits nothing", modes.count == 0 and params.count == 0)
    check("set_from_state sets the knobs and the basis pill",
          bc.softness_row.value() == 30.0 and bc.bias_row.value() == -20.0
          and bc.luminance_radio.isChecked() and not bc.per_channel_radio.isChecked())
    check("canon mode shows softness / bias / basis, hides the hint",
          bc.softness_row.isVisible() and bc.bias_row.isVisible() and bc.basis_row.isVisible()
          and not bc.hint_label.isVisible())
    bc.set_from_state("multiply", {})
    check("continuous mode hides softness / bias / basis",
          not bc.softness_row.isVisible() and not bc.bias_row.isVisible()
          and not bc.basis_row.isVisible())
    check("continuous mode shows the dim hint", bc.hint_label.isVisible()
          and "opacity" in bc.hint_label.text().lower())
    check("mode combo shows only the mode (still visible)", bc.mode_combo.isVisible())
    for name in ("average", "screen", "grain_merge", "overlay"):
        bc.set_from_state(name, {})
        if bc.softness_row.isVisible() or bc.basis_row.isVisible():
            check(f"mode {name} hides parameter widgets", False)
            break
    else:
        check("all 5 continuous modes hide parameter widgets", True)
    bc.set_from_state("canon_dark", {})
    check("canon_dark shows them again", bc.softness_row.isVisible() and bc.bias_row.isVisible()
          and bc.basis_row.isVisible())
    check("basis defaults to per-channel", bc.per_channel_radio.isChecked())

    # signals
    bc.softness_row.set_raw_from_user(45)
    check("softness knob -> param_changed('softness', 45.0)",
          params.args[-1] == ("softness", 45.0), str(params.args))
    bc.bias_row.set_raw_from_user(-70)
    check("bias knob -> param_changed('bias', -70.0)", params.args[-1] == ("bias", -70.0))
    params.clear()
    QTest.mouseClick(bc.luminance_radio, Qt.LeftButton)
    check("LUMA pill -> param_changed('basis', 'luminance') once",
          params.count == 1 and params.args[0] == ("basis", "luminance"), str(params.args))
    check("LUMA is checked, PER-CH is not",
          bc.luminance_radio.isChecked() and not bc.per_channel_radio.isChecked())
    params.clear()
    QTest.mouseClick(bc.per_channel_radio, Qt.LeftButton)
    check("PER-CH pill -> param_changed('basis', 'per_channel') once",
          params.count == 1 and params.args[0] == ("basis", "per_channel"))
    params.clear()
    bc.luminance_radio.setChecked(True)
    check("radio shim: setChecked(True) emits basis once", params.args == [("basis", "luminance")])
    bc.per_channel_radio.setChecked(True)
    got: list = []
    bc.per_channel_radio.toggled.connect(got.append)
    bc.luminance_radio.setChecked(True)
    check("radio shim: toggled signal fires", got == [False])
    check("basis pill is exclusive", [bc.per_channel_radio.isChecked(),
                                      bc.luminance_radio.isChecked()] == [False, True])
    # mode combo
    modes.clear()
    bc.mode_combo.setCurrentIndex(bc.mode_combo.findData("average"))
    check("mode combo emits mode_changed(name)", modes.args == [("average",)], str(modes.args))
    check("combo items carry registry names", bc.mode_combo.findData("canon_bright") >= 0)
    # knob defaults
    bc.softness_row.set_value(50)
    bc.softness_row.reset()
    check("softness double-click default = 0", bc.softness_row.value() == 0.0)
    bc.set_compact(True)
    check("blend compact mode shrinks knobs", bc.softness_row.dial_size() == Knob.COMPACT)
    bc.set_compact(False)
    check("blend grab renders", not bc.grab().isNull())
    bc.close()


def test_histogram(app: QApplication) -> None:
    import numpy as np

    h = HistogramWidget()
    h.show()
    check("empty histogram has no data", not h.has_data())
    check("empty histogram renders", not h.grab().isNull())
    x = np.arange(256)
    data = np.stack([np.exp(-((x - m) / 40.0) ** 2) * 1000 for m in (80, 110, 150, 120)]).astype(int)
    h.set_data(data)
    check("histogram has data", h.has_data() and not h.grab().isNull())
    try:
        h.set_data(np.zeros((3, 256)))
        bad = False
    except ValueError:
        bad = True
    check("wrong-shaped histogram is rejected", bad)
    h.set_data(None)
    check("set_data(None) clears", not h.has_data())
    h.close()


def test_sizes_and_contrast(app: QApplication) -> None:
    ap = AdjustmentsPanel().sizeHint()
    check("AdjustmentsPanel sizeHint height <= 250", ap.height() <= 250, f"{ap}")
    check("AdjustmentsPanel sizeHint width <= 840", ap.width() <= 840, f"{ap}")
    check("AdjustmentsPanel minimumSizeHint <= sizeHint",
          AdjustmentsPanel().minimumSizeHint().width() <= ap.width()
          and AdjustmentsPanel().minimumSizeHint().height() <= ap.height())
    bc = BlendControls().sizeHint()
    check("BlendControls sizeHint <= 210 x 240", bc.width() <= 210 and bc.height() <= 240, f"{bc}")
    hs = HistogramWidget()
    check("HistogramWidget sizeHint ~ 200x100",
          (hs.sizeHint().width(), hs.sizeHint().height()) == (200, 100), f"{hs.sizeHint()}")
    check("HistogramWidget imposes no large minimum height",
          hs.minimumSizeHint().height() <= 60 and hs.minimumHeight() == 0)
    # at 1280 the three fit at their preferred sizes; at 1200 they fit at their
    # minimum sizes (the histogram flexes), with 10 px gutters/spacing
    total = bc.width() + ap.width() + hs.sizeHint().width() + 4 * 10
    check("rack fits 1280 px wide at preferred sizes", total <= 1280, f"{total}")
    small = (BlendControls().minimumSizeHint().width() + AdjustmentsPanel().minimumSizeHint().width()
             + hs.minimumSizeHint().width() + 4 * 10)
    check("rack fits 1200 px wide at minimum sizes", small <= 1200, f"{small}")

    # every foreground/background pair hard-wired in these widgets >= 4.5:1
    pairs = []
    for mod in (_m_knob, _m_panel if hasattr(_m_panel, "contrast_pairs") else None,
                _m_blend, _m_curve, _m_order, _m_hist):
        if mod is not None and hasattr(mod, "contrast_pairs"):
            pairs.extend(mod.contrast_pairs())
    check("contrast pairs were collected from every widget module", len(pairs) >= 25, f"{len(pairs)}")
    for name, fg, bg in pairs:
        r = theme.contrast_ratio(fg, bg)
        check(f"contrast >= 4.5: {name}", r >= 4.5, f"{r:.2f}:1 ({fg} on {bg})")
    # the theme's own guard still holds for the roles used here
    for name, fg, bg, minimum in theme.text_pairs():
        r = theme.contrast_ratio(fg, bg)
        check(f"theme pair {name}", r >= minimum, f"{r:.2f}")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply(app)
    test_knob(app)
    test_panel(app)
    test_blend(app)
    test_curve_editor(app)
    test_order_panel(app)
    test_histogram(app)
    test_sizes_and_contrast(app)
    failed = [n for n, ok, _ in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
