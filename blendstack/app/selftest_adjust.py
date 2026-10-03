"""Offscreen checks for the adjustments panel, curve editor and order panel.

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
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from blendstack.core.adjustments import (  # noqa: E402
    DEFAULT_ORDER,
    IDENTITY_CURVE,
    STAGES,
    Adjustments,
)
from blendstack.app.adjustments_panel import AdjustmentsPanel  # noqa: E402
from blendstack.app.curve_editor import HIT_RADIUS, MIN_X_GAP, CurveEditor  # noqa: E402
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

def _ev(kind, widget, pos: QPointF, button, buttons):
    gp = widget.mapToGlobal(pos)
    return QMouseEvent(kind, pos, gp, button, buttons, Qt.NoModifier)


def press(w, pos, button=Qt.LeftButton):
    QApplication.sendEvent(w, _ev(QEvent.MouseButtonPress, w, pos, button, button))


def move(w, pos, buttons=Qt.LeftButton):
    QApplication.sendEvent(w, _ev(QEvent.MouseMove, w, pos, Qt.NoButton, buttons))


def release(w, pos, button=Qt.LeftButton):
    QApplication.sendEvent(w, _ev(QEvent.MouseButtonRelease, w, pos, button, Qt.NoButton))


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

    # off-grid values survive (not re-quantised by the sliders)
    odd = replace(RICH, exposure=0.123, contrast=-12.5, denoise=33.3)
    panel.set_values(odd)
    check("off-grid values preserved by values()",
          panel.values().exposure == 0.123 and panel.values().contrast == -12.5
          and panel.values().denoise == 33.3)
    panel.set_values(RICH)

    # each slider edit changes only its field, placement preserved
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
        getattr(panel, row_name).slider.setValue(raw)
        ok = rec.count == 1
        a = rec.args[-1][0] if rec.args else None
        check(f"{row_name} emits exactly once", ok, f"count={rec.count}")
        if a is not None:
            check(f"{row_name}: only '{field}' changed",
                  only_changed(a, RICH) == {field},
                  f"changed={only_changed(a, RICH)}")
            check(f"{row_name}: value {expect}",
                  abs(getattr(a, field) - expect) < 1e-9, f"{getattr(a, field)}")
            check(f"{row_name}: move_x/move_y preserved",
                  (a.move_x, a.move_y) == (RICH.move_x, RICH.move_y))
    panel.set_values(RICH)
    check("sharpen radius slider min 0.5 / max 20.0",
          abs(panel.radius_row.slider.minimum() * 0.1 - 0.5) < 1e-9
          and abs(panel.radius_row.slider.maximum() * 0.1 - 20.0) < 1e-9)
    check("noise slider range 0..100",
          (panel.denoise_row.slider.minimum(), panel.denoise_row.slider.maximum()) == (0, 100))

    # placement survives a chain of edits and an emitted-state echo
    panel.set_values(RICH)
    rec.clear()
    panel.contrast_row.slider.setValue(10)
    panel.amount_row.slider.setValue(20)
    check("placement preserved across consecutive edits",
          rec.args[-1][0].move_x == 0.125 and rec.args[-1][0].move_y == -0.25)

    # reset keeps placement, restores the rest
    panel.set_values(RICH)
    rec.clear()
    panel.reset()
    a = rec.args[-1][0] if rec.args else None
    check("reset() emits once", rec.count == 1)
    expect = replace(Adjustments(), move_x=0.125, move_y=-0.25)
    check("reset() restores defaults, keeps move_x/move_y", a == expect,
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
    panel.denoise_row.slider.setValue(30)
    states = {op.row_item(i).data(Qt.UserRole): op.row_item(i).font().italic()
              for i in range(op.list.count())}
    check("dimming updates after edit (denoise active, others dim)",
          states["denoise"] is False and all(v for k, v in states.items() if k != "denoise"),
          str(states))
    panel.set_values(RICH)
    states = {op.row_item(i).data(Qt.UserRole): op.row_item(i).font().italic()
              for i in range(op.list.count())}
    check("dimming after set_values: all stages active", not any(states.values()), str(states))

    # width fits the 330 px right column
    w = panel.minimumSizeHint().width()
    check("panel minimumSizeHint().width() <= 300", w <= 300, f"{w}")
    # embedded in a scroll area (as in the main window) the panel's height
    # must not force a tall minimum on its host
    from PySide6.QtWidgets import QScrollArea, QWidget, QVBoxLayout

    host = QWidget()
    hl = QVBoxLayout(host)
    hl.setContentsMargins(0, 0, 0, 0)
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    inner = AdjustmentsPanel()
    inner.setEnabled(True)
    sc.setWidget(inner)
    hl.addWidget(sc)
    host.setFixedWidth(330)
    host.resize(330, 400)
    host.show()
    app.processEvents()
    check("scroll-hosted panel: host minimum height stays small",
          host.minimumSizeHint().height() < 300, f"{host.minimumSizeHint().height()}")
    check("scroll-hosted panel: no horizontal overflow",
          inner.width() <= sc.viewport().width()
          and not sc.horizontalScrollBar().isVisible())
    check("panel has no explicit min/max height set",
          panel.maximumHeight() > 10000 and inner.curve_editor.minimumHeight() == 0)
    host.close()
    panel.close()


def test_curve_editor(app: QApplication) -> None:
    ed = CurveEditor()
    ed.resize(280, 400)
    ed.show()
    app.processEvents()
    rec = Recorder(ed.curves_changed)
    plot = ed.plot
    check("plot is square (heightForWidth)", plot.heightForWidth(250) == 250)
    check("plot min size ~220", plot.minimumSizeHint().width() >= 200)
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
          "In 128" in ed._readout.text() and "Out 128" in ed._readout.text(),
          ed._readout.text())
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
    check("rows listed with handle glyph and labels",
          [t.split("  ", 1)[1] for t in labels]
          == ["Exposure", "Contrast", "Curves", "Saturation", "Noise removal", "Sharpen"]
          and all(t.startswith("⋮⋮") for t in labels), str(labels))
    check("ids stored in UserRole",
          tuple(op.row_item(i).data(Qt.UserRole) for i in range(6)) == STAGES)
    check("title text", "Processing order" in op.title_label.text()
          and "drag to reorder" in op.title_label.text())

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


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    test_panel(app)
    test_curve_editor(app)
    test_order_panel(app)
    failed = [n for n, ok, _ in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
