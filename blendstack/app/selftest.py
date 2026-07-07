"""Offscreen smoke test for the standalone app (brief §8 Phase-2 checks).

Run with::

    QT_QPA_PLATFORM=offscreen python -m blendstack.app.selftest

Exercises, against a real (offscreen) MainWindow:

1. add 3 generated images of different sizes/formats → strip + preview;
2. preview render completes; pixmap non-null; histogram has data;
3. softness + one image's exposure changes trigger a re-render
   (generation counter advances);
4. drag-style reorder changes strip and document order;
5. preset save → clear → load round-trip restores images and settings;
6. the 21st image is refused with a clear message;
7. full-resolution export to 16-bit TIFF (dims + dtype verified).

Exits non-zero if any check fails.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from blendstack.core import engine  # noqa: E402
from blendstack.core import io as bs_io  # noqa: E402
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
