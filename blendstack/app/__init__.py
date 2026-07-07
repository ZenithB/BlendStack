"""BlendStack standalone macOS app (PySide6) — project brief §5.

Run with ``python -m blendstack.app``.

The app is a thin frontend over the frozen :mod:`blendstack.core` engine:

* :mod:`blendstack.app.state` — document model (ordered images, per-image
  adjustments, global blend settings) with Qt change signals.
* :mod:`blendstack.app.preview` — live-preview render controller and
  background worker thread (debounce + stale-render cancellation).
* :mod:`blendstack.app.image_strip` — reorderable image strip.
* :mod:`blendstack.app.adjustments_panel` / :mod:`blendstack.app.blend_controls`
  — right-hand control panels.
* :mod:`blendstack.app.histogram` — custom-painted composite histogram.
* :mod:`blendstack.app.presets` — ``.bsp`` (JSON) preset save/load.
* :mod:`blendstack.app.main_window` — assembles everything.
"""
