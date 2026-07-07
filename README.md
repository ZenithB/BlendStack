# BlendStack

A standalone image blending tool that faithfully recreates the Canon EOS R5's in-camera multiple exposure blend modes — **Comparative Bright** and **Comparative Dark** — with per-image adjustments and creative controls the camera does not offer.

Two frontends over one shared engine:

- **macOS desktop app** (PySide6) — drag-and-drop, reorderable image list, live preview, composite histogram, presets, 16-bit export.
- **GIMP 3.2 plugin** — blends all visible layers of the open image and inserts the result as a new top layer.

## Why "Canon-faithful"?

The R5's Comparative Bright/Dark are per-pixel, **per-channel** max/min selections. Because `max`/`min` are invariant under any monotonic tone curve, a per-channel max on gamma-encoded files selects the identical winning pixels the camera selects in linear RAW — these are the two Canon modes that can be replicated bit-faithfully outside the camera. At default settings (softness 0, bias 0, per-channel basis, 100% opacity, no adjustments) the output is pixel-identical to the camera's selection and order-independent.

On top of the faithful core, BlendStack adds what the camera doesn't offer:

- **Softness** — feathers the winner-takes-all boundary with a sigmoid (never overshoots either source).
- **Bias** — shifts the comparison so the incoming image wins ties more or less often.
- **Comparison basis** — per-channel (Canon-authentic, colours may mix) or luminance (winning pixel keeps its colour intact).
- **Per-image adjustments** (desktop app) — exposure trim (true linear-light EV), contrast, saturation, unsharp-mask sharpening, opacity.
- Up to **20 images** per blend (the camera caps at 9).

## Repository layout

```
blendstack/
├── core/          # Pure engine — NumPy pipeline, blend mode registry, IO, geometry
├── app/           # PySide6 standalone macOS app
gimp_plugin/       # GIMP 3.2 plugin (GObject Introspection)
scripts/           # CLI test harness (blend a folder of images)
tests/             # Engine acceptance tests
docs/              # Project brief / specification
```

## Quick start (engine + CLI)

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install numpy pillow rawpy PySide6

# Blend a folder of images
python scripts/blend_folder.py /path/to/images --mode canon_bright --out blend.tif
```

## Desktop app

```bash
python -m blendstack.app
```

Drag images in (TIFF, JPEG, PNG, GIF, BMP, WebP, RAW), reorder the strip (top = base image), tweak per-image adjustments and blend controls, watch the live preview and composite histogram, and export 16-bit TIFF/PNG or 8-bit JPEG.

## GIMP plugin

Copy `gimp_plugin/blendstack` into your GIMP 3.2 plug-ins directory, then find it under **Filters → Combine → BlendStack…**. It blends all visible layers (top layer = base) and inserts the result as a new top layer.

## Tests

```bash
python -m pytest tests/
```

The suite verifies the faithfulness claims: exact `np.maximum`/`np.minimum` equivalence at defaults, per-channel colour mixing vs. luminance-basis purity, order-independence, softness boundedness, monotone-curve invariance, 16-bit round-trip losslessness, and geometry correctness.

## License

MIT
