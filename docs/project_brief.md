# BlendStack — Project Brief v1.0

Working title: **BlendStack** (see naming options in cover message; find-and-replace before development starts).

A standalone image blending tool that faithfully recreates the Canon EOS R5's in-camera multiple exposure blend modes — Comparative Bright and Comparative Dark — with per-image adjustments and creative controls the camera does not offer. Delivered as two frontends over one shared engine: a drag-and-drop macOS desktop app and a GIMP 3.2 plugin.

---

## 1. Background and research findings

Canon does not publish the blend algorithm, but the behaviour is well pinned down from the R5 manual, Canon's own technical articles, and the observable artifacts.

**The core operation is a per-pixel, per-channel comparative select.** The R5 manual describes it as: brightness (or darkness) of the base image and added images is compared at the same position, and bright (or dark) portions are retained. This is the standard Lighten/Darken compositing operator: `out = max(A, B)` for Bright, `out = min(A, B)` for Dark, folded left across N images.

**The comparison is per-channel, not per-luminance.** The manual warns that "some overlapping colors may be mixed, depending on the relative brightness of the images," and Canon UK notes "odd colours bleeding through where you don't expect them." Colour mixing is the signature artifact of per-channel max/min: red may win from image A while green and blue win from image B, producing a colour present in neither source. A whole-pixel luminance comparison (Photoshop's "Lighter Color") never mixes colours. Canon's warning tells us it is per-channel.

**The camera blends in linear RAW space — and for these two modes, it doesn't matter.** The R5 saves the composite as a CR3 RAW, so the comparison happens on linear sensor data. But `max` and `min` are invariant under any monotonic per-channel tone curve: whichever value is larger in linear space is still larger after gamma encoding. A per-channel max on gamma-encoded 8/16-bit files selects the identical winning pixels the camera selects in linear RAW. **Comparative Bright/Dark are therefore the two Canon modes that can be replicated faithfully outside the camera without linearisation.** (Additive and Average are not invariant — they require linear-light processing to match. The engine architecture must leave room for this; see §3.)

**Properties worth preserving in the design:**

- Pure max/min is commutative and associative: image order does not affect the result. Once opacity, softness, or bias are non-default, the fold becomes order-dependent — this is an accepted trade-off, and the GUI must therefore support reordering the image list.
- The camera caps at 9 exposures; this tool caps at 20.
- The camera shows a running composite histogram during build-up; the standalone app replicates this.

---

## 2. Scope

### In scope (v1)

- Two blend modes, labelled in the UI as **"Canon Bright"** and **"Canon Dark"**.
- 2–20 input images per blend.
- Input formats: TIFF, JPEG, PNG, GIF (first frame only), BMP, WebP, and RAW (Canon .CR2/.CR3 plus common others via LibRaw) using LibRaw default demosaic, white balance, and tone mapping — no RAW controls in v1.
- Output formats: TIFF (16-bit), PNG (16-bit), JPEG (8-bit). Internal processing is float32 throughout.
- Size mismatch handling: all images scaled to the dimensions of the smallest image using aspect-preserving cover scaling with centre-crop (no distortion; edge content may be lost). Lanczos resampling.
- Per-image pre-blend adjustments (standalone app only): exposure trim, contrast, saturation, sharpness (radius + amount), opacity.
- Blend controls (both frontends): softness, bias, comparison basis (per-channel / luminance).
- Standalone macOS app: drag-and-drop, reorderable image list, live preview, composite histogram, preset save/load, packaged as a double-clickable `.app` distributable to others.
- GIMP 3.2 plugin: blends all visible layers of the open image, inserts result as a new top layer.

### Out of scope (v1, door left open)

- Additive and Average modes (require linear-light pipeline — see §3 architecture note).
- Custom/novel blend algorithms (the mode registry in §3 exists for these).
- RAW development controls (WB, exposure, demosaic choice).
- Manual per-image transform/positioning (compositor territory).
- Animated GIF handling beyond first frame.
- Windows/Linux builds.

---

## 3. Architecture

Three components, one language (Python 3.11+):

```
blendstack/
├── core/                  # Pure engine. No UI imports. NumPy only.
│   ├── engine.py          # Pipeline orchestration
│   ├── modes/             # Blend mode registry
│   │   ├── registry.py    # register_mode() decorator + lookup
│   │   ├── canon_bright.py
│   │   └── canon_dark.py
│   ├── adjustments.py     # exposure, contrast, saturation, sharpen
│   ├── io.py              # load (Pillow + rawpy), save, bit-depth handling
│   └── geometry.py        # scale-to-smallest, cover-crop, Lanczos
├── app/                   # PySide6 standalone GUI
└── gimp_plugin/           # GIMP 3.2 GObject Introspection plugin
```

**Design rules:**

1. `core` has zero UI dependencies and zero GIMP dependencies. Both frontends import it. All blend logic lives here once.
2. Blend modes are pluggable. Each mode is a class registered by name, declaring: its parameters (name, range, default), its fold function `blend(accumulator, incoming, params) -> ndarray`, and whether it requires linear-light input (`needs_linear: bool`, False for both v1 modes). This is the door for Additive, Average, and custom modes: when a linear mode is added, the engine linearises (sRGB EOTF) before the fold and re-encodes after, with no frontend changes.
3. The engine operates on float32 arrays normalised to 0.0–1.0, RGB, shape (H, W, 3). Alpha channels in inputs are flattened against black on load.

### Why Python (rationale for the record)

The GIMP 3.x plugin API is Python via GObject Introspection; the standalone GUI needs a mature cross-widget toolkit with drag-and-drop (PySide6/Qt); NumPy vectorises the blend maths so 20-image folds at 24MP run in seconds. Any other language would force the engine to be written twice.

---

## 4. Processing pipeline (engine specification)

For each render (preview or export):

```
for each image i in user-defined order:
    load -> float32 RGB 0..1           (RAW via LibRaw defaults; GIF frame 0; alpha flattened)
    geometry: cover-scale + centre-crop to target dims (smallest image's dims)
    adjustments, in this fixed order:
        1. exposure trim
        2. contrast
        3. saturation
        4. sharpen (unsharp mask)
accumulator = image[0] (after its adjustments; opacity ignored for the first image)
for each subsequent image i:
    blended = mode.blend(accumulator, image[i], params)
    accumulator = lerp(accumulator, blended, opacity[i])
clip accumulator to 0..1
encode to output format/bit depth
```

### 4.1 Adjustment definitions

| Adjustment | Range (UI) | Definition |
|---|---|---|
| Exposure trim | −3.0 … +3.0 EV | Linearise sRGB → multiply by 2^EV → re-encode sRGB. Done in linear light so it behaves like a real exposure change, not a gamma-space gain. |
| Contrast | −100 … +100 | Gamma-space pivot at 0.5: `out = (in − 0.5) × k + 0.5`, where k maps −100→0.5, 0→1.0, +100→2.0. |
| Saturation | −100 … +100 | `out = lerp(luma, in, s)` per pixel, luma = Rec.709 weights (0.2126, 0.7152, 0.0722); s maps −100→0.0 (greyscale), 0→1.0, +100→2.0. |
| Sharpness | radius 0.5–10 px, amount 0–200% | Unsharp mask: `out = in + amount × (in − gaussian_blur(in, radius))`. |
| Opacity | 0–100% | Applied at the fold step as defined above, not as an adjustment to the image itself. |

### 4.2 Blend mode maths

Both modes share one formulation; they differ only in comparison direction. Per pixel:

**Comparison value.** With basis = *per-channel* (default, Canon-faithful): the comparison value is each channel independently — three independent selections per pixel; colours may mix (the authentic Canon artifact). With basis = *luminance*: the comparison value is Rec.709 luma computed once per pixel; one selection weight applied to all three channels — the winning pixel keeps its colour intact, no fringing.

**Selection weight (Canon Bright).**

```
d = (B_cmp − A_cmp) + bias          # A = accumulator, B = incoming
if softness == 0:  w = (d > 0) ? 1 : 0        # hard max — exact Canon behaviour
else:              w = sigmoid(d / t)          # t = softness mapped to (0, 0.25]
out = A × (1 − w) + B × w
```

**Canon Dark** is identical with `d = (A_cmp − B_cmp) + bias` (select the smaller).

Because `out` is a convex combination of A and B, the soft version never overshoots either source — it feathers the winner-takes-all boundary rather than glowing past it. This is deliberate: it directly addresses harsh transition edges without introducing new blowout.

**Parameter semantics:**

| Parameter | Range (UI) | Mapping | At default |
|---|---|---|---|
| Softness | 0–100 | 0 → hard select; 1–100 → sigmoid temperature t linear in (0.0025, 0.25] normalised units | 0 = pixel-exact Canon |
| Bias | −100 … +100 | offset −0.25 … +0.25 in normalised 0–1 units, added to the incoming image's side of the comparison. Positive = incoming image wins ties and near-ties more often. | 0 = pixel-exact Canon |
| Comparison basis | per-channel / luminance | as above | per-channel = Canon |

**Faithfulness claim, stated precisely:** with softness 0, bias 0, basis per-channel, all opacities 100%, and no per-image adjustments, the output pixel selection is identical to the R5's Comparative Bright/Dark for the same source pixels, and the result is order-independent. Any non-default setting departs from camera behaviour and makes order significant.

### 4.3 Geometry

Target dimensions = the smallest input by area. Each other image: scale by `max(target_w / w, target_h / h)` (cover), Lanczos, then centre-crop to target. Identical-size inputs pass through untouched.

### 4.4 Output encoding

TIFF and PNG write 16-bit; JPEG writes 8-bit, quality 95, 4:4:4 if the encoder allows. Float accumulator is clipped then quantised with rounding. Default filename: `blend_<mode>_<YYYYMMDD-HHMMSS>.<ext>`. sRGB profile tagged on output.

---

## 5. Standalone app (PySide6)

### Layout

Left: reorderable image strip (drag to reorder, drag files in to add, per-item remove; order = fold order, top = first/base). Centre: live preview canvas. Right: two stacked panels — per-image adjustments for the selected image (exposure, contrast, saturation, sharpen radius/amount, opacity, reset button) and global blend controls (mode dropdown, softness, bias, basis toggle). Bottom right or overlaid: composite RGB + luma histogram of the current preview. Toolbar: Open, Save Preset, Load Preset, Export.

### Live preview

Mandatory. Strategy: on load, each image gets a proxy downscaled to ≤1440 px long edge (proxies cached after per-image adjustments where possible — geometry and adjustments are cached per image and invalidated only when that image's settings change, so slider drags re-run only the fold). Renders run in a background QThread with debounce (~80 ms) and cancellation of stale renders; the UI thread never blocks. Full-resolution processing happens only on Export, with a progress dialog.

### Histogram

Computed from the preview accumulator (post-clip): 256-bin R, G, B and luma overlaid. Updates with every preview render. Purpose: judge highlight accumulation during build-up, matching the R5's composite histogram workflow.

### Presets

"Preset" (user-facing term — never "recipe"): JSON file containing schema version, blend mode + parameters, output settings, and the ordered image list with absolute paths and all per-image adjustments. Loading a preset with missing files warns per-file and loads what it can. Extension: `.bsp` (BlendStack preset) or plain `.json` — developer's choice, document it.

### Packaging and distribution

PyInstaller `.app` bundle, arm64 (Apple Silicon). Ad-hoc codesign (`codesign --force --deep -s -`) so it launches on other Macs; document the Gatekeeper right-click → Open first-launch step in a bundled README since the app will not be notarised. Include rawpy/LibRaw and Qt frameworks in the bundle; verify bundle size stays reasonable (<300 MB).

---

## 6. GIMP 3.2 plugin

- Registered via GObject Introspection (`gi.require_version('Gimp', '3.0')` — the API is stable for the whole 3.x series, so 3.0 bindings run on 3.2).
- Menu location: `Filters > Combine > BlendStack…`.
- Behaviour: takes all **visible layers** of the active image, top layer = first/base image in the fold, descending order. Layers are read at the image's full size (layers offset or smaller than canvas are composited against black at canvas size). Result inserted as a **new layer at the top of the stack**, named `BlendStack <mode>`; originals untouched.
- Dialog exposes blend controls only: mode, softness, bias, comparison basis. No per-image adjustments (GIMP itself is the adjustment tool) and no preview in v1 (GIMP 3's non-destructive filter preview machinery does not apply to multi-layer plugins; re-run is cheap).
- Precision: request pixel data as float via GEGL buffers (Babl format "R'G'B' float"), process with the shared core, write back float. Works correctly on 8-, 16-, and 32-bit images.

### Known risk: NumPy inside GIMP's Python

GIMP's bundled Python interpreter on macOS may not ship NumPy. Mitigations, in order of preference: (1) detect at plugin load; if missing, show a dialog with the one-time install command targeting GIMP's own interpreter (`/Applications/GIMP.app/Contents/MacOS/python3 -m pip install numpy`) — a clear one-liner with an explanation, consistent with the no-terminal preference being bent only once at install time; (2) vendor a NumPy wheel inside the plugin folder; (3) worst case, a slow pure-Python fallback for the fold only. Resolve during Phase 3; do not let this block Phases 1–2.

---

## 7. Milestones

**Phase 1 — Core engine.** `core/` complete with both modes, adjustments, geometry, IO, mode registry. Deliverable: a test harness script that blends a folder of images and writes output, plus the unit tests below. No GUI yet.

**Phase 2 — Standalone app.** Full PySide6 app against the finished core: drag-and-drop, reorder, adjustments, live preview with threading, histogram, presets, export.

**Phase 3 — GIMP plugin.** Wrapper over the same core; resolve the NumPy dependency question.

**Phase 4 — Packaging.** PyInstaller `.app`, ad-hoc signing, bundled README, hand-off test on a second Mac.

---

## 8. Acceptance criteria and verification checklist

### Engine correctness (Phase 1, automated tests)

- [ ] Canon Bright at defaults on two synthetic gradients equals `np.maximum(A, B)` exactly; Canon Dark equals `np.minimum(A, B)`.
- [ ] Per-channel mode produces mixed colours on a red-vs-green test pair (out contains (r_A, g_B, b) pixels); luminance mode on the same pair never produces a colour absent from both sources.
- [ ] Order-independence: shuffling 5 images at default settings produces bit-identical output; with opacity 50% on one image, shuffled output differs (documents the trade-off).
- [ ] Softness > 0 output is bounded: `min(A,B) ≤ out ≤ max(A,B)` everywhere.
- [ ] Monotone-invariance sanity check: applying a gamma curve to both inputs then blending selects the same winners as blending then applying the curve (hard mode, per-channel).
- [ ] 20-image blend at 24 MP completes and stays within memory budget.
- [ ] Round-trip: 16-bit TIFF in → no adjustments, single image "blend" → 16-bit TIFF out is lossless.
- [ ] RAW file loads via rawpy defaults without error; GIF loads frame 0; alpha PNG flattens against black.
- [ ] Mismatched sizes: outputs match smallest image's dimensions, no distortion (aspect check on a circle test image).

### Standalone app (Phase 2, manual checklist)

- [ ] Drag 3 files of mixed formats onto the window; all appear in the strip in drop order.
- [ ] Dragging strip items reorders; preview updates and (with non-default opacity) visibly changes.
- [ ] Every slider updates the preview within ~200 ms without UI freeze; rapid slider scrubbing does not queue stale renders.
- [ ] Histogram visibly shifts right as bright-winning content accumulates in Canon Bright.
- [ ] Save preset, quit, relaunch, load preset: identical preview restored.
- [ ] Load preset with one source file renamed: warning names the missing file; remaining images load.
- [ ] Export 16-bit TIFF and open in darktable/GIMP: dimensions, bit depth, sRGB tag correct; result visually matches preview.
- [ ] 21st image is refused with a clear message.

### GIMP plugin (Phase 3)

- [ ] On a 3-visible-layer image, plugin adds one new top layer; originals untouched; undo removes it cleanly.
- [ ] Result on identical inputs matches the standalone app at the same settings (export both, diff).
- [ ] Works on 8-bit and 32-bit float precision images.
- [ ] Hidden layers are excluded.

### Packaging (Phase 4)

- [ ] `.app` double-click launches on the development Mac and on one other Apple Silicon Mac (right-click → Open documented for first launch).
- [ ] RAW loading works inside the bundle (LibRaw shipped correctly).

---

## 9. Open items deferred to future versions

Additive and Average modes via the linear-light path already scaffolded in the registry; user-authored blend modes (the registry makes each one a single file); RAW development controls; per-image geometric transforms; Windows build; animated output.
