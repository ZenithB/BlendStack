# BlendStack 1.0.0

BlendStack is a standalone image blending tool that faithfully recreates
the Canon EOS R5's in-camera multiple exposure blend modes — **Comparative
Bright** and **Comparative Dark** ("Canon Bright" / "Canon Dark" in the
app) — with per-image adjustments and creative controls the camera does
not offer. Drop 2–20 images onto the window, reorder them, tune exposure,
contrast, saturation, sharpness and opacity per image, adjust softness and
bias on the blend, watch the live preview and composite histogram, and
export a 16-bit TIFF/PNG or JPEG. RAW files (Canon .CR2/.CR3 and other
common formats) are supported out of the box — LibRaw is bundled inside
the app via rawpy, so no extra installs are needed.

## System requirements

- Apple Silicon Mac (M1 or later). This build is **arm64 only** — it will
  not run on Intel Macs.
- macOS 12 (Monterey) or later.
- No other software required; Python, Qt, NumPy and LibRaw are all inside
  the app bundle.

## First launch (important!)

BlendStack is signed with an *ad-hoc* signature and is **not notarised by
Apple**, so the very first launch is blocked by Gatekeeper with a message
like "BlendStack cannot be opened because the developer cannot be
verified" or "Apple could not verify…". This is expected. Open it one of
these two ways — you only have to do this once:

**Option A — right-click Open**

1. In Finder, **right-click** (or Control-click) `BlendStack.app`.
2. Choose **Open** from the menu.
3. In the warning dialog, click **Open** (or **Open Anyway**).

**Option B — System Settings**

1. Double-click `BlendStack.app`; dismiss the warning.
2. Open **System Settings ▸ Privacy & Security**, scroll down to the
   Security section.
3. Next to the message about BlendStack, click **Open Anyway**, then
   confirm.

On recent macOS versions (Sequoia and later) Option B may be required.
After the first successful launch, BlendStack opens normally with a
double-click.

If macOS reports the app as "damaged" after downloading (a quarantine
quirk with unsigned apps in some browsers), clear the quarantine flag in
Terminal: `xattr -dr com.apple.quarantine /path/to/BlendStack.app`

## Basic usage

1. **Add images** — drag image files onto the window (or use *Open*).
   TIFF, JPEG, PNG, GIF (first frame), BMP, WebP and RAW are accepted;
   2 to 20 images per blend.
2. **Order matters** (once you use non-default settings) — the left-hand
   strip is the fold order, top = base image. Drag items to reorder.
3. **Pick a mode** — *Canon Bright* keeps the brighter pixel (star trails,
   light painting); *Canon Dark* keeps the darker pixel.
4. **Adjust per image** — select an image in the strip and use the
   right-hand panel: exposure trim, contrast, saturation, sharpen,
   opacity.
5. **Tune the blend** — softness feathers the winner-takes-all edge, bias
   shifts which image wins near-ties, and the comparison basis toggles
   between per-channel (authentic Canon, colours may mix) and luminance
   (no colour fringing). All defaults = pixel-exact Canon behaviour.
6. **Watch the histogram** — the composite RGB + luma histogram updates
   live, matching the R5's build-up workflow.
7. **Presets** — save/load the whole setup (images, order, all settings)
   from the toolbar.
8. **Export** — full-resolution render to 16-bit TIFF, 16-bit PNG, or
   JPEG. The live preview uses downscaled proxies; export always
   re-renders at full resolution.

## RAW support

RAW decoding (Canon .CR2/.CR3, plus .NEF, .ARW, .DNG and other common
formats) is handled by **LibRaw via rawpy, bundled inside the app** —
nothing to install. v1 uses LibRaw's default demosaic, white balance and
tone mapping; there are no RAW development controls yet.

---

BlendStack 1.0.0 · Apple Silicon (arm64) · not notarised — see
"First launch" above.
