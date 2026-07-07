# BlendStack GIMP plugin

Blends all **visible layers** of the open image with the Canon-faithful
Comparative Bright / Comparative Dark engine (project brief §6) and inserts
the result as a new layer at the top of the stack. The plugin **bundles its
own copy of the BlendStack core engine** (vendored by `sync_core.py`), so it
has no dependency on this repository once installed — the only external
requirement is NumPy (see below).

## Install (macOS, GIMP 3.2)

1. Copy the whole `blendstack-blend` folder (the folder, not just the `.py`
   file — GIMP 3 requires the plugin file to live in a folder of the same
   name) into GIMP's user plug-ins directory:

   ```
   cp -R gimp_plugin/blendstack-blend \
     ~/Library/Application\ Support/GIMP/3.0/plug-ins/
   ```

   GIMP 3.2 still uses the `GIMP/3.0` configuration directory — the config
   folder is versioned once for the whole GIMP 3 series. If your install
   differs, the authoritative path is listed in GIMP under
   **Edit ▸ Preferences ▸ Folders ▸ Plug-ins**.

2. Make the plugin executable (GIMP silently ignores non-executable
   plugin files):

   ```
   chmod +x ~/Library/Application\ Support/GIMP/3.0/plug-ins/blendstack-blend/blendstack-blend.py
   ```

3. Install NumPy into **GIMP's own** Python interpreter (one time only).
   GIMP's bundled Python does not ship NumPy, which the blend engine needs
   for its image maths — this is the only terminal step:

   ```
   /Applications/GIMP.app/Contents/MacOS/python3 -m pip install numpy
   ```

   If you skip this, the plugin still appears in the menu and shows this
   exact command when run.

4. Restart GIMP.

## Usage

**Filters ▸ Combine ▸ BlendStack…**

The dialog exposes the blend controls only (per-image adjustments are
GIMP's job, and there is no preview in v1 — re-running is cheap):

| Control | Range | Default | Meaning |
|---|---|---|---|
| Mode | Canon Bright / Canon Dark | Canon Bright | comparative select: brighter / darker wins |
| Softness | 0–100 | 0 | 0 = hard, pixel-exact Canon select; higher feathers the winner-takes-all boundary |
| Bias | −100…+100 | 0 | positive lets lower layers win near-ties more often |
| Comparison basis | Per channel / Luminance | Per channel | per-channel = Canon-faithful (colours may mix); luminance keeps the winning pixel's colour intact |

At the defaults the result is pixel-identical to the Canon EOS R5's
in-camera Comparative Bright/Dark selection.

It is also scriptable non-interactively via the PDB as
`plug-in-blendstack` with arguments `mode` (choice), `softness` (double),
`bias` (double), `basis` (choice: `per_channel` / `luminance`).

## Behaviour notes

- **All visible layers** are blended; hidden layers are skipped. Needs at
  least 2 and at most 20 visible layers (engine limit).
- **Top layer = base**: the fold runs down the layer stack, top-first
  (at default settings the result is order-independent anyway).
- Layers that are **offset or smaller than the canvas** are composited
  against **black** at canvas size before blending; layer alpha is also
  flattened against black (matching the standalone app's file-loading
  policy). Top-level layer groups are blended as their composited content.
- The result is inserted as a **new top layer** named
  `BlendStack <mode>`; the original layers are untouched. The insertion is
  one undo step — **Edit ▸ Undo** removes it cleanly.
- Works identically on 8-, 16- and 32-bit images: pixels are read and
  written as float through GEGL/Babl (`R'G'B' float`), and the v1 modes are
  invariant under bit-depth/gamma re-encoding. RGB images only (convert via
  **Image ▸ Mode ▸ RGB** first).

## Development

- `sync_core.py` — re-vendors the frozen core
  (`blendstack/__init__.py` + `blendstack/core/`, minus `__pycache__`) into
  `blendstack-blend/blendstack/`. Run it after any core change; never edit
  the vendored copy by hand.
- `test_plugin_logic.py` — verifies the plugin's numeric path
  (`blend_logic.py`) bit-exact against `blendstack.core.engine` without
  GIMP: `.venv/bin/python gimp_plugin/test_plugin_logic.py`.
- The plugin deliberately imports `blendstack.core.modes` +
  `blendstack.core.adjustments` (NumPy-only) and **not**
  `blendstack.core.engine`, because the engine imports `io`/`geometry`
  which need Pillow — absent from GIMP's Python. The fold loop in
  `blend_logic.py` mirrors `engine.fold_images` exactly and is tested
  bit-exact against it.
