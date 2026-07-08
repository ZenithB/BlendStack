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
   name) into GIMP's user plug-ins directory. The version-specific config
   directory depends on your build — GIMP 3.2 installs use either
   `GIMP/3.2/` or `GIMP/3.0/`. The authoritative path is shown in GIMP under
   **Edit ▸ Preferences ▸ Folders ▸ Plug-ins**; on a stock GIMP 3.2 it is:

   ```
   cp -R gimp_plugin/blendstack-blend \
     ~/Library/Application\ Support/GIMP/3.2/plug-ins/
   ```

   (If your GIMP shows `3.0` there instead, substitute that — the plugin is
   identical either way.)

2. Make the plugin executable (GIMP silently ignores non-executable
   plugin files):

   ```
   chmod +x ~/Library/Application\ Support/GIMP/3.2/plug-ins/blendstack-blend/blendstack-blend.py
   ```

3. Nothing else to do — **no NumPy install is required**. The plugin ships
   a stdlib-only fallback and works out of the box.

4. Restart GIMP.

### Why there's no NumPy install step (macOS)

The blend maths normally use NumPy, and most guides tell you to install it
into GIMP's own Python:

```
/Applications/GIMP.app/Contents/MacOS/python3 -m pip install numpy
```

On current **Apple Silicon macOS this does not work**, for two stacked
reasons we verified against the system logs:

1. GIMP's bundled `python3.10` **cannot be launched directly from a shell**
   — macOS Launch Constraints (AMFI) kill it (`zsh: killed`) because it may
   only run as a child of the `gimp` process. So the `pip install` command
   above never even starts.
2. Even with NumPy present, GIMP's Python runs under a **hardened runtime**
   that enforces *library validation*: it refuses to load NumPy's
   ad-hoc-signed C extensions (`Library Validation failed … different Team
   IDs`). The only way to load them is to re-sign GIMP itself to disable
   library validation — a security-relevant change to a third-party app.

So the plugin uses brief §6's **mitigation 3**: a pure-Python fold
(`fold_purepy.py`, stdlib only). It is slower than NumPy — a hard
`Canon Bright`/`Dark` blend of a few-megapixel image across a handful of
layers takes a few seconds; a 20-layer 24-megapixel worst case can take a
couple of minutes — but it needs no dependencies, no terminal step, and no
change to GIMP, and it is **bit-identical to the NumPy engine at default
settings**.

**Optional — full NumPy speed:** if you are on Linux/Windows/Intel, or a
GIMP build without library validation, and NumPy is importable by GIMP's
Python, the plugin detects it and uses it automatically. There is nothing
to configure.

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
- On macOS the fold runs in pure Python (see "Why there's no NumPy install
  step" above), so very large blends take longer than the standalone app.
  For a fast full-resolution export, blend in the **standalone BlendStack
  app** instead — both frontends share the same engine and give identical
  results at the same settings.

## Development

- `sync_core.py` — re-vendors the frozen core
  (`blendstack/__init__.py` + `blendstack/core/`, minus `__pycache__`) into
  `blendstack-blend/blendstack/`. Run it after any core change; never edit
  the vendored copy by hand.
- `fold_purepy.py` — the stdlib-only fallback backend (no NumPy, no core
  import). Used inside GIMP on macOS where NumPy can't load; also the
  source of the mode metadata and layer limits so the plugin registers
  without NumPy. Never edit by hand without re-running the tests.
- `test_plugin_logic.py` — verifies **both** backends against
  `blendstack.core.engine` without GIMP: the NumPy path (`blend_logic.py`)
  and the pure-Python path (`fold_purepy.py`), bit-exact at defaults and
  within a float32 ULP for soft/biased/luminance blends. Run with
  `.venv/bin/python gimp_plugin/test_plugin_logic.py`.
- Backend selection: at load time the plugin tries `import numpy` +
  `blend_logic`; if that fails it uses `fold_purepy`. Neither backend ever
  imports `blendstack.core.engine`, because the engine pulls in
  `io`/`geometry` which need Pillow (absent from GIMP's Python). The NumPy
  backend imports only `blendstack.core.modes` + `adjustments`; the
  pure-Python backend imports nothing outside the stdlib. Both mirror
  `engine.fold_images` and are tested against it.
