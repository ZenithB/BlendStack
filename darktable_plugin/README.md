# BlendStack darktable integration

Blends several developed images from darktable with the BlendStack Canon /
ICM blend modes, and imports the result back into your film roll.

## What kind of plugin this is (and isn't)

darktable's darkroom processes **one image** through a fixed pixel pipeline,
and there is no third-party pixel-module ("iop") plugin ABI — so BlendStack
**cannot** be a darkroom module. Instead it plugs in the same way darktable's
HDR / enfuse / focus-stacking tools do: as a **Lua export storage**. You
select images, darktable develops and exports each one, and BlendStack
combines the developed results into a single new image.

This is actually the nicest of the three BlendStack frontends: darktable
does all the per-image RAW development (white balance, exposure, tone,
denoise, everything), and BlendStack does the multi-image blend on top — so
this frontend only needs the blend controls.

Requires a darktable built with Lua support (the official macOS build ships
it) and a working BlendStack engine (the repo's Python venv). Tested against
darktable 5.6.

## Install (macOS)

1. Copy the script into darktable's Lua folder:

   ```
   mkdir -p ~/.config/darktable/lua
   cp darktable_plugin/blendstack.lua ~/.config/darktable/lua/
   ```

2. Tell darktable to load it — add this line to `~/.config/darktable/luarc`
   (create the file if it doesn't exist):

   ```
   require "blendstack"
   ```

3. Restart darktable.

4. First run only: open **darktable → Preferences → Lua options** and check
   the two paths, correcting them if your setup differs from the defaults:
   - **BlendStack: python executable** — the Python that can run the engine
     (the repo venv, e.g. `…/BlendStack/.venv/bin/python`).
   - **BlendStack: repository folder** — the folder containing
     `scripts/blend_cli.py` (e.g. `…/BlendStack`).
   - **BlendStack: output folder** (optional) — where to write the blended
     file; leave empty to write next to the first source image.

## Usage

1. In **lighttable**, select **2–20 images** to blend.
2. Open the **export** panel (right-hand side).
3. Set the **storage / target** dropdown to **BlendStack**.
4. Choose the **blend mode**, and for the Canon modes the **softness**,
   **bias** and **comparison basis**. (Those three controls are ignored by
   the Average/Screen/Multiply/Grain Merge/Overlay modes.)
5. Set the file **format** to **TIFF** (16-bit recommended) and pick any
   quality/size options — these control how darktable develops each frame
   before blending.
6. Click **export**. darktable renders each selected image, BlendStack
   blends them, and the single result is imported into the current film
   roll (a toast reports the new filename).

## Notes

- **Fold order** is the source images sorted by filename; the first is the
  base. Order only changes the result for the order-dependent modes
  (**Overlay**, **Grain Merge**); the others are order-independent.
- The blend runs on the **developed** exports, so any darktable edits you've
  made are baked in first — this is the intended workflow.
- **Average** simulates a single long exposure from many frames (great for
  ICM); it is computed in linear light for correct brightness.
- The engine is shared with the standalone app and the GIMP plugin, so a
  given mode + settings produces the same result across all three.
- Errors (bad paths, blend failure) are reported as a darktable toast; the
  full command and any Python error go to darktable's log
  (run darktable from a terminal, or check `~/.config/darktable/`).

## How it works

`blendstack.lua` registers an export storage. Its `initialize` validates the
2–20 selection, `store` collects each rendered file, and `finalize` sorts
them, calls `scripts/blend_cli.py` (the shared engine) with the chosen mode
and parameters, imports the resulting TIFF, and cleans up darktable's
temporary renders.
