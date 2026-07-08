--[[
   BlendStack — darktable export integration

   darktable has no multi-image operation in its darkroom pixel pipeline, so
   BlendStack plugs in the way darktable's HDR / enfuse / focus-stacking
   tools do: as an export *storage*. You select several images in
   lighttable, develop them however you like, then export with "BlendStack"
   as the target. darktable renders each selected image (applying all your
   darktable edits) to a temporary file; this script then hands the rendered
   files to the shared BlendStack engine (scripts/blend_cli.py) to combine
   them with a Canon comparative or ICM blend mode, and imports the single
   blended result back into the current film roll.

   The blend runs on the *developed* images, so this is the cleanest of the
   three BlendStack frontends: darktable does per-image RAW development, then
   BlendStack does the multi-image blend.

   INSTALL (macOS):
     1. Copy this file to  ~/.config/darktable/lua/blendstack.lua
     2. Add this line to    ~/.config/darktable/luarc :
            require "blendstack"
        (create luarc if it does not exist).
     3. Restart darktable. In lighttable, open the Export panel, and pick
        "BlendStack" in the "storage" (target) dropdown.
     4. First run: set the "BlendStack: python executable" and
        "BlendStack: repository folder" paths under
        Preferences → Lua options, if the defaults do not match your setup.

   USAGE:
     - Select 2-20 images in lighttable.
     - Export panel → target storage = BlendStack. Choose the blend mode and
       (for the Canon modes) softness / bias / basis. TIFF (16-bit) output
       format is recommended. Click export.
     - The blended TIFF is written next to the first source image (or the
       configured output folder) and imported.

   Fold order = source images sorted by filename; the first is the base.
   Order only matters for the order-dependent modes (Overlay, Grain Merge);
   the others are order-independent.

   API: darktable 5.x Lua (tested target: darktable 5.6). Requires a
   darktable built with Lua support (this build ships liblua).
]]

local dt = require "darktable"

local MODULE = "blendstack"

-- Modes offered, in the same order as the engine's registry. Each entry is
-- {registry_name, ui_label}.
local MODES = {
  { "canon_bright", "Canon Bright" },
  { "canon_dark",   "Canon Dark" },
  { "average",      "Average" },
  { "screen",       "Screen" },
  { "multiply",     "Multiply" },
  { "grain_merge",  "Grain Merge" },
  { "overlay",      "Overlay" },
}

-- ------------------------------------------------------------------ prefs

dt.preferences.register(
  MODULE, "python", "file",
  "BlendStack: python executable",
  "Python interpreter that can import the BlendStack engine (its venv)",
  "/Users/chris/Documents/BlendStack/.venv/bin/python"
)
dt.preferences.register(
  MODULE, "repo", "directory",
  "BlendStack: repository folder",
  "Folder containing scripts/blend_cli.py",
  "/Users/chris/Documents/BlendStack"
)
dt.preferences.register(
  MODULE, "outdir", "directory",
  "BlendStack: output folder (optional)",
  "Where to write the blended result; empty = next to the first source image",
  ""
)

-- ------------------------------------------------------------------ helpers

-- Shell-quote a path for a POSIX shell (wrap in single quotes, escape any).
local function shquote(s)
  return "'" .. tostring(s):gsub("'", "'\\''") .. "'"
end

local function basename(path)
  return tostring(path):gsub(".*/", "")
end

-- Read the mode/param widget values into a plain table.
local function current_settings()
  local mode = MODES[mode_widget.selected] or MODES[1]
  local basis = (basis_widget.selected == 2) and "luminance" or "per_channel"
  return {
    mode = mode[1],
    softness = softness_widget.value,
    bias = bias_widget.value,
    basis = basis,
  }
end

-- ------------------------------------------------------------------ storage

-- store: called once per exported image. Accumulate the rendered files.
local exported = {}

local function store(storage, image, format, filename,
                     number, total, high_quality, extra_data)
  table.insert(exported, { name = image.filename, file = filename })
  dt.print_log(MODULE .. ": rendered " .. tostring(number) .. "/" .. tostring(total))
end

-- finalize: called once, after all images are exported. Blend and import.
local function finalize(storage, image_table, extra_data)
  -- Prefer the accumulated list (has the source filename for ordering);
  -- fall back to image_table if store() was bypassed.
  local items = exported
  if #items == 0 then
    for image, file in pairs(image_table) do
      table.insert(items, { name = image.filename, file = file })
    end
  end

  if #items < 2 then
    dt.print("BlendStack: need at least 2 images (got " .. #items .. ")")
    exported = {}
    return
  end
  if #items > 20 then
    dt.print("BlendStack: at most 20 images (got " .. #items .. ")")
    exported = {}
    return
  end

  -- Deterministic fold order: sort by source filename (first = base).
  table.sort(items, function(a, b) return a.name < b.name end)

  local settings = current_settings()

  -- Output location: configured folder, else the folder of the first
  -- selected source image.
  local outdir = dt.preferences.read(MODULE, "outdir", "directory")
  if outdir == nil or outdir == "" then
    local first = nil
    for image in pairs(image_table) do
      if first == nil or image.filename == items[1].name then first = image end
    end
    outdir = first and first.path or "/tmp"
  end
  local out = outdir .. "/blend_" .. settings.mode .. "_" .. os.time() .. ".tif"

  local python = dt.preferences.read(MODULE, "python", "file")
  local repo = dt.preferences.read(MODULE, "repo", "directory")
  local cli = repo .. "/scripts/blend_cli.py"

  local cmd = table.concat({
    shquote(python), shquote(cli),
    "--mode", settings.mode,
    "--softness", tostring(settings.softness),
    "--bias", tostring(settings.bias),
    "--basis", settings.basis,
    "--out", shquote(out),
  }, " ")
  for _, it in ipairs(items) do
    cmd = cmd .. " " .. shquote(it.file)
  end

  dt.print("BlendStack: blending " .. #items .. " images (" .. settings.mode .. ")…")
  dt.print_log(MODULE .. ": " .. cmd)

  local rc = dt.control.execute(cmd)

  -- Clean up darktable's temporary rendered files.
  for _, it in ipairs(items) do os.remove(it.file) end
  exported = {}

  if rc ~= 0 then
    dt.print("BlendStack: blend failed (exit " .. tostring(rc) ..
             ") — see ~/.config/darktable log for the command output")
    return
  end

  local img = dt.database.import(out)
  if img then
    dt.print("BlendStack: created " .. basename(out))
  else
    dt.print("BlendStack: wrote " .. out .. " but import failed")
  end
end

-- supported: which export formats this storage accepts. BlendStack loads
-- TIFF/PNG/JPEG; accept the common raster formats (TIFF 16-bit recommended).
local function supported(storage, format)
  local ext = format.extension
  return ext == "tif" or ext == "tiff" or ext == "png"
      or ext == "jpg" or ext == "jpeg"
end

-- initialize: validate the selection count up front and reset state.
local function initialize(storage, format, images, high_quality, extra_data)
  exported = {}
  if #images < 2 or #images > 20 then
    dt.print("BlendStack: select 2-20 images to blend (selected " ..
             #images .. ")")
    return {}  -- abort the export
  end
  return images
end

-- ------------------------------------------------------------------ widgets

mode_widget = dt.new_widget("combobox"){
  label = "blend mode",
  tooltip = "Canon comparative modes are pixel-exact to the R5; the ICM "
         .. "modes (Average/Screen/Multiply/Grain Merge/Overlay) combine "
         .. "developed frames",
  value = 1,
  MODES[1][2], MODES[2][2], MODES[3][2], MODES[4][2],
  MODES[5][2], MODES[6][2], MODES[7][2],
}

softness_widget = dt.new_widget("slider"){
  label = "softness",
  tooltip = "0 = hard, pixel-exact Canon; higher feathers the edge "
         .. "(Canon modes only)",
  soft_min = 0, soft_max = 100, hard_min = 0, hard_max = 100,
  step = 1, digits = 0, value = 0,
}

bias_widget = dt.new_widget("slider"){
  label = "bias",
  tooltip = "-100..+100; positive favours later images near ties "
         .. "(Canon modes only)",
  soft_min = -100, soft_max = 100, hard_min = -100, hard_max = 100,
  step = 1, digits = 0, value = 0,
}

basis_widget = dt.new_widget("combobox"){
  label = "comparison basis",
  tooltip = "Per channel = Canon-faithful (colours may mix); Luminance "
         .. "keeps the winning pixel's colour (Canon modes only)",
  value = 1,
  "Per channel (Canon)", "Luminance",
}

local widget = dt.new_widget("box"){
  orientation = "vertical",
  mode_widget, softness_widget, bias_widget, basis_widget,
}

-- ------------------------------------------------------------------ register

dt.register_storage(
  "module_blendstack",   -- plugin name
  "BlendStack",          -- display name in the export target dropdown
  store, finalize, supported, initialize, widget
)

-- script_manager compatibility: return a small controllable table so the
-- Lua "script manager" can enable/disable this cleanly. Harmless when the
-- script is loaded via a plain `require "blendstack"` in luarc.
local script_data = {}
script_data.metadata = {
  name = "BlendStack",
  purpose = "blend selected images with BlendStack Canon/ICM modes",
  author = "BlendStack project",
}
script_data.destroy = function()
  pcall(function() dt.destroy_storage("module_blendstack") end)
end
script_data.restart = nil
script_data.show = nil
script_data.destroy_method = nil

return script_data
