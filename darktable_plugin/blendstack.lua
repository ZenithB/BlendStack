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
   BlendStack does the multi-image blend on top.

   Threading / concurrency (why the code is shaped this way):
     * store() and finalize() run on darktable's EXPORT thread, not the GUI
       thread, so they must NEVER read the live GTK widgets. Widget values
       are captured in initialize() (GUI thread) into `extra_data`, which is
       per-export, and read back in finalize().
     * The list of rendered files is likewise kept in `extra_data`, not a
       module global, so several blends running at once cannot corrupt each
       other's file lists.
     * Output filenames are made unique (timestamp + per-run counter +
       existence check) so two blends started in the same second cannot
       write or import the same path.
     * finalize() is wrapped in pcall: any Lua error becomes a toast, never
       a crash. The engine writes the blended file to disk BEFORE the import
       step, so a blend is never lost even if import misbehaves.

   INSTALL (macOS):
     1. Copy this file to  ~/.config/darktable/lua/blendstack.lua
     2. Add   require "blendstack"   to  ~/.config/darktable/luarc
     3. Restart darktable; pick "BlendStack" as the export storage/target.
     4. First run: set the python / repository paths under
        Preferences -> Lua options if the defaults do not match your setup.

   API: darktable 5.x Lua (tested target: darktable 5.6).
]]

local dt = require "darktable"

local MODULE = "blendstack"

-- Modes offered, in the engine's registry order: {registry_name, ui_label}.
local MODES = {
  { "canon_bright", "Canon Bright" },
  { "canon_dark",   "Canon Dark" },
  { "average",      "Average" },
  { "screen",       "Screen" },
  { "multiply",     "Multiply" },
  { "grain_merge",  "Grain Merge" },
  { "overlay",      "Overlay" },
}

-- Forward declarations so the storage callbacks capture the widgets as
-- upvalues (the widgets are built lower down). Only initialize() reads them,
-- and only on the GUI thread.
local mode_widget, softness_widget, bias_widget, basis_widget

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
dt.preferences.register(
  MODULE, "autoimport", "bool",
  "BlendStack: import result into darktable",
  "Import the blended file back into the film roll. Turn off if importing "
    .. "ever destabilises darktable; the file is still written to the folder.",
  true
)

-- ------------------------------------------------------------------ helpers

-- Shell-quote a path for a POSIX shell (wrap in single quotes, escape any).
local function shquote(s)
  return "'" .. tostring(s):gsub("'", "'\\''") .. "'"
end

local function basename(path)
  return tostring(path):gsub(".*/", "")
end

local function file_exists(path)
  local f = io.open(path, "r")
  if f then f:close(); return true end
  return false
end

-- A unique, non-existing output path. Timestamp + a per-run counter make it
-- distinct even for blends started in the same second; the existence loop is
-- a final guard. `run_counter` increments on darktable's single Lua thread,
-- so it is race-free.
local run_counter = 0
local function unique_output_path(dir, mode)
  run_counter = run_counter + 1
  local stamp = os.date("!%Y%m%d-%H%M%S")
  local stem = dir .. "/blend_" .. mode .. "_" .. stamp .. "_" .. run_counter
  local candidate = stem .. ".tif"
  local n = 1
  while file_exists(candidate) do
    n = n + 1
    candidate = stem .. "_" .. n .. ".tif"
  end
  return candidate
end

-- ------------------------------------------------------------------ storage

-- initialize: runs on the GUI thread before the export starts. Validate the
-- selection and capture the current control values into extra_data so the
-- export-thread callbacks never touch the live widgets.
local function initialize(storage, format, images, high_quality, extra_data)
  if #images < 2 or #images > 20 then
    dt.print("BlendStack: select 2-20 images to blend (selected " .. #images .. ")")
    return {}  -- abort the export
  end
  local mode = MODES[mode_widget.selected] or MODES[1]
  extra_data.mode = mode[1]
  extra_data.softness = softness_widget.value
  extra_data.bias = bias_widget.value
  extra_data.basis = (basis_widget.selected == 2) and "luminance" or "per_channel"
  extra_data.files = {}
  return images
end

-- store: one call per exported image (export thread). Record the rendered
-- file plus the source's folder/name — image objects are safe to read here;
-- widgets are not.
local function store(storage, image, format, filename,
                     number, total, high_quality, extra_data)
  extra_data.files = extra_data.files or {}
  table.insert(extra_data.files, {
    name = image.filename,
    path = image.path,
    file = filename,
  })
  dt.print_log(MODULE .. ": rendered " .. tostring(number) .. "/" .. tostring(total))
end

-- The actual finalize work, called inside a pcall so nothing here can crash
-- darktable.
local function do_finalize(storage, image_table, extra_data)
  local items = extra_data.files or {}
  if #items < 2 then
    dt.print("BlendStack: nothing to blend (need 2-20 images)")
    return
  end

  -- Deterministic fold order: sort by source filename (first = base).
  table.sort(items, function(a, b) return a.name < b.name end)

  local mode = extra_data.mode or "canon_bright"
  local softness = extra_data.softness or 0
  local bias = extra_data.bias or 0
  local basis = extra_data.basis or "per_channel"

  local outdir = dt.preferences.read(MODULE, "outdir", "directory")
  if outdir == nil or outdir == "" then
    outdir = items[1].path or "/tmp"
  end
  local out = unique_output_path(outdir, mode)

  local python = dt.preferences.read(MODULE, "python", "file")
  local repo = dt.preferences.read(MODULE, "repo", "directory")
  local cli = repo .. "/scripts/blend_cli.py"

  local cmd = table.concat({
    shquote(python), shquote(cli),
    "--mode", mode,
    "--softness", tostring(softness),
    "--bias", tostring(bias),
    "--basis", basis,
    "--out", shquote(out),
  }, " ")
  for _, it in ipairs(items) do
    cmd = cmd .. " " .. shquote(it.file)
  end

  dt.print("BlendStack: blending " .. #items .. " images (" .. mode .. ")...")
  dt.print_log(MODULE .. ": " .. cmd)

  local rc = dt.control.execute(cmd)

  -- Remove darktable's temporary rendered files.
  for _, it in ipairs(items) do os.remove(it.file) end

  if rc ~= 0 then
    dt.print("BlendStack: blend failed (exit " .. tostring(rc) ..
             "); see the darktable log for the command output")
    return
  end

  if dt.preferences.read(MODULE, "autoimport", "bool") then
    local img = dt.database.import(out)
    if img then
      dt.print("BlendStack: created " .. basename(out))
    else
      dt.print("BlendStack: wrote " .. out .. " (import returned nothing)")
    end
  else
    dt.print("BlendStack: wrote " .. basename(out) .. " (auto-import off)")
  end
end

-- finalize: called once after all images are exported. Guarded so a Lua
-- error can never crash darktable; the file is already on disk by then.
local function finalize(storage, image_table, extra_data)
  local ok, err = pcall(do_finalize, storage, image_table, extra_data)
  if not ok then
    dt.print("BlendStack: error - " .. tostring(err))
    dt.print_log(MODULE .. ": finalize error: " .. tostring(err))
  end
end

-- supported: BlendStack loads TIFF/PNG/JPEG (TIFF 16-bit recommended).
local function supported(storage, format)
  local ext = format.extension
  return ext == "tif" or ext == "tiff" or ext == "png"
      or ext == "jpg" or ext == "jpeg"
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

-- script_manager compatibility (harmless under a plain require in luarc).
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
