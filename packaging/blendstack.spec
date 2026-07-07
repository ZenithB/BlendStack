# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the BlendStack macOS .app bundle (brief §5, Phase 4).

Builds a *windowed, onedir* arm64 bundle named "BlendStack.app".
(onefile .app bundles unpack to a temp dir on every launch — slow and
fragile — so onedir is used deliberately.)

Run from the project root:

    .venv/bin/pyinstaller --clean --noconfirm packaging/blendstack.spec

Notes
-----
* rawpy ships LibRaw as ``libraw_r*.dylib`` inside the wheel (plus
  liblcms2/libjasper/libjpeg under ``rawpy/.dylibs``).  The ``_rawpy``
  extension links it via ``@loader_path/libraw_r.25.dylib``, so the
  dylibs must land *next to the extension* inside the bundle —
  ``collect_dynamic_libs('rawpy')`` preserves that layout.
* scipy is installed in the dev venv but nothing under ``blendstack/``
  imports it (verified with grep — core is pure NumPy), so it is
  excluded, along with matplotlib/tkinter guards and every PySide6
  module the app does not use (the app imports only QtCore, QtGui,
  QtWidgets).  Excluding the unused PySide6 *Python* bindings stops the
  PySide6 hooks from bundling the matching Qt frameworks/plugins/QML.
* imageio is pure Python and small; its heavy optional backends
  (ffmpeg, pyav, tifffile, …) are not installed in the venv, so only
  the pillow-based bits the app actually uses get bundled.  A few are
  named in ``excludes`` anyway as a guard against future env drift.
* No .icns icon is available for v1, so the bundle uses the default
  PyInstaller icon (``icon=None``).  CFBundleDocumentTypes below is
  therefore declared without per-type icons.
"""

import os

from PyInstaller.utils.hooks import collect_dynamic_libs

# SPECPATH is provided by PyInstaller: the directory containing this spec.
PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

APP_NAME = "BlendStack"
VERSION = "1.0.0"
BUNDLE_ID = "com.blendstack.app"

# ---------------------------------------------------------------------------
# Binaries: make sure LibRaw (and rawpy's other vendored dylibs) ship.
# ---------------------------------------------------------------------------
binaries = collect_dynamic_libs("rawpy")

# ---------------------------------------------------------------------------
# Hidden imports: rawpy and the imageio v3 pillow path are imported lazily
# (inside functions) in blendstack.core.io; name them explicitly so a
# future PyInstaller bytecode-scan change can never drop them.
# ---------------------------------------------------------------------------
hiddenimports = [
    "rawpy",
    "rawpy._rawpy",
    "imageio",
    "imageio.v3",
    "imageio.plugins.pillow",
    "imageio.plugins.pillow_legacy",
]

# ---------------------------------------------------------------------------
# Excludes (size control, brief §5: bundle < 300 MB).
# ---------------------------------------------------------------------------
excludes = [
    # -- PySide6 modules the app never imports.  The PyInstaller pyside6
    #    hooks bundle Qt frameworks per *included* binding, so excluding
    #    these keeps QtWebEngine/QtQuick/QML etc. out of the bundle.
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.QtWebView",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickTest",
    "PySide6.QtQuickWidgets",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.QtGraphsWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtSpatialAudio",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtNetwork",          # app is fully offline; nothing imports it
    "PySide6.QtNetworkAuth",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialBus",
    "PySide6.QtSerialPort",
    "PySide6.QtStateMachine",
    "PySide6.QtTextToSpeech",
    "PySide6.QtHelp",
    "PySide6.QtDesigner",
    "PySide6.QtUiTools",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtConcurrent",
    "PySide6.QtDBus",
    "PySide6.QtPrintSupport",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtXml",
    "PySide6.QtHttpServer",
    # -- Not dependencies of BlendStack; guards in case the venv grows.
    "scipy",                       # verified: no blendstack import (pure NumPy core)
    "matplotlib",
    "tkinter",
    "_tkinter",
    # -- imageio optional heavy backends (not installed; guard anyway).
    "imageio.plugins.ffmpeg",
    "imageio.plugins.pyav",
    "imageio_ffmpeg",
    "av",
    # -- Dev/test-only packages present in the venv.
    "pytest",
    "_pytest",
    "pygments",
    "setuptools",
    "pip",
]

a = Analysis(
    [os.path.join(SPECPATH, "launch_blendstack.py")],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,       # onedir — binaries live in COLLECT
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,               # windowed GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",         # Apple Silicon only (brief §5)
    codesign_identity=None,      # ad-hoc signing happens in build_app.sh
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    icon=None,                   # no .icns asset available in v1 (noted in brief hand-off)
    bundle_identifier=BUNDLE_ID,
    version=VERSION,
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
        "LSMinimumSystemVersion": "12.0",
        "LSApplicationCategoryType": "public.app-category.photography",
        # Optional document-type declarations (no per-type icons — no
        # .icns available).  BlendStack is a multi-image tool, so these
        # are advertised as "Alternate" viewers only; the primary
        # workflow is drag-and-drop onto the open window.
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "Image",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Alternate",
                "LSItemContentTypes": [
                    "public.tiff",
                    "public.jpeg",
                    "public.png",
                    "com.compuserve.gif",
                    "com.microsoft.bmp",
                    "org.webmproject.webp",
                    "public.camera-raw-image",
                ],
            }
        ],
    },
)
