#!/bin/zsh
# build_app.sh — Phase 4 packaging for BlendStack (brief §5).
#
# Builds dist/BlendStack.app (arm64, windowed onedir), ad-hoc signs it,
# verifies the signature, stages the distributable zip
# (dist/BlendStack-1.0.0-arm64.zip = BlendStack.app + DISTRIBUTION_README.md),
# and prints the bundle size.
#
# Usage:  packaging/build_app.sh          (from anywhere; paths are absolute)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$PROJECT_ROOT/.venv/bin/python"
UV="/opt/homebrew/bin/uv"
VERSION="1.0.0"
APP="$PROJECT_ROOT/dist/BlendStack.app"
STAGE="$PROJECT_ROOT/dist/BlendStack-$VERSION-arm64"
ZIP="$PROJECT_ROOT/dist/BlendStack-$VERSION-arm64.zip"

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------- pyinstaller
if ! "$VENV_PY" -c "import PyInstaller" >/dev/null 2>&1; then
    echo "==> Installing PyInstaller into .venv"
    "$UV" pip install --python "$VENV_PY" pyinstaller
fi

# --------------------------------------------------------------------- build
echo "==> Building BlendStack.app (clean)"
"$VENV_PY" -m PyInstaller --clean --noconfirm \
    --distpath "$PROJECT_ROOT/dist" \
    --workpath "$PROJECT_ROOT/build" \
    "$PROJECT_ROOT/packaging/blendstack.spec"

# ---------------------------------------------------------------------- sign
# Ad-hoc sign so the app launches on other Apple Silicon Macs (brief §5).
# PyInstaller 6.x already ad-hoc signs each Mach-O it emits on arm64; the
# deep re-sign below stamps the whole bundle (incl. Info.plist) with one
# consistent ad-hoc seal.  --force replaces the existing signatures —
# without it, codesign refuses to re-sign the PyInstaller-signed pieces.
echo "==> Ad-hoc code signing"
codesign --force --deep -s - "$APP"

echo "==> Verifying signature"
codesign --verify --deep --strict --verbose=2 "$APP"

# ----------------------------------------------------------------- staging
echo "==> Staging distributable zip"
cp "$PROJECT_ROOT/packaging/DISTRIBUTION_README.md" "$PROJECT_ROOT/dist/DISTRIBUTION_README.md"
rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE"
# ditto preserves resource forks, symlinks and code signatures — a plain
# `zip -r` can break Mach-O signatures inside .app bundles.
ditto "$APP" "$STAGE/BlendStack.app"
cp "$PROJECT_ROOT/packaging/DISTRIBUTION_README.md" "$STAGE/README.md"
ditto -c -k --keepParent "$STAGE" "$ZIP"

# ----------------------------------------------------------------- report
echo "==> Bundle size"
du -sh "$APP"
echo "==> Zip artifact"
du -sh "$ZIP"
echo "==> Done: $APP"
