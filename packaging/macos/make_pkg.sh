#!/usr/bin/env bash
set -e

VERSION="0.1.0"
IDENTIFIER="com.kaohaoyi.llmzip"
TARGET_NAME="$1" # e.g. aarch64-apple-darwin or x86_64-apple-darwin

ROOT_DIR="pkg_root"
rm -rf "$ROOT_DIR"
mkdir -p "$ROOT_DIR/usr/local/bin"
mkdir -p "$ROOT_DIR/usr/local/include"
mkdir -p "$ROOT_DIR/usr/local/lib"

cp dist_staging/llmzip "$ROOT_DIR/usr/local/bin/"
chmod 755 "$ROOT_DIR/usr/local/bin/llmzip"

if [ -f "include/llmzip.h" ]; then
  cp include/llmzip.h "$ROOT_DIR/usr/local/include/"
fi

if [ -f "target/${TARGET_NAME}/release/libllmzip_ffi.dylib" ]; then
  cp "target/${TARGET_NAME}/release/libllmzip_ffi.dylib" "$ROOT_DIR/usr/local/lib/"
  chmod 755 "$ROOT_DIR/usr/local/lib/libllmzip_ffi.dylib"
fi

PKG_OUT="llmzip-${VERSION}-${TARGET_NAME}.pkg"
pkgbuild --root "$ROOT_DIR" --identifier "$IDENTIFIER" --version "$VERSION" --install-location "/" "$PKG_OUT"
rm -rf "$ROOT_DIR"
echo "[✓] Built macOS package: $PKG_OUT"
