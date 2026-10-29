#!/usr/bin/env bash
set -e

VERSION="0.1.0"
ARCH="amd64"
PKG_DIR="llmzip_${VERSION}_${ARCH}"

rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/usr/local/bin"
mkdir -p "$PKG_DIR/usr/local/include"
mkdir -p "$PKG_DIR/usr/local/lib"

# Copy binaries
cp dist_staging/llmzip "$PKG_DIR/usr/local/bin/"
chmod 755 "$PKG_DIR/usr/local/bin/llmzip"

if [ -f "include/llmzip.h" ]; then
  cp include/llmzip.h "$PKG_DIR/usr/local/include/"
fi

if [ -f "target/x86_64-unknown-linux-gnu/release/libllmzip_ffi.so" ]; then
  cp target/x86_64-unknown-linux-gnu/release/libllmzip_ffi.so "$PKG_DIR/usr/local/lib/"
  chmod 755 "$PKG_DIR/usr/local/lib/libllmzip_ffi.so"
fi

cat <<EOF > "$PKG_DIR/DEBIAN/control"
Package: llmzip
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: kaohaoyi <kaohaoyi@users.noreply.github.com>
Description: High-performance GGUF & Safetensors deduplication, semantic compression and zero-copy mmap runtime
 High-density chunk deduplication and instant runtime for local LLMs, resolving the double-space dilemma.
EOF

dpkg-deb --build "$PKG_DIR"
mv "${PKG_DIR}.deb" "llmzip_${VERSION}_amd64.deb"
rm -rf "$PKG_DIR"
echo "[✓] Built Debian package: llmzip_${VERSION}_amd64.deb"
