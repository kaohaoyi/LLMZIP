#!/usr/bin/env sh
set -e

# LLMZIP Agent Automatic Installer for Linux & macOS
REPO="kaohaoyi/LLMZIP"
INSTALL_DIR="${LLMZIP_INSTALL_DIR:-$HOME/.local/bin}"

mkdir -p "$INSTALL_DIR"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "$OS" in
  linux)
    TARGET="x86_64-unknown-linux-gnu"
    ;;
  darwin)
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
      TARGET="aarch64-apple-darwin"
    else
      TARGET="x86_64-apple-darwin"
    fi
    ;;
  *)
    echo "[!] Unsupported OS: $OS. Please install via 'cargo install llmzip-cli'." >&2
    exit 1
    ;;
esac

TAG="${LLMZIP_VERSION:-v0.2.0}"
ASSET="llmzip-${TAG}-${TARGET}.tar.gz"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"

echo "[*] Downloading LLMZIP for ${TARGET}..."
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$URL" -o /tmp/llmzip.tar.gz || {
    echo "[!] Prebuilt binary not yet on release. Falling back to compiling via cargo git..." >&2
    cargo install --git "https://github.com/${REPO}.git" llmzip-cli --locked
    exit 0
  }
elif command -v wget >/dev/null 2>&1; then
  wget -qO /tmp/llmzip.tar.gz "$URL" || {
    echo "[!] Prebuilt binary not yet on release. Falling back to compiling via cargo git..." >&2
    cargo install --git "https://github.com/${REPO}.git" llmzip-cli --locked
    exit 0
  }
fi

tar -xzf /tmp/llmzip.tar.gz -C "$INSTALL_DIR" llmzip
chmod +x "$INSTALL_DIR/llmzip"
rm -f /tmp/llmzip.tar.gz

echo "[✓] LLMZIP installed successfully to $INSTALL_DIR/llmzip"
if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
  echo "[i] Note: Add '$INSTALL_DIR' to your PATH if not already included."
fi
