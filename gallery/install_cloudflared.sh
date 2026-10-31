#!/usr/bin/env bash
# Install the Cloudflare Quick Tunnel binary on a CPU Linux host.
# No sudo, TPU, Cloudflare login, or background tunnel startup.
set -euo pipefail
VERSION='2026.10.5'
BASE_URL="https://github.com/cloudflare/cloudflared/releases/download/${VERSION}"
OS="$(uname -s)"
ARCH="$(uname -m)"
if [[ "$OS" != 'Linux' ]]; then
  echo "Unsupported OS: $OS. See Cloudflare's official installation guide." >&2
  exit 2
fi
case "$ARCH" in
  x86_64|amd64)
    ASSET='cloudflared-linux-amd64'
    EXPECTED_SHA256='e1bc0cf084fd9e590554fe769a86b1db46701c3f847a43d902f7b974a4e16298'
    ;;
  aarch64|arm64)
    ASSET='cloudflared-linux-arm64'
    EXPECTED_SHA256='72e0d05a67e09da87cf495e409a28c3cc5994dd23fc66a982eebea6a55923b7d'
    ;;
  *)
    echo "Unsupported Linux architecture: $ARCH. Supported: x86_64, aarch64." >&2
    exit 2
    ;;
esac
DEST_DIR="${QWEN38_CLOUDFLARED_BIN_DIR:-${HOME}/.local/bin}"
TARGET="${DEST_DIR}/cloudflared"
URL="${BASE_URL}/${ASSET}"
if [[ "${1:-}" == '--check' ]]; then
  printf 'Version: %s\nOS/architecture: %s/%s\nAsset: %s\nSHA256: %s\nTarget: %s\n' \
    "$VERSION" "$OS" "$ARCH" "$ASSET" "$EXPECTED_SHA256" "$TARGET"
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo "Usage: bash gallery/install_cloudflared.sh [--check]" >&2
  exit 2
fi
command -v sha256sum >/dev/null || { echo 'sha256sum is required' >&2; exit 1; }
command -v curl >/dev/null || { echo 'curl is required' >&2; exit 1; }
if [[ -e "$TARGET" || -L "$TARGET" ]]; then
  if [[ -f "$TARGET" && ! -L "$TARGET" ]] && \
    printf '%s  %s\n' "$EXPECTED_SHA256" "$TARGET" | sha256sum --check --status; then
    chmod 755 "$TARGET"
    printf 'Verified existing installation: %s\n' "$TARGET"
    "$TARGET" --version
    exit 0
  fi
  echo "Refusing to overwrite an existing unverified file: $TARGET" >&2
  exit 1
fi
mkdir -p "$DEST_DIR"
TEMP="$(mktemp "${DEST_DIR}/.cloudflared.download.XXXXXXXX")"
trap 'rm -f "$TEMP"' EXIT
printf 'Downloading %s (Cloudflare release %s)\n' "$ASSET" "$VERSION"
curl --fail --location --silent --show-error --retry 3 --connect-timeout 15 \
  --output "$TEMP" "$URL"
printf '%s  %s\n' "$EXPECTED_SHA256" "$TEMP" | sha256sum --check --status || {
  echo "SHA256 mismatch: refusing to execute downloaded binary." >&2
  exit 1
}
chmod 755 "$TEMP"
mv -n "$TEMP" "$TARGET"
if ! printf '%s  %s\n' "$EXPECTED_SHA256" "$TARGET" | sha256sum --check --status; then
  echo 'Installation could not be verified.' >&2
  exit 1
fi
printf 'Verified installation: %s\n' "$TARGET"
"$TARGET" --version
