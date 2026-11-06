"""Temporary Cloudflare Gallery tunnel from the verified offline support bundle.

Serves only the finished Gallery. Cell 17 imports this module from the
SHA256-verified source bundle in Runtime Support Dataset v1.
"""
from __future__ import annotations

import functools
import hashlib
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_VERSION = "2026.10.5"
_ASSETS = {
    "x86_64": ("cloudflared-linux-amd64", "e1bc0cf084fd9e590554fe769a86b1db46701c3f847a43d902f7b974a4e16298"),
    "aarch64": ("cloudflared-linux-arm64", "72e0d05a67e09da87cf495e409a28c3cc5994dd23fc66a982eebea6a55923b7d"),
}
_LINK = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com\b")


def _hold_seconds():
    try:
        n = int(os.environ.get("QWEN38_TUNNEL_HOLD_SECONDS", "900"))
    except ValueError as exc:
        raise ValueError("QWEN38_TUNNEL_HOLD_SECONDS must be an integer") from exc
    if not 60 <= n <= 3600:
        raise ValueError("QWEN38_TUNNEL_HOLD_SECONDS must be 60..3600")
    return n


def _verified_cloudflared():
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(platform.machine().lower(), platform.machine().lower())
    if platform.system() != "Linux" or arch not in _ASSETS:
        raise RuntimeError("Only Linux x86_64 and aarch64 are supported")
    asset, digest = _ASSETS[arch]
    path = Path(tempfile.gettempdir()) / ("qwen38-cloudflared-" + _VERSION + "-" + asset)
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Existing cloudflared failed SHA256 verification")
    else:
        url = "https://github.com/cloudflare/cloudflared/releases/download/" + _VERSION + "/" + asset
        fd, temporary = tempfile.mkstemp(prefix=".qwen38-cloudflared-", dir=str(path.parent))
        h, total = hashlib.sha256(), 0
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Qwen38-Gallery-Preview/1.0"})
            with os.fdopen(fd, "wb") as sink, urllib.request.urlopen(request, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 120_000_000:
                        raise RuntimeError("Unexpected cloudflared download size")
                    sink.write(chunk)
                    h.update(chunk)
            if h.hexdigest() != digest:
                raise RuntimeError("cloudflared SHA256 mismatch: refusing execution")
            os.chmod(temporary, 0o755)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    path.chmod(0o755)
    print("GALLERY_CLOUDFLARED_SHA256_VERIFIED", _VERSION, flush=True)
    return path


class _GalleryOnlyHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def _prepare_public_gallery(gallery):
    """Presentation-only label refresh for the unchanged inference Gallery renderer.

    The raw JSON, image assets, evidence and source SHA256 files are unchanged.
    Refresh the matching ZIP so the hosted HTML and download agree.
    """
    page = gallery / "index.html"
    original = page.read_text(encoding="utf-8")
    before, after = "BF16 · Internal review", "BF16 · Inference showcase"
    if before not in original:
        return False
    if original.count(before) != 1:
        raise RuntimeError("Unexpected duplicate Gallery heading")
    archive = gallery.parent / "inference-gallery.zip"
    page.write_text(original.replace(before, after, 1), encoding="utf-8")
    try:
        if archive.is_file():
            temporary = Path(shutil.make_archive(
                str(gallery.parent / "inference-gallery.preview.building"), "zip",
                root_dir=gallery.parent, base_dir=gallery.name))
            os.replace(temporary, archive)
    except Exception:
        page.write_text(original, encoding="utf-8")
        raise
    print("GALLERY_PUBLIC_PRESENTATION_READY", flush=True)
    return True


def launch_gallery_tunnel(gallery_dir=None):
    if os.environ.get("QWEN38_TUNNEL_ENABLED", "1").strip().lower() in ("0", "false", "no"):
        print("GALLERY_TUNNEL_DISABLED", flush=True)
        return None
    gallery = Path(gallery_dir) if gallery_dir is not None else Path(workflow.require("RUN_ROOT")) / "gallery"
    if not (gallery / "index.html").is_file():
        print("GALLERY_TUNNEL_UNAVAILABLE: run step 16 first", flush=True)
        return None
    try:
        hold = _hold_seconds()
        _prepare_public_gallery(gallery)
        executable = _verified_cloudflared()
    except Exception as error:
        print("GALLERY_TUNNEL_UNAVAILABLE:", type(error).__name__, str(error), flush=True)
        print("The Gallery ZIP is still available under /kaggle/working.", flush=True)
        return None
    server = process = None
    try:
        handler = functools.partial(_GalleryOnlyHandler, directory=str(gallery))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        # Verify that a local Gallery HTTP response is available first.
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError("Local Gallery HTTP server is not ready")
        process = subprocess.Popen(
            [str(executable), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        found = threading.Event()
        result, logs = {"url": None}, deque(maxlen=30)

        def read_logs():
            for line in process.stdout:
                logs.append(line.rstrip())
                match = _LINK.search(line)
                if match:
                    result["url"] = match.group(0)
                    found.set()
        threading.Thread(target=read_logs, daemon=True).start()
        timeout = time.monotonic() + 90
        while not found.is_set() and process.poll() is None and time.monotonic() < timeout:
            found.wait(0.5)
        if not result["url"]:
            print("GALLERY_TUNNEL_UNAVAILABLE: no URL returned", "\\n".join(logs)[-1600:], flush=True)
            return None
        url = result["url"]
        print("GALLERY_PUBLIC_URL", url, flush=True)
        print("OPEN NOW while cell 17 is running. This URL will expire when the tunnel ends.", flush=True)
        try:
            from IPython.display import Markdown, display
            display(Markdown("**Gallery URL / Liên kết Gallery:** [" + url + "](" + url + ")"))
        except ImportError:
            pass
        print("GALLERY_TUNNEL_HOLD_SECONDS", hold, flush=True)
        print("This temporary public link expires when this cell/session ends.", flush=True)
        deadline = time.monotonic() + hold
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(min(2, deadline - time.monotonic()))
        print("GALLERY_TUNNEL_CLOSED", flush=True)
        return url
    except KeyboardInterrupt:
        print("GALLERY_TUNNEL_INTERRUPTED", flush=True)
        return None
    except Exception as error:
        print("GALLERY_TUNNEL_UNAVAILABLE:", type(error).__name__, str(error), flush=True)
        return None
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    launch_gallery_tunnel()
