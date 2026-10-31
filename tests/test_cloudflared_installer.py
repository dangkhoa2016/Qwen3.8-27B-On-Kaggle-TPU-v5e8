"""CPU-only offline checks for optional Cloudflare Quick Tunnel installer."""
import os
import platform
import subprocess
from pathlib import Path

import pytest

INSTALLER=Path(__file__).resolve().parents[1]/"gallery/install_cloudflared.sh"


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux-only script")
def test_installer_dry_run_prints_pinned_asset_without_downloading(tmp_path):
    env={**os.environ, "QWEN38_CLOUDFLARED_BIN_DIR":str(tmp_path/"custom-bin")}
    output=subprocess.run(["bash", str(INSTALLER), "--check"],
                          capture_output=True, text=True, env=env, check=True)
    assert "Version: 2026.10.5" in output.stdout
    assert "SHA256: " in output.stdout
    assert str(tmp_path/"custom-bin/cloudflared") in output.stdout
    assert not (tmp_path/"custom-bin").exists()


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux-only script")
def test_installer_refuses_to_overwrite_invalid_existing_binary(tmp_path):
    destination=tmp_path/"custom-bin"
    destination.mkdir()
    target=destination/"cloudflared"
    target.write_text("NOT_CLOUDFLARED")
    env={**os.environ,"QWEN38_CLOUDFLARED_BIN_DIR":str(destination)}
    output=subprocess.run(["bash",str(INSTALLER)],
                          capture_output=True,text=True,env=env)
    assert output.returncode != 0
    assert "Refusing to overwrite" in output.stderr
    assert target.read_text() == "NOT_CLOUDFLARED"


def test_installer_pins_official_hashes_and_restricts_execution():
    source=INSTALLER.read_text()
    assert "e1bc0cf084fd9e590554fe769a86b1db46701c3f847a43d902f7b974a4e16298" in source
    assert "72e0d05a67e09da87cf495e409a28c3cc5994dd23fc66a982eebea6a55923b7d" in source
    assert "curl --fail --location" in source
    assert "sha256sum --check --status" in source
    assert "chmod 755" in source
    assert "cloudflared tunnel" not in source
    subprocess.run(["bash","-n",str(INSTALLER)],check=True)
