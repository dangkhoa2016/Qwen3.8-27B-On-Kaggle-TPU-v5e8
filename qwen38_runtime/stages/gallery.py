"""Gallery stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    FULL19_FINAL = context.require("FULL19_FINAL")
    RUN_ROOT = context.require("RUN_ROOT")
    SUPPORT_DIR = context.require("SUPPORT_DIR")
    VI_FINAL = context.require("VI_FINAL")
    # The original audited stage implementation
    from gallery.render import refresh_gallery
    # Build and archive only authenticated outputs from this run.
    import os
    if os.environ.get("QWEN38_RENDER_HTML", "1").strip().lower() not in ("0", "false", "no"):
        import shutil
        review_file = RUN_ROOT / "semantic-review.v1.json"
        gallery_html = refresh_gallery(
            FULL19_FINAL, SUPPORT_DIR / "images", RUN_ROOT / "gallery",
            VI_FINAL, SUPPORT_DIR / "cases/assets_manifest.json",
            review=review_file if review_file.is_file() else None,
        )
        target = RUN_ROOT / "inference-gallery.zip"
        temp = shutil.make_archive(
            str(RUN_ROOT / "inference-gallery.building"), "zip",
            root_dir=RUN_ROOT, base_dir="gallery",
        )
        os.replace(temp, target)
        print("INFERENCE_GALLERY_READY", gallery_html, target, flush=True)
    else:
        print("INFERENCE_GALLERY_DISABLED")
    # Explicitly return the stage's named results to the workflow.
    return {}
