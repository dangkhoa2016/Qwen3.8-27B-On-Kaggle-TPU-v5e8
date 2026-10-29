"""Tpu stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    MAXTEXT_DIR = context.require("MAXTEXT_DIR")
    os = context.require("os")
    # The original audited stage implementation
    # 5. Initialize JAX TPU and patched MaxText.
    import sys, time
    os.environ['MAXTEXT_REPO_ROOT'] = str(MAXTEXT_DIR)
    sys.path.insert(0, str(MAXTEXT_DIR / 'src'))

    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer
    from maxtext.configs import pyconfig
    from maxtext.inference.maxengine.maxengine import MaxEngine
    from maxtext.utils.globals import MAXTEXT_REPO_ROOT

    print('backend=', jax.default_backend())
    print('device_count=', jax.device_count())
    print('local_device_count=', jax.local_device_count())
    assert jax.default_backend() == 'tpu'
    assert jax.device_count() == 8
    print('TPU_GATE=PASS')
    # Explicitly return the stage's named results to the workflow.
    return {'MAXTEXT_REPO_ROOT': MAXTEXT_REPO_ROOT, 'jax': jax, 'jnp': jnp, 'pyconfig': pyconfig, 'MaxEngine': MaxEngine, 'AutoTokenizer': AutoTokenizer, 'time': time}
