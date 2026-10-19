"""TPU integration boundary, adapted from evidence notebook cells 15 and 17.

TPU_FACTORY requires a bootstrapped JAX/MaxText environment and a pinned
configuration. The factory is deliberately not auto-invoked on CPU.
This code has NOT undergone a live TPU acceptance run.
"""
from __future__ import annotations

import time
from pathlib import Path


class MaxEngineRunner:
    """One engine, one BF16 checkpoint; serialized deterministic inference."""

    def __init__(self, engine, params, config, tokenizer, *, prefill_length=256, max_target_length=384):
        self.engine, self.params, self.config, self.tokenizer = engine, params, config, tokenizer
        self.prefill_length = prefill_length
        self.max_target_length = max_target_length

    def _decode(self, prefix, first, limit):
        import jax
        state = self.engine.init_decode_state()
        state = self.engine.insert(prefix, state, slot=0)
        tokens = [int(jax.device_get(first.data)[0, 0])]
        for _ in range(limit - 1):
            state, output = self.engine.generate(params=self.params, decode_state=state, algorithm="greedy")
            jax.block_until_ready(output.data)
            token = int(jax.device_get(output.data)[0, 0])
            tokens.append(token)
            if token == self.tokenizer.eos_token_id:
                break
        return {"generated_token_ids": tokens, "generated_text": self.tokenizer.decode(tokens, skip_special_tokens=False)}

    def generate_text(self, prompt: str, *, max_decode_steps: int = 24):
        import jax
        import jax.numpy as jnp
        import numpy as np
        tok = self.tokenizer
        rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        if len(ids) > self.prefill_length:
            raise ValueError("Text prompt exceeds fixed prefill budget")
        pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        padded = np.asarray(ids + [pad] * (self.prefill_length - len(ids)), dtype=np.int32)
        start = time.monotonic()
        prefix, first = self.engine.prefill(params=self.params, padded_tokens=jnp.asarray(padded),
                                            true_length=len(ids), algorithm="greedy")
        jax.block_until_ready(first.data)
        prefill = time.monotonic() - start
        result = self._decode(prefix, first, min(max_decode_steps, self.max_target_length - len(ids)))
        return {**result, "prefill_seconds": prefill, "prompt_token_count": len(ids)}

    def generate_vision(self, image_path: str, prompt: str, *, max_decode_steps: int = 160):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from maxtext.configs import pyconfig
        from maxtext.multimodal import processor as mm_processor
        from maxtext.multimodal import processor_qwen3_omni as qwen_mm
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        pd = self.config._pydantic_config.model_copy(update={"image_path": str(path), "prompt": prompt})
        cfg = pyconfig.HyperParameters(pd)
        po = mm_processor.preprocess_mm_data(cfg)
        offset = int(mm_processor.get_image_offsets(config=cfg, processor_output=po))
        tok = self.tokenizer
        content = f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"
        rendered = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        base = tok(rendered, add_special_tokens=False)["input_ids"]
        budget = self.prefill_length - offset
        if budget <= 0 or len(base) > budget:
            raise ValueError("Multimodal prompt exceeds fixed prefill budget")
        pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        padded = np.asarray(base + [pad] * (budget - len(base)), dtype=np.int32)
        fused = mm_processor.prepare_text_for_image_fusion(tokens=padded, config=cfg, processor_output=po)
        true_length = len(base) + offset
        mask = (np.arange(len(fused))[None, :] < true_length).astype(np.int32)
        pos, delta = qwen_mm.get_rope_index(
            input_ids=fused[None, :], image_grid_thw=po.pixel_grid_thw, attention_mask=mask,
            spatial_merge_size=cfg.spatial_merge_size_for_vit,
            position_id_per_seconds=cfg.position_id_per_seconds, config=cfg)
        start = time.monotonic()
        prefix, first = self.engine.prefill(
            params=self.params, padded_tokens=jnp.asarray(fused, dtype=jnp.int32),
            positions=jnp.asarray(pos), mrope_deltas=jnp.asarray(delta),
            images=jnp.asarray(po.pixel_values), true_length=true_length, algorithm="greedy")
        jax.block_until_ready(first.data)
        result = self._decode(prefix, first, min(max_decode_steps, self.max_target_length - true_length))
        return {**result, "prefill_seconds": time.monotonic() - start, "image_offset_tokens": offset}

    def close(self):
        # JAX backend resources are owned by the process; never unload/reload on each request.
        self.params = None
        self.engine = None
