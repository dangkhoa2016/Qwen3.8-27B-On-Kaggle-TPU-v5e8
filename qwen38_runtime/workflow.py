"""Readable, ordered execution of the production TPU showcase.

The heavy inference implementation lives in stages/*.py; this object holds
the explicitly returned state between cells without relying on IPython globals.
No model loads or TPU imports occur until the corresponding step is executed.
"""
from __future__ import annotations

from importlib import import_module

STEPS = (
    "preflight", "dependencies", "maxtext", "inputs",
    "checkpoint", "tpu", "engine", "inference_helpers",
    "text_schedule", "ocr_probe", "ocr_evidence", "vision_reuse",
    "image_cases", "text_cases", "vietnamese", "gallery",
)


class Workflow:
    def __init__(self) -> None:
        self.state: dict[str, object] = {}
        self.completed: list[str] = []

    def __contains__(self, key: str) -> bool:
        return key in self.state

    def require(self, key: str):
        if key not in self.state:
            raise RuntimeError(f"Notebook stage dependency missing: {key}")
        return self.state[key]

    def run(self, name: str) -> None:
        if name not in STEPS:
            raise ValueError(f"Unknown notebook step: {name}")
        expected = STEPS[len(self.completed)] if len(self.completed) < len(STEPS) else None
        if name != expected:
            # The Vietnamese resume cell and Gallery may be rerun after Full19
            # is complete; all other TPU steps must execute in strict order.
            if not (name in ("vietnamese", "gallery")
                    and len(self.completed) >= STEPS.index(name) + 1):
                raise RuntimeError(f"Run cells in order; next step is {expected!r}")
        module = import_module("qwen38_runtime.stages." + name)
        result = module.execute(self)
        if not isinstance(result, dict):
            raise TypeError(f"Stage {name} must return named state values")
        if name == expected:
            self.completed.append(name)
        self.state.update(result)
        print("NOTEBOOK_STAGE_COMPLETE", name, flush=True)
