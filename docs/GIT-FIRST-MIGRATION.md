# Git-first Kaggle TPU notebook — migration guide

[Tiếng Việt](GIT-FIRST-MIGRATION.vi.md)

## What changed

The **single canonical notebook** at [notebooks/kaggle-tpu-v5e8-production-showcase.ipynb](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) clones implementation code from GitHub. It contains 17 code cells: 16 ordered workflow stages (19 English cases, optional 17 Vietnamese generations and CPU HTML Gallery) and one optional interactive Gallery Tunnel cell. The original Kaggle offline execution remains historical evidence; it is not a second notebook in Git.

Project code is no longer distributed through the **new** Runtime Assets Dataset. The seven pinned support archives still contain the original **48** runtime/data payloads, including offline wheels, input images and a patched **third-party MaxText source archive**. That archive is an executable dependency; it is **not** the project's 23 Python modules.

## Kaggle setup

1. Enable **Internet = On** in Kaggle notebook settings **before** running the first code cell. This is required to reach GitHub.
2. Attach the existing, original Qwen3.8-27B BF16 Kaggle model (Transformers/default/version 1).
3. Attach the [Runtime Assets Dataset v1](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-assets) (ready, private). Kaggle provides seven runtime archives and 48 SHA256-verified payloads; no project Python source bundle is included. The historical Dataset v1 remains unchanged.
4. GitHub is public and allows anonymous clones without GitHub credentials. The BF16 Kaggle model and Runtime Assets Dataset are separate inputs: make them accessible to the intended Kaggle users before claiming third-party reproducibility.
5. The notebook defaults to `main` and records the exact fetched Git HEAD for every run. For a deliberate reproduction of a historical run, optionally set `QWEN38_GIT_REF` to its **full 40-character commit SHA** before bootstrap. `QWEN38_EXPECTED_COMMIT` can separately verify an expected SHA while cloning a branch/tag. A mismatch stops **before TPU initialization**.
6. Execute notebook cells in order. Stage 1 validates the runtime Dataset's 48 original SHA256 records and independently verifies Git HEAD, clean worktree and origin URL. The run writes `source-provenance.json` alongside generated TPU results.

`main` is intentionally **not pinned**: it is a developer convenience and must not be described as a frozen benchmark. Each executed run prints and records the exact Git commit used.

## For maintainers: how the already-published Runtime Assets Dataset v1 was built

The new builder validates every archive and all 48 payloads from the original distribution, retaining the same manifest bytes. It copies attribution sidecars and third-party licenses while **omitting the entire project-code ZIP and source manifest**:

```bash
python3 support/package_runtime_assets.py \
  --source /path/to/verified-legacy-dataset-v1 \
  --destination /path/to/new-empty-runtime-assets
```

Input can be the seven original ZIPs or their Kaggle-extracted category directories, alongside the original manifest and notices. The output has the new Dataset ID:

`dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-assets`

The **private Kaggle Runtime Assets Dataset version 1** is ready. All 48 payload SHA256 values were independently verified after re-downloading the published Dataset. Git source changes no longer require repackaging this Dataset.

## Local acceptance and limitations

```bash
python3 support/git_first_bootstrap.py --check
python3 -m pytest -q tests
npm run test:e2e
```

CPU tests include a synthetic Git clone, pinned-SHA mismatch rejection, runtime-only packaging and the legacy notebook regressions. They do **not** prove the new Git-first notebook has completed a real Kaggle TPU run. A fresh TPU acceptance run is needed before describing this **new execution path** as production-tested.

Do not edit historical TPU responses, substitute a post-migration Git SHA for the earlier embedded-source TPU run, or rewrite the old Dataset v1. Keep its original execution evidence immutable.
