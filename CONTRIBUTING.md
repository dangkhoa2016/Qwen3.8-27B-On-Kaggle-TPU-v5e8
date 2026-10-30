# Contributing to the TPU Text + Vision showcase

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CONTRIBUTING.vi.md)

Contributions are welcome, especially improvements that make **TPU inference easier to understand, reproduce and evaluate**. You do not need a TPU to contribute: API tests, documentation, accessibility and image-evaluation improvements can all be reviewed on CPU.

## Good places to start

- Clarify JAX/MaxText shapes, MRoPE positioning, sharding or compilation behavior with runnable examples.
- Add regression tests for request validation, image SHA256, error states and browser accessibility.
- Improve Full19 case descriptions, ground-truth annotations and numerical reasoning checks.
- Strengthen bilingual English/Vietnamese explanations without modifying raw evidence collected during execution.
- Help document upstream license obligations and exact package versions.

## Before opening a pull request

1. Start with a focused issue or explanation of the improvement.
2. Branch from the current default branch and make a small, reviewable change.
3. Describe **why** the change helps another engineer, not just which files changed.
4. Run applicable tests:

```bash
python -m pip install -r requirements-cpu.txt
python -m pytest -q tests
npm ci
npx playwright install chromium
npm run test:e2e
```

5. If a change concerns real TPU inference, state whether it was **actually executed on TPU**. Include device topology, runtime versions, input hashes, notebook revision and original evidence. A successful CPU mock does not validate TPU generation.
6. Keep English and Vietnamese versions together where a paired page exists.
7. Do not add tokens, passwords, personal data, unlicensed weights, inaccessible private files or fabricated model outputs.

## Evidence and integrity expectations

Preserve original prompt and image SHA256, generated token IDs/text, EOS, duration and all failures. If you improve a validator, identify whether the change affects **test logic** or **model generation**; never retroactively rewrite a model answer to create a PASS.

## Communication

Use GitHub issues for general feedback and reproducible bugs. For vulnerabilities or exposed credentials, follow the [security policy](SECURITY.md) instead of opening a public issue.

Thank you for making TPU model experiments more understandable and useful to others.
