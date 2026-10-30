# Pull request

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](PULL_REQUEST_TEMPLATE.vi.md)

Thank you for contributing! Help reviewers understand **what becomes easier or more reliable** after this change.

## What does this improve?

Describe the user-facing benefit and relevant issue.

## What changed?

Summarize the implementation and its impact on documentation, runtime, datasets and tests.

## How was it validated?

- [ ] Python tests run; provide command and result
- [ ] Browser tests run if UI was changed
- [ ] EN/VI documentation updated where applicable
- [ ] Model outputs are genuine, unedited and accompanied by input SHA256
- [ ] TPU behavior is marked **tested on hardware** or **not tested on hardware**
- [ ] No credentials, nonpublic material or unlicensed model weights included

## Evidence

Record the exact Git revision, environment, dataset/model versions, screenshots or safe logs, and any measured metrics. Clearly separate CPU mock tests from actual TPU inference.

## Limitations and follow-up

What still needs testing? Mention known failure cases and missing measurements instead of hiding them.
