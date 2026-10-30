# Security and responsible disclosure

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](SECURITY.vi.md)

This repository includes an **educational CPU mock API** and an **experimental TPU adapter**, not a production-hardened public Qwen service. Use the mock playground on **127.0.0.1** and review authentication, request limits, error handling and real TPU readiness before deploying any endpoint.

## Report a vulnerability

Send security reports privately to **i.am@dangkhoa.dev**. Include an affected component, a minimal reproduction, potential impact and a safe way to contact you. Please **do not** publish exploit steps, credentials or private logs in a GitHub issue before the maintainer has had an opportunity to investigate.

## Keep credentials and assets safe

- Never commit Kaggle API tokens, GitHub PATs, private keys or cloud credentials.
- Sanitize error logs, screenshots and notebook outputs before sharing.
- Do not expose the optional TPU HTTP adapter through a public tunnel until actual authentication and live load tests have passed.
- Preserve least privilege and use bounded requests/concurrency for a model that consumes large memory.
- Rotate affected tokens immediately if credentials are accidentally exposed.

## Understanding the limits

Unit tests for authorization and bounded requests are useful, but **they are not a substitute for a full security review of a deployed TPU service**. See [API integration](api/TPU-INTEGRATION.md) for the component boundaries.
