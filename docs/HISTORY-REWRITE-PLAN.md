# History rewrite audit

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](HISTORY-REWRITE-PLAN.vi.md)

This project reconstructed **existing commits without inventing timestamps**. Original history was independently backed up at `backup/pre-governance-rewrite-20261019`. Original 11 commits have actual author/committer timestamps 2026-10-18 through 2026-10-19. Author identity can be corrected to `Đăng Khoa <i.am@dangkhoa.dev>` without pretending new source existed earlier.

Newly drafted governance commits use their actual creation timestamp. Code changes should stay near or under 600 added/deleted lines per commit (large binary/evidence or bilingual documentation exceptions).

Before replacing `main`: compare full final trees (excluding intended governance changes), verify SHA256 evidence, inspect notebook's pinned examples SHA and ensure reachable, run CI and E2E, and retain backup reference. Use `--force-with-lease`, never blind `--force`.
