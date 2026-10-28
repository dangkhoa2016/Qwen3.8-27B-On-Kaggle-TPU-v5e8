# Git history and prior backup audit

> **English** | [Tiếng Việt](GIT-HISTORY-AUDIT.vi.md)

## Published branch and local preparation

As verified on 28 October 2026, remote main points to a single imported root commit bcac36a, dated 19 October. Its README-focused commit title does not describe all **82 files and 3,229 added lines**, including notebook, API, tests and assets. Author and committer timestamps differ by approximately 3h10m. Do not falsely describe this root as a sequence of feature commits.

The restored LOCAL preparation branch prepublication-review-20261027 adds **four focused commits**, which have original matching author and committer timestamps and meaningful commit bodies:

| SHA | Description | UTC |
|---|---|---|
| 6ac67f8 | Add five images, complete sixteen-asset manifest and photography cases | 27 October, 16:47:16 |
| 3f93c38 | Document real experimental Full19 results and limitations | 27 October, 16:47:16 |
| 3804889 | Prepare single 320/768 Split-Vision production source and checks | 27 October, 17:04:25 |
| 571145a | Document portable offline support and remaining TPU gates | 27 October, 17:04:25 |

Matching timestamps are real from the saved commits. Do not artificially space them merely to create a more attractive development graph. README English and Vietnamese remain DRAFT ONLY and uncommitted pending final acceptance.

## Authenticated remote backup audit

Eight backup refs remain on the private GitHub remote. Among them:

| Backup ref suffix | Commits |
|---|---:|
| pre-feature-history-rewrite-20261019 | 28 |
| pre-message-readme-rewrite-20261019 | 28 |
| pre-evidence-timestamp-rewrite-20261019 | 14 |
| pre-governance-rewrite-20261019 | 11 |

These are older backup/rewrite candidates, **not current main history**. The 28-commit candidate differs from main in **27 files**, including an older executed notebook, evidence files, image versions and README. The tree-to-tree diff from this candidate to main is approximately **387 added and 4,243 deleted lines**. Blindly resetting main to the backup would reintroduce superseded material.

## Recommendation before publication

Preserve the honest imported snapshot and feature-scoped new commits. Any future history migration needs a separately approved plan, proof of exact final Git tree and SHA256, verified backup refs and explicit authorization before a non-fast-forward update. Before changing the repository from Private to Public, review or archive the historical backup/review refs that could expose outdated evidence.

No remote push, merge, delete or history rewrite was executed by this audit. The official fresh TPU Run All remains pending.
