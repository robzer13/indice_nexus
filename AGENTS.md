# OroTitan — AI/Code Agent Entry Point (PUBLIC repository)

**Read first:** `docs/orotitan-equity/OROTITAN_REPOSITORY_BOUNDARY_R0.md` (canonical R0).

Repository identity: `robzer13/indice_nexus` — PUBLIC application, contracts, migrations, validation, workflows, and tests.

Mandatory behavior:
- Never put private research JSON, historical evidence, workpapers, ZIP/base64 C03 payloads, financial statements, or credentials in this public repository, including PR branches, fixtures, logs, or workflow artifacts.
- The private immutable research archive is `robzer13/real-orotitan`; do not modify it as a side effect of a code change here.
- Route mixed requests into independent repository-scoped PRs. Declare the exact target repository, run ID and approved paths before any write.
- Preserve Registry contracts and existing canonical migrations unless an independently reviewed change is explicitly required.
- Use SHADOW only when specifically authorized. Never mutate PRODUCTION or deploy as an implicit side effect.
- Never push to `main`; open a scoped PR. Require `repo-boundary` CI PASS. Do not bypass required GitHub checks or claim the gate is active before branch-protection settings are confirmed.
- For governance-only branches, prevent automatic Vercel deployment prior to publication; repository automation is not an authorization to deploy.

If target repository, evidence authority, or permission is unclear, **BLOCK and report** without writes.
