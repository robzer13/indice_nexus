# OroTitan — Repository Boundary R0 (canonical governance)

**Scope:** all human, ChatGPT, Codex and CI changes. **Default: DENY on ambiguity.**

## Owners (one object, one canonical repository)
- **PRIVATE** `robzer13/real-orotitan`: confidential research data, workpapers, sealed JSON outputs, analytical manifests and immutable content bytes. Historic private application modules may be maintained, but new application features belong in `indice_nexus`.
- **PUBLIC** `robzer13/indice_nexus`: application and interfaces, technical contracts, schema, PostgreSQL migrations/RPC, validators, GitHub workflows and tests. No private JSON, financial source extracts, archives or credentials.
- **Supabase Registry SHADOW**: operational run/stage state, CAS, authority state, event ledger, immutable artifact locators and integrity metadata. It is **not** a substitute for storing private artifact bodies. PRODUCTION is never an implicit target.

## Immutable cross-repository handoff
1. Produce and review private research artifacts inside `real-orotitan`; do not send them to the public repository.
2. Persist new output artifacts in an immutable private Git commit. Never replace an existing artifact identity/version.
3. Build and persist the final manifest only after output commits and blob SHA/size/SHA-256 are known. No simulated commit, fictitious locator, or circular self-reference.
4. Validate and independently reread exact bytes from `real-orotitan` before attesting them. Preserve the existing Registry RPCs and contracts (no parallel registry).
5. Admit in SHADOW only through an explicitly authorized protected operation with appropriate role; use CAS and V3.1 post-COMMIT trusted loader. Simulated C03 receipts are LOCAL ONLY and never confer remote authority.

## Pull requests and merge rules
- One PR, one repository and responsibility. Mixed tasks become separate PRs. Never push directly to main.
- CI `repo-boundary` must run on **every PR** without path exclusions, and be a **required status check** in the protection of main in **both** repositories.
- Protect the boundary script, workflows, R0 policy and `/artifacts/` with CODEOWNERS and enforce review where supported. Owner self-review does not constitute independent approval.
- Private PRs that introduce manifest references to predecessor Git commits **must use a merge commit**, not squash/rebase; ensure referenced commits remain ancestors of the resulting main. GitHub repository merge settings and required reviews must enforce this operationally.
- Never bypass required checks or reassign a missing check to an optional workflow.
- Public `repo-boundary` denies private-artifact paths/filenames, likely private JSON run instances, private bundle archives, and common credentials. It is defense in depth; it does **not** replace GitHub secret scanning or human review.

## Legacy scope and change control
Historic duplicated code and documents remain in place until separately reviewed migration; **do not** perform a risky mass move under governance R0.
No new service, third repository, registry, secret relay, or FREEZE contract change is authorized by this governance proposal.
Any exception to a denied operation requires a separate, independently reviewed policy/validator change and evidence; not a one-off bypass.

## Deployment and privileges
Repository governance PRs are documentation/CI only. No deployment, no Supabase mutation, no direct main write. Use a branch already explicitly excluded from Vercel builds or verify that a new branch is excluded **before** publishing changes.
