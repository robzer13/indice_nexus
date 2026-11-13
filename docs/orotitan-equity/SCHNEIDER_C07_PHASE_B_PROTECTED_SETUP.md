# OroTitan Schneider C07 — Protected Phase B preflight (NOT a V2 admission)

## Authoritative scope and non-goals

- Application repository: `robzer13/indice_nexus`, C07 Phase B: standalone draft PR against main, independent of the still-unmerged #449 and #450.
- Private immutable historical artifact repository: `robzer13/real-orotitan`, **read-only**.
- Only SHADOW project: `awgsurdyvsyolcgpnygh`, Schneider run `db00ce77-04eb-492b-a76b-164ecbe9ccfe`, **SQL SELECT only**.
- PRODUCTION project `cugpgtzygqqlxetyeven`: excluded; no connection.
- The archive is a **local C03 simulation**, not a sealed historical artifact and not remotely persisted V2.
- This manual workflow only checks the 19 historical objects, 3 local candidate objects, and collects a read-only Registry snapshot. It **does not** replay the Schneider checkpoint, commit CAS 7/2 to 8/3, run the V3.1 trusted loader after local COMMIT, merge, deploy, or grant authority to V2.
- The existing C07 public CI already tests three synthetic PostgreSQL CAS rejection cases; it does not access Schneider's private bytes.

## Important GitHub workflow activation limitation

GitHub exposes `workflow_dispatch` only for workflows present on the repository default branch.
The YAML in the standalone Phase B PR **cannot be manually triggered yet**. It must undergo security review and be merged through the normal, separately authorized review route. **Do not merge PR #449 or #450 to activate this workflow**. Do **not** bypass this by enabling secret-bearing PR workflows or granting access to arbitrary branches. Until then the Phase B workflow is `PREPARED_NOT_ACTIVATED`.

## One-time GitHub environment configuration (manual)

In `robzer13/indice_nexus` → Settings → Environments, create
`orotitan-schneider-c07-readonly`. Configure:

1. **Required reviewers**: at least one trusted reviewer, and disable self-review where available.
2. **Deployment branches/tags**: selected branch **`main` only**. Do not allow PR branches, arbitrary branches or tags.
3. Environment secrets (all three stored in GitHub UI, never pasted in code/chat):
   - `C07_PRIVATE_READ_TOKEN`: fine-grained token with repository access **only** to the private `robzer13/real-orotitan` and `Contents: Read`. No `Contents: Write`, administration or actions write permission.
   - `C07_C03_BUNDLE_BASE64`: complete *single-line* Base64 value of the prebuilt minimal five-file C03 input ZIP. The source is the private attachment `OROTITAN_SCHNEIDER_C07_PHASE_B_SECRET_BASE64.txt`, not a repository file. It is approximately 17 KB, below the normal GitHub secret size limit.
   - `C07_SHADOW_RO_PGPASSWORD`: password of a *dedicated*, pre-existing PostgreSQL LOGIN granted `SELECT` only on `orotitan_runs`, `orotitan_run_stages`, `orotitan_artifacts`, `orotitan_artifact_edges` and `orotitan_run_events`, plus necessary schema/CONNECT privileges. No DML, DDL, EXECUTE on mutation RPCs or privileged Supabase key. Use a database administrator to provision this role separately; this workflow never grants privileges.
4. Environment variable (not a secret):
   - `C07_SHADOW_RO_PGUSER`: name of that dedicated read-only PostgreSQL login.

The workflow fixes the server to `db.awgsurdyvsyolcgpnygh.supabase.co:5432`, database `postgres`, with `sslmode=verify-full`, read-only session and `BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY`. If this direct endpoint is unreachable from GitHub-hosted runners (including IPv6 limitations), the job **fails closed**. Do not substitute a pooler or an unverified connection string without a fresh security review proving the target project.

## Manual dispatch (only after workflow lands on reviewed main)

GitHub → Actions → **OroTitan Schneider C07 protected private preflight** → Run workflow:
- Branch: `main`
- `confirm`: `C07_READONLY`
- `bundle_sha256`: `8e1f7b87965e8f737ae285fcb1734afe553541af4288abf9d10bda632c396928`

Approve the environment only after checking that the workflow code on `main` is the reviewed version and required protection remains enabled.

## Expected evidence and interpretation

- Pass: `19/19 historical exact-byte proofs verified`; `3/3 local simulated candidates verified`; `SHADOW SELECT-only snapshot: PASS`.
- Any mismatch/credential/network error: `BLOCKED_C07`, no fallback or mutation.
- Always: `SCHNEIDER_PG17_RPC_REPLAY=NOT_EXECUTED`; `TRUSTED_LOADER_POSTCOMMIT=NOT_EXECUTED`; `FUNDAMENTALS_V2=NO_GO`.
- No private JSON, token, Registry rows, or source bodies are uploaded as GitHub Actions artifacts or printed in logs. Everything remains in the ephemeral runner and is removed after execution.

## Remaining technical work after protected preflight

A separate reviewed implementation must build an exact local Schneider Registry instance from the captured rows with complete FK/event authority and local-only attestation receipts, then execute real `checkpoint_orotitan_stage` CAS 7/2 → 8/3, negative SQLSTATE cases, and the pinned V3.1 trusted loader against **post-COMMIT** local bytes. None of that is implemented by this preflight; no false GO.

**Status: Phase A PASS on PostgreSQL 17.12; Phase B protected preflight PREPARED_NOT_ACTIVATED; Schneider Fundamentals V2 NO-GO.**
