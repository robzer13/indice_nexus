# D32 — PostgreSQL private mobile intent approval + submitter gate
**2026-10-23 — REVIEW ONLY, DISPOSABLE LOCAL PG17.** Stacked on D31/#411.
No Supabase branch, paid plan, hosted migration, cloud write role, deployed MCP
tool, Vercel deployment or OroTitan Registry change.

## Security objective

The D30 function `confirm_request_v1(preview,subject,client,resource,hash)`
takes caller-supplied identity and must NEVER receive `EXECUTE` from
browser, OAuth `authenticated`, ChatGPT MCP, or a general runtime role.

D32 proposes exactly two narrow role-facing procedures inside a
**non-exposed private schema**:

1. `approve_preview_v2` — a separately authenticated **trusted authority**
   supplies the D31 preview identity, kind, fingerprint and a **SHA-256 digest
   of the server-only D31 sealed HMAC capability**. This approval function
   checks every stored preview field, its two-minute expiry, revocation,
   previous confirmation, and disallows replacing an existing proof.
2. `confirm_approved_request_v2(previewId,sealedCapability)` — the
   **submitter** can supply only the preview ID and secret server capability.
   It cannot choose `subject`, `client_id`, `resource`, `kind` or
   `fingerprint`. The non-login definer retrieves those fields from a
   pre-authorized private preview, verifies the capability digest, expiry
   and revocation under a row lock, and calls the D30 original immutable
   request routine inside the same transaction.

The three roles are `NOLOGIN NOINHERIT NOBYPASSRLS`:
`orotitan_d32_approval_issuer`, `orotitan_d32_command_submitter`
and `orotitan_d32_private_executor`. Runtime caller roles receive
**EXECUTE-only** on their respective functions, no direct table SELECT,
INSERT, UPDATE or DELETE and no direct EXECUTE on D30's forgeable function.
The definer has only narrowly granted D30 tables, canonical I1 identity
reads and explicit RLS policies; functions have locked `search_path`.
Their owner is non-login/non-bypass. No `PUBLIC` or `authenticated`
EXECUTE privileges.

## Exact guarantee and boundaries

A signed OAuth `client_id` is not in itself an authorization. D31's
independent server policy must bind OAuth subject/client/resource and
explicit operator confirmation before an approval is registered. The
approval role is HIGH-TRUST: if compromised, it can mint approval digests
for private previews. Both roles, the database connection pool and
secret lifecycle require separate isolation and operational review.

The D32 DB checks a **registered SHA-256 digest of the sealed capability**.
It does *not* parse or cryptographically verify the underlying HMAC itself,
and bytea equality is not promised constant-time. The server holds the
only copy of the sealed token and high-entropy HMAC key. For public exposure
further checks must cover nonce replay/one-use semantics, rate limiting,
transactional audit, revocation, rotation, queue status and authentication
session lifecycle. The current fixture permits same-preview idempotent
retry inside the short approval window; it never executes Research.

## Offline tests and limits

`tests/postgres/run-d32-mobile-private-writer-tests.sh` installs D30
and D32 proposals in a fresh **localhost PG17** database, inserts only
synthetic identity/previews, exercises the two roles using `SET ROLE`,
and drops the database and its cluster-local test roles. It must refuse
remote hosts and role collisions. The matrix checks:

- `anon`/`authenticated` and direct D30 function access denied;
- issuer cannot tamper actor or replace an approval digest;
- submitter cannot confirm before authorization, forge a capability,
  reuse another preview's capability or bypass revocation/expiry;
- valid `add_company`/`request_research` each create a durable request
  **only**, with idempotent retry and no company/issuer/security or
  Registry mutation.

CI runs alongside TypeScript and previous local Registry, D29b, D30 suites.
A local privileged PostgreSQL fixture is NOT a hosted Supabase RLS/security
validation. No D32 schema has been applied to the connected project.

## Production NO-GO / next gates

Before any real MCP write:
- Real authenticated operator confirmation and encrypted server-side token
  storage, separate issuer/submitter database credentials, approved role
  provisioning and revocation/audit.
- Independent ISIN checksum and listing/issuer evidence resolver; prohibit
  automated `issuers`, `securities` or `companies` writes on ambiguous data.
- Independent security review of D30 and D32 SQL for privilege escalation,
  tenancy and race conditions, and real hosted RLS Storage verification.
- **Explicit separate user authorization** for any production Supabase
  migration, new credentials, Vercel deployment or MCP write activation.

Besi remains RESEARCH / `READY_FOR_DEEP_DIVE=NO`; D32 does not
admit evidence, start a run, advance a stage, or publish a report.
