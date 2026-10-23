# D30 — ChatGPT mobile multi-device command intents V1 (REVIEW ONLY)

**2026-10-23.** Stacked DRAFT on PR #409; this proposal is never merged,
migrated or deployed without separate security and acceptance approval.

## Objective

A single ChatGPT-connected OroTitan cockpit can be used from a personal PC,
work PC and phone. The app and persistent state will be centralized, not hosted
on whichever computer runs Docker. Local Docker/PG17 are for development only.
No installation on a work computer and no public access to a developer DB.

Two commands, **not yet exposed** by the OAuth MCP server:

| Proposed command | First version's result | NOT a result |
| --- | --- | --- |
| `add_company` | Confirmed onboarding request, to be reviewed and reconciled with Screener and I1 `issuers/securities` | Not a `companies`, `issuers` or `securities` INSERT |
| `request_research` | Confirmed durable request for a verified canonical `issuer_id` | Not `create_orotitan_run`, no new Research stage, no analysis execution |

This preserves I1's **ISSUER != SECURITY** boundary. Screener tracking rows
are not automatically identical to canonical issuers or Registry runs.

## Exact two-phase operator flow

1. `preview_add_company(name,ticker,exchange,isin)` or
   `preview_request_research(issuer_id)` — server-authenticated actor bound
   to subject, approved ChatGPT client and fixed MCP resource. Require an
   unambiguous identity lookup (or a registered I1 issuer/security).
   Receive a fresh, 2-minute, fingerprinted preview and explicit summary.
2. Operator expressly approves the **exact preview**, not an LLM-implied
   confirmation. `add_company(previewId,fingerprint,"CONFIRM")` or
   `request_research(...)` — recheck actor, preview, expiry, fingerprint,
   current canonical identity and duplicate target **inside one private
   transaction**. Display the durable receipt, status
   `REQUESTED_NOT_EXECUTED`, and never claim the company was inserted or
   research started.
3. Future separately approved worker will reconcile/onboard the company
   or validate admission to `create_orotitan_run`. It must bind a frozen
   contract-pack, exact issuer and data cutoff and produce Registry events
   via the canonical RPC. A request is NEVER authority to run, register a
   dossier, admit evidence, advance a stage or publish.

**Identity assurance caveat:** an ISIN pattern match is not independent
issuer/security verification, and the checker currently validates formatting
rather than the ISO 6166 checksum. The private identity resolver MUST compare
against independently grounded listing/security data and reject uncertainty.
The review-only fixture has a simulated resolver; no public market lookup is
performed. A staged future implementation must also implement checksum and
source-provenance checks.

## Current implementation

- `lib/orotitan-equity/v3/mobile-command-intents-candidate-v1.ts` has pure
  input validation, independent authorization-port integration, preview
  fingerprinting, explicit approval, TTL and non-executable receipt checks.
- `tests/equity-v3-mobile-command-intents-candidate.test.ts` runs offline
  for forgery, stale previews, duplicate operations, cross-user identity,
  canonical drift and *zero real execution*; mock state is not durable.
- `docs/orotitan-equity/proposals/D30_MOBILE_REQUEST_QUEUE_SQL_REVIEW_ONLY.sql`
  models a private, RLS-enabled PostgreSQL queue. It grants nothing to
  `PUBLIC`, `anon` or `authenticated`, exposes no production RPC,
  and persists operator requests in a **disposable PG17 test DB** only.
  The local test superuser acts as a synthetic trusted backend; that is
  NOT a production role model.
- The SQL queue reconfirms current I1 issuer/security identity for Research;
  rejects unknown ISIN and already-tracked equities for onboarding;
  enforces one active request per target via a unique index;
  rejects stale actor, client, resource, hash and preview.
- `npm test` and `npm run test:postgres` run the new tests in the existing
  GitHub Actions CI. No paid Supabase plan, hosted branch or live changes.

## Open blockers before activating actual MCP writes

1. An independently reviewed, dedicated **trusted command issuer** must
   cryptographically bind verified OAuth subject/client/resource, permissions,
   cost and explicit user confirmation to each operation. Current MCP
   `mcpWriteEnabled()` is hardcoded **false** by design; do not change this
   flag as part of D30.
2. Independently verified identifier resolution: ISO 6166 checksum,
   legal issuer security/listing evidence, duplicate crosswalk, ambiguous
   dual listings and ticker reuse. No ticker-only creation.
3. Private runtime SQL principal with least privilege, rate limits, audit,
   transactional idempotency, RLS and immutable confirmation proof; no
   service key in the public MCP client; no direct table permissions.
4. Persisted work queue independent of HTTP request/phone lifetime, and
   separate reviewed worker/orchestrator for any Registry mutation. A DB row
   marked `REQUESTED_NOT_EXECUTED` must NOT be represented as a running job.
5. Mobile ChatGPT connector capability and exact supported OAuth scopes need
   end-to-end user testing on real iOS/Android. Passing local tests does not
   imply availability in the mobile app.
6. No personal account information, secrets, bank details or business
   confidential research should be copied onto an employer-managed computer
   without the employer's policy approval.

## Acceptance / negative assertions

No live `companies`, `issuers`, `securities`,
`orotitan_runs`, `orotitan_run_stages` or `research_snapshots`
are touched. No company is added and no Research starts in production.
No OAuth write tool is advertised by the deployed pilot. Original Besi
run stays RESEARCH, blocker state unchanged. All draft PRs unmerged.
