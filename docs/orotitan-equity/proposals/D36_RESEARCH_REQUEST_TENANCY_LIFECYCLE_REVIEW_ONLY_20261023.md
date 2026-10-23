# D36 — per-operator Research requests, scoped active idempotency, and lifecycle
**Date:** 2026-10-23. **Disposition:** stacked DRAFT on #415; **NO-GO** for hosted
Supabase migration / deployed MCP writes.

## D35 finding addressed: persistent, globally unique Research target

In the older D30 queue, `UNIQUE(kind,target_key)` permanently blocked
repeat Research for one issuer, irrespective of operator, contract pack or
analysis cutoff. This created both a cross-account denial-of-service risk
and a dead-end lifecycle: every row remained `REQUESTED_NOT_EXECUTED`.

D36 introduces a **parallel versioned Research-only** PostgreSQL 17 model,
rather than secretly rewriting D30/D32 and breaking established callers:

- `research_requests_v4` stores a stable authenticated operator subject
  + fixed MCP resource, canonical issuer ID, frozen contract-pack SHA-256,
  cutoff date and lifecycle state. Its partial unique index applies only
  to *active* requests for the **same operator + issuer + contract + cutoff**.
- Same human / different ChatGPT client with identical scope returns a
  duplicate of **their own** active request. Another OAuth subject receives
  an independent request even for the same issuer/scope (no other actor
  receipt is returned).
- Changing the frozen contract pack or cutoff is an independent request.
  Once a trusted, separately authorized controller moves a request from
  `REQUESTED_NOT_EXECUTED` to a terminal state
  (`FULFILLED`, `CANCELLED`, `REJECTED`), a fresh preview may request
  the same scope. Old request history remains intact; the original
  preview retains its own immutable idempotent receipt.
- A `research_preview_scope_v4` row binds the approved D30/D31/D34
  two-minute preview to the contract pack, cutoff and canonical issuer.
  The scope is stored **before** D32 approval and is not modifiable by
  the submitter. SQL rechecks the I1 issuer-security crosswalk at both
  scope registration and Research request confirmation.
- The Research submitter supplies only an opaque server-held D31 sealed
  capability and preview UUID, never actor, subject, scope or cutoff.
  The PostgreSQL function verifies its D32-approved SHA-256 digest under a
  row lock. This is not independent cryptographic HMAC verification (D31
  still does that in the trusted backend).

## Privilege and test boundaries

D36 introduces three `NOLOGIN NOINHERIT NOBYPASSRLS` roles:
`orotitan_d36_scope_issuer`, `orotitan_d36_research_submitter`, and
`orotitan_d36_lifecycle_controller`. Each gets only EXECUTE on its
respective private-schema SECURITY DEFINER function; the D32 private
executor owns the functions, with explicitly scoped table grants and
RLS. Neither user/browser nor Supabase `anon`/`authenticated` has a
D36 permission.

The disposable PG17 test checks the same user on **mobile and desktop**,
another user requesting the same issuer, distinct contract/cutoff,
wrong actor's capability, idempotent retry, terminal closure, and a new
request after closure. Negative assertions prohibit direct table
access and `anon`/`authenticated` EXECUTE. It verifies no Registry
run is created.

## Critical unresolved gates / NO-GO

This PR does **not** change the live D30/D32 queue or currently deployed
MCP. It supplies a candidate V4 model with tests, not a production data
migration, not a Background Research worker, and not a mobile feature.

- **D35-03 still BLOCKED:** even narrow `issue_research_scope_v4`
  accepts actor information provided by an internal trusted caller.
  Isolated credentials and a server policy tied to *verified OAuth
  subject/client/resource and a live session* are required. SQL's
  `NOLOGIN`/RLS do not authenticate these parameters.
- **D35-04 partially addressed:** the V4 model supports scoped repeat
  Research; the legacy D30 global uniqueness remains until a separately
  designed cutover and migration is authorized. V4 also requires a
  trustworthy lifecycle controller to avoid premature closure, quota
  abuse and reissuance. Terminal states reflect **queue disposition**,
  NOT evidence validation or Registry Deep Dive approval.
- **Tenancy:** one signed OAuth subject is currently used as the operator
  security boundary; no organization/team sharing, role revocation across
  subjects, subject-merge or resource migration is implemented.
- **Contract pack and cutoff:** SHA-256 and date formats are validated,
  but D36 does **not** prove that a referenced contract-pack actually
  exists, is frozen, valid for that issuer, or matches a particular run.
  The trusted operator and worker must independently enforce this.
- **Market provenance:** no ISIN checksum verification or external
  identity attribution. Fixtures use synthetic ISINs.
- **Authorization lifecycle:** server-only encrypted key storage,
  refresh/replay, rate limiting, audit logging, retention and real
  mobile end-to-end acceptance remain release blockers.
- **CI gates:** all scripts run only on disposable local PostgreSQL 17
  within GitHub Actions. No actual Supabase migration or API permission
  grant, no paid resource, no Vercel deploy, no merge, no Registry or
  Besi stage mutation. Independent security review is still required.

**Release decision remains NO-GO.** D36 is an engineering improvement,
not permission to turn on ChatGPT write commands.
