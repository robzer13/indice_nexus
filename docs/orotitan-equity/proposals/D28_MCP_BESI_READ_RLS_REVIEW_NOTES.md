# D28 — Besi OAuth delegated Registry read, REVIEW ONLY

**Status:** PROPOSAL / NOT DEPLOYED  
**Scope:** `OroTitan Besi — OAuth V2` isolated ChatGPT MCP pilot only.  
**Target:** Supabase project `orotitan-screener` (project ref intentionally omitted from operational SQL).
**No production DDL, no changes to RLS, no service-role shortcut, no writes.**

## Live observations, 2026-10-21

- `public.orotitan_runs`, `orotitan_run_stages`, `issuers`, `securities` all exist and have RLS enabled.
- The `authenticated` and `anon` roles have no `SELECT` on those tables.
- No pre-existing `pg_policies` on these four tables.
- The Besi run exists with status `ACTIVE` and current stage `RESEARCH`.
- A real, authenticated MCP tool call `get_orotitan_run` returned `REGISTRY_UNAVAILABLE`. The route reaches the database under the caller's JWT; the failed `orotitan_runs` read explains the error.

## Proposed least privilege

The SQL in `D28_MCP_BESI_DELEGATED_READ_RLS_GRANT_REVIEW_ONLY.sql` grants **column-level SELECT only**, not full-table SELECT, for the four tables read by the existing `get_orotitan_run` implementation.

The `orotitan_runs` and `orotitan_run_stages` rows must satisfy ALL:
- exact approved Supabase Auth user via `auth.uid()`,
- exact pre-registered OAuth `client_id` via `auth.jwt()`,
- exact signed `resource` = isolated MCP service URL,
- exact allowed Besi `run_id`.

The `issuers` and `securities` labels become readable only when they belong to the same visible Besi run via an RLS-filtered `orotitan_runs` subquery. No other company label is exposed. No grants on `orotitan_artifacts`, `orotitan_run_events`, storage, private Research materials or RPCs. No `INSERT`, `UPDATE`, `DELETE` or `EXECUTE` authority granted.

The SQL has **three placeholders** for the precise run, user and client identities. Keep their real values out of the public repository and review them via a separate, owner-authorized rollout. It is not automatically part of `migrations/`.

## Safety limitations and rollout gate

- `get_orotitan_next_prompt` and `list_orotitan_submissions` remain deliberately restricted in OAuth mode. D28 does not enable them.
- A successful `get_orotitan_run` is an observation, **not** an artifact validation or stage admission.
- The tests simulate trusted Supabase JWT claims on isolated PostgreSQL 17; they do **not** prove real API behavior end-to-end.
- Before live installation, inspect effective RLS, role inheritance, policies and column grants again; compare the current production schema; keep rollout transaction atomic.
- Run a protected, operator-authorized rollout with a test token for the exact OAuth identity, verify only Besi is visible; verify other authenticated sessions remain denied.
- If the rollout is not authorized or preflight fails, remain fail-closed. Changes require a **separate explicit approval**.

## Local PG17 verification

`bash tests/postgres/run-mcp-besi-rls-grant-proposal-tests.sh`

The runner refuses non-loopback PostgreSQL and non-fixture `PGUSER`. It creates and drops a test-only DB with synthetic identities, applies the proposal there and tests positive/negative RLS and column grants.
