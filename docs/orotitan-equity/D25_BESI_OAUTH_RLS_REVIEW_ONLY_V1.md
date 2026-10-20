# D25 — Besi OAuth MCP read-only RLS — REVIEW ONLY / NOT APPROVED

**Status:** PROPOSAL ONLY; SQL below is **not a deployed migration**. Never apply without a separate explicit user approval for the production Supabase change, final registered OAuth client ID and a verified custom access-token hook. Keeping this file in GitHub is not authorization to run the SQL.

## Verified live baseline — 2026-10-20 (read-only)

Project: `cugpgtzygqqlxetyeven`, **not** `vnext-shadow`.

- All four required `public` tables have RLS enabled; **zero policies** on them at audit.
- `authenticated` has no SELECT privilege for the relevant tables/columns; `anon` has no table-level SELECT.
- Exactly **one confirmed Supabase Auth user**; the UUID must be transferred privately to the audited migration, never committed to this public PR.
- Exactly one Registry run `48b9426b-4d0d-4858-9d87-efcd13ac415d`: ACTIVE, RESEARCH, state_version 6, CHECKPOINT, `BLOCKED_INSUFFICIENT_INPUT`, handoff NO, **6 blocker entries**, one active stage.
- The public Vercel OAuth pilot has no database credentials; no OAuth app is yet registered/confirmed.
- The public read-only MCP route uses the caller's validated bearer JWT and a publishable key under RLS. The privileged verified-next-prompt and private quarantine paths are disabled in the delegated mode.

## Exact access surface

| Table | SELECT columns to `authenticated` | Visible rows with admitted OAuth JWT |
|---|---|---|
| `public.orotitan_runs` | `run_id, issuer_id, security_id, run_status, current_stage, state_version, data_cutoff, process_version, pilotage_contract_version, canonical_mode` | Exactly the ACTIVE/RESEARCH Besi run |
| `public.orotitan_run_stages` | `run_id, stage_code, lifecycle_status, contract_status_code, handoff_gate_name, handoff_gate_state, active_manifest_kind, blocker_summary, state_version` | Only Besi's RESEARCH stage, while its run remains admitted |
| `public.issuers` | `issuer_id, display_name` | Only the issuer linked by the admitted Besi run |
| `public.securities` | `security_id, issuer_id, ticker, isin` | Only the security linked by the admitted Besi run |

No INSERT, UPDATE, DELETE, broad table-level SELECT, `anon` grant, view, function, SECURITY DEFINER, `service_role` credential, or access to other run artifacts is part of this proposal.

## Draft SQL — DO NOT EXECUTE YET

This text contains deliberately **invalid placeholders** for the *actual* user UUID and *registered* OAuth client ID. Never replace them in a public Git commit or share a credential in ChatGPT. The migration executor must perform those substitutions only after final security approval. **Do not grant access until the OAuth access token has a signed client-specific `resource` claim** bound to the pilot's MCP endpoint, and the client is allowlisted in the resource server too.

```sql
-- FUTURE MIGRATION DESIGN ONLY / NOT EXECUTED.
-- Preconditions: verify zero existing policies on all four tables,
-- RLS remains ON, exact table schemas, single Besi run in RESEARCH,
-- approved confirmed Auth user, registered OAuth client, signed resource claim.
-- Apply in one reviewed transaction only after approval.

begin;

grant select (
  run_id, issuer_id, security_id, run_status, current_stage,
  state_version, data_cutoff, process_version,
  pilotage_contract_version, canonical_mode
) on table public.orotitan_runs to authenticated;

grant select (
  run_id, stage_code, lifecycle_status, contract_status_code,
  handoff_gate_name, handoff_gate_state, active_manifest_kind,
  blocker_summary, state_version
) on table public.orotitan_run_stages to authenticated;

grant select (issuer_id, display_name)
  on table public.issuers to authenticated;

grant select (security_id, issuer_id, ticker, isin)
  on table public.securities to authenticated;

create policy mcp_besi_run_read_v1
  on public.orotitan_runs
  as permissive for select to authenticated
  using (
    run_id = '48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid
    and run_status = 'ACTIVE'
    and current_stage = 'RESEARCH'
    and (select auth.uid()) = '__APPROVED_SUPABASE_AUTH_USER_UUID__'::uuid
    and (select auth.jwt() ->> 'client_id') = '__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__'
    and (select auth.jwt() ->> 'resource') = 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
    and coalesce((select auth.jwt() ->> 'is_anonymous'), 'false') = 'false'
  );

create policy mcp_besi_stage_read_v1
  on public.orotitan_run_stages
  as permissive for select to authenticated
  using (
    run_id = '48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid
    and stage_code = 'RESEARCH'
    and (select auth.uid()) = '__APPROVED_SUPABASE_AUTH_USER_UUID__'::uuid
    and (select auth.jwt() ->> 'client_id') = '__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__'
    and (select auth.jwt() ->> 'resource') = 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
    and coalesce((select auth.jwt() ->> 'is_anonymous'), 'false') = 'false'
    and exists (
      select 1 from public.orotitan_runs r
      where r.run_id = '48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid
      and r.run_status = 'ACTIVE'
      and r.current_stage = 'RESEARCH'
        and r.run_id = orotitan_run_stages.run_id
    )
  );

create policy mcp_besi_issuer_read_v1
  on public.issuers
  as permissive for select to authenticated
  using (
    (select auth.uid()) = '__APPROVED_SUPABASE_AUTH_USER_UUID__'::uuid
    and (select auth.jwt() ->> 'client_id') = '__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__'
    and (select auth.jwt() ->> 'resource') = 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
    and coalesce((select auth.jwt() ->> 'is_anonymous'), 'false') = 'false'
    and exists (
      select 1 from public.orotitan_runs r
      where r.run_id = '48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid
      and r.run_status = 'ACTIVE'
      and r.current_stage = 'RESEARCH'
        and r.issuer_id = issuers.issuer_id
    )
  );

create policy mcp_besi_security_read_v1
  on public.securities
  as permissive for select to authenticated
  using (
    (select auth.uid()) = '__APPROVED_SUPABASE_AUTH_USER_UUID__'::uuid
    and (select auth.jwt() ->> 'client_id') = '__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__'
    and (select auth.jwt() ->> 'resource') = 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
    and coalesce((select auth.jwt() ->> 'is_anonymous'), 'false') = 'false'
    and exists (
      select 1 from public.orotitan_runs r
      where r.run_id = '48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid
      and r.run_status = 'ACTIVE'
      and r.current_stage = 'RESEARCH'
        and r.security_id = securities.security_id
        and r.issuer_id = securities.issuer_id
    )
  );

commit;
```

**Important policy composition note:** PostgreSQL permissive policies combine with OR. Zero policies existed at this audit; **re-check all policies and grants immediately before applying**. Any new/broad permissive policy or role grant changes the security analysis and invalidates this reviewed proposal.

## Required authorization & negative tests before release

1. Confirm a real registered `client_id` matching the ChatGPT OAuth client and the allowlist configured in the consent UI and MCP JWT validator.
2. Verify a Supabase-signed ES256 OAuth access token with `sub` of the selected user, `client_id`, and `resource` exactly equal to `https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp`. The resource claim must be supplied by a reviewed, **client-specific** Custom Access Token Hook; do **not** disable the JWT resource check to make a test pass. Supabase OAuth scopes like `openid email profile` do not themselves authorize table access.
3. Verify RLS as `anon`, a different authenticated user, the same user with a normal Supabase Auth token (without OAuth `client_id/resource`), a different OAuth client, an absent or wrong resource, and a different run ID. **All must see zero rows or permission denied.**
4. Approved client/user/resource must read **one** Besi run, **one** Research stage, **one** issuer, **one** security, **six** blockers. Other run IDs must read zero rows.
5. Confirm `SELECT *` is not a granted privilege, and INSERT/UPDATE/DELETE and access to unrelated tables remain denied. Confirm no public `anon` reader and no service-role fallback in the public MCP route.
6. Confirm user approval of any shared-project Supabase Auth Site URL change separately; do not replace `localhost` by the public pilot URL without checking all other authentication consumers. A separate Supabase Auth test project may be safer if shared-site URL changes cannot be isolated.
7. Verify CI and a single user-authorized Vercel deployment **only after** the actual OAuth test configuration is internally reviewed. Stay on draft PR until E2E tests succeed.

## Rollback design (separately reviewed)

Revoke **only** the exact column-level grants introduced here and drop the four named `mcp_besi_*_read_v1` policies, **after re-checking** whether other consumers have gained legitimate dependencies or new grants. Do not disable RLS, delete rows, alter Registry stage, rotate keys or revoke unrelated policies. The prechange baseline snapshot must be retained.

## Sources

- Supabase OAuth Server / Token Security and RLS: https://supabase.com/docs/guides/auth/oauth-server/token-security
- Supabase RLS policy guidance: https://supabase.com/docs/guides/database/postgres/row-level-security
- Existing source: `lib/orotitan-equity/v3/mcp-bridge-service-v1.ts`, `lib/orotitan-equity/v3/mcp-oauth-v1.ts`, `app/api/orotitan/mcp/route.ts`
