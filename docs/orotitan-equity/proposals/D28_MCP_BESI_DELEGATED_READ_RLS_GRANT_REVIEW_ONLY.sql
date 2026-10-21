-- D28 — REVIEW ONLY. NOT a production migration. DO NOT apply without
-- a separate owner-authorized rollout and a final live preflight.
-- Replace ONLY the 3 exact placeholders in a controlled deployment process.
-- Explicitly designed for the isolated ChatGPT OAuth V2 Besi MCP pilot.
-- This never authorizes writes, stage transitions, artifact reads or publication.
--
-- Source of truth: verified live schema/ACL audit 2026-10-21 (read-only).
-- Never publish real Supabase user identifiers in this public repository.
begin;

do $d28_preflight$
declare
  target text;
begin
  if current_setting('server_version_num')::int < 170000 then
    raise exception 'D28 requires PostgreSQL 17+';
  end if;
  if not exists (select 1 from pg_roles where rolname='authenticated')
    or not exists (select 1 from pg_roles where rolname='anon') then
    raise exception 'D28 requires Supabase authenticated/anon roles';
  end if;
  if to_regprocedure('auth.uid()') is null
     or to_regprocedure('auth.jwt()') is null then
    raise exception 'D28 requires Supabase auth.uid() and auth.jwt()';
  end if;
  foreach target in array array[
    'orotitan_runs','orotitan_run_stages','issuers','securities'
  ] loop
    if not exists (
      select 1 from pg_class c
      join pg_namespace n on n.oid=c.relnamespace
      where n.nspname='public' and c.relname=target
        and c.relkind in ('r','p') and c.relrowsecurity
    ) then
      raise exception 'D28: expected RLS table public.% absent or changed', target;
    end if;
    if has_table_privilege('authenticated',format('public.%I',target),'SELECT')
       or has_table_privilege('authenticated',format('public.%I',target),'INSERT')
       or has_table_privilege('authenticated',format('public.%I',target),'UPDATE')
       or has_table_privilege('authenticated',format('public.%I',target),'DELETE')
       or has_table_privilege('anon',format('public.%I',target),'SELECT') then
      raise exception 'D28: unexpected pre-existing privilege on public.%',target;
    end if;
    if exists (
      select 1 from pg_policies where schemaname='public' and tablename=target
    ) then
      raise exception 'D28: pre-existing RLS policy on public.% requires review',target;
    end if;
  end loop;
end;
$d28_preflight$;

-- Column-level SELECT only; no unrestricted table SELECT.
grant select (
  run_id,issuer_id,security_id,run_status,current_stage,state_version,
  data_cutoff,process_version,pilotage_contract_version,canonical_mode
) on public.orotitan_runs to authenticated;

grant select (
  run_id,stage_code,lifecycle_status,contract_status_code,
  handoff_gate_name,handoff_gate_state,active_manifest_kind,
  blocker_summary,state_version
) on public.orotitan_run_stages to authenticated;

grant select (issuer_id,display_name)
  on public.issuers to authenticated;
grant select (security_id,issuer_id,ticker,isin)
  on public.securities to authenticated;

-- Four independent, fail-closed row predicates. Each explicitly checks the
-- signed Supabase identity, fixed OAuth client, resource binding and Besi run.
-- A standard authenticated session lacking client_id/resource gets zero rows.
create policy orotitan_mcp_besi_read_v1 on public.orotitan_runs
for select to authenticated using (
  run_id = '__BESI_RUN_UUID__'::uuid
  and (select auth.uid()) = '__APPROVED_SUPABASE_AUTH_USER_UUID__'::uuid
  and (select auth.jwt()->>'client_id') = '__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__'
  and (select auth.jwt()->>'resource') =
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
);

create policy orotitan_mcp_besi_read_v1 on public.orotitan_run_stages
for select to authenticated using (
  run_id = '__BESI_RUN_UUID__'::uuid
  and (select auth.uid()) = '__APPROVED_SUPABASE_AUTH_USER_UUID__'::uuid
  and (select auth.jwt()->>'client_id') = '__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__'
  and (select auth.jwt()->>'resource') =
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
);

-- Labels: no direct arbitrary issuer/security access. A row is visible only
-- if it belongs to the same RLS-visible Besi run (including its client/subject
-- and resource restrictions, inherited through the subquery).
create policy orotitan_mcp_besi_read_v1 on public.issuers
for select to authenticated using (
  exists (
    select 1 from public.orotitan_runs r
    where r.run_id = '__BESI_RUN_UUID__'::uuid
      and r.issuer_id = issuers.issuer_id
  )
);

create policy orotitan_mcp_besi_read_v1 on public.securities
for select to authenticated using (
  exists (
    select 1 from public.orotitan_runs r
    where r.run_id = '__BESI_RUN_UUID__'::uuid
      and r.security_id = securities.security_id
      and r.issuer_id = securities.issuer_id
  )
);

commit;
