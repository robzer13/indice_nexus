-- OroTitan Registry V1.5 — required contract-pin completeness guard.
-- CODE-ONLY IMPLEMENTATION. NOT APPLIED TO PRODUCTION.

begin;

create or replace function public.orotitan_contract_pins_complete(p_contract_pins jsonb)
returns boolean
language plpgsql
immutable
set search_path = pg_catalog, public
as $$
declare
  v_required constant text[] := array[
    'process',
    'pilotage',
    'research_stage',
    'deep_dive_stage',
    'integration_stage',
    'analysis_standard',
    'master_prompt',
    'investment_policy',
    'execution_patch',
    'integration_spec',
    'screener_schema',
    'i2',
    'i3b'
  ];
  v_key text;
  v_pin jsonb;
  v_locator jsonb;
begin
  if p_contract_pins is null or jsonb_typeof(p_contract_pins) <> 'object' then
    return false;
  end if;

  foreach v_key in array v_required
  loop
    if not (p_contract_pins ? v_key) then return false; end if;
    v_pin := p_contract_pins->v_key;
    if jsonb_typeof(v_pin) <> 'object'
       or jsonb_typeof(v_pin->'name') <> 'string'
       or jsonb_typeof(v_pin->'version') <> 'string'
       or jsonb_typeof(v_pin->'content_sha256') <> 'string'
       or (v_pin->>'content_sha256') !~ '^[0-9a-f]{64}$'
       or jsonb_typeof(v_pin->'locator') <> 'object' then
      return false;
    end if;

    v_locator := v_pin->'locator';
    if v_locator->>'backend' = 'GITHUB_IMMUTABLE' then
      if jsonb_typeof(v_locator->'repository') <> 'string'
         or jsonb_typeof(v_locator->'path') <> 'string'
         or (v_locator->>'commit_sha') !~ '^[0-9a-f]{40}$'
         or (v_locator->>'blob_sha') !~ '^[0-9a-f]{40}$' then
        return false;
      end if;
    elsif v_locator->>'backend' = 'HASH_ADDRESSED_PRIVATE_STORE' then
      if jsonb_typeof(v_locator->'storage_uri') <> 'string'
         or length(btrim(v_locator->>'storage_uri')) = 0 then
        return false;
      end if;
    else
      return false;
    end if;
  end loop;

  return true;
end;
$$;

alter table public.orotitan_runs
  add constraint orotitan_runs_contract_pins_complete_check
  check (public.orotitan_contract_pins_complete(contract_pins));

revoke all on function public.orotitan_contract_pins_complete(jsonb)
  from public, anon, authenticated, service_role;

comment on function public.orotitan_contract_pins_complete(jsonb) is
  'Validates the 13 required frozen execution/method contract pins and durable locator shape for a live OroTitan run.';

commit;
