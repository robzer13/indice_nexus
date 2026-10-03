-- OroTitan Pilotage Anti-Loop / Lossless Resume V1 PostgreSQL regression.

do $$
declare
  v_run uuid := gen_random_uuid();
  v_source public.orotitan_runs%rowtype;
  v_result jsonb;
  v_wrong_route_rejected boolean := false;
  v_stale_rejected boolean := false;
begin
  select *
    into v_source
  from public.orotitan_runs
  where issuer_id is not null
    and security_id is not null
    and dossier_id is not null
  order by created_at
  limit 1;

  if not found then
    raise exception 'anti-loop fixture requires an existing bound OroTitan run';
  end if;

  insert into public.orotitan_runs (
    run_id,
    run_display_key,
    creation_idempotency_key,
    run_scope,
    issuer_id,
    security_id,
    dossier_id,
    parent_run_id,
    baseline_snapshot_id,
    upstream_discovery_artifact_id,
    upstream_discovery_artifact_version,
    entry_path,
    canonical_mode,
    run_type,
    run_status,
    current_stage,
    data_cutoff,
    process_version,
    pilotage_contract_version,
    contract_pins,
    contract_set_sha256,
    state_version
  ) values (
    v_run,
    null,
    'anti-loop-test:' || v_run::text,
    'COMPANY_ANALYSIS',
    v_source.issuer_id,
    v_source.security_id,
    v_source.dossier_id,
    null,
    null,
    null,
    null,
    'IMPOSED_COMPANY',
    'ANALYZE',
    'INITIAL',
    'ACTIVE',
    'RESEARCH',
    v_source.data_cutoff,
    v_source.process_version,
    v_source.pilotage_contract_version,
    v_source.contract_pins,
    v_source.contract_set_sha256,
    1
  );

  insert into public.orotitan_run_stages (
    run_id,
    stage_code,
    stage_revision,
    stage_contract_name,
    stage_contract_version,
    stage_contract_sha256,
    lifecycle_status,
    contract_status_code,
    handoff_gate_name,
    handoff_gate_state,
    active_manifest_artifact_id,
    active_manifest_version,
    active_manifest_kind,
    blocker_summary,
    state_version,
    started_at,
    completed_at
  ) values (
    v_run,
    'RESEARCH',
    1,
    v_source.contract_pins->'research_stage'->>'name',
    v_source.contract_pins->'research_stage'->>'version',
    v_source.contract_pins->'research_stage'->>'content_sha256',
    'IN_PROGRESS',
    null,
    'READY_FOR_DEEP_DIVE',
    'NOT_EVALUATED',
    null,
    null,
    null,
    '[]'::jsonb,
    1,
    now(),
    null
  );

  v_result := public.register_orotitan_pilotage_attempt(
    v_run,
    'RESEARCH',
    1,
    1,
    repeat('a',64),
    'SAVE_DURABLE_CHECKPOINT',
    'SAVE_DURABLE_CHECKPOINT'
  );

  if v_result->>'decision' <> 'FIRST_ATTEMPT'
     or (v_result->>'attempt_count')::bigint <> 1
     or (v_result->>'retry_without_reload_allowed')::boolean then
    raise exception 'first anti-loop continuation attempt invalid: %', v_result;
  end if;

  v_result := public.register_orotitan_pilotage_attempt(
    v_run,
    'RESEARCH',
    1,
    1,
    repeat('a',64),
    'SAVE_DURABLE_CHECKPOINT',
    'SAVE_DURABLE_CHECKPOINT'
  );

  if v_result->>'decision' <> 'NO_PROGRESS_REPLAY'
     or (v_result->>'attempt_count')::bigint <> 2
     or (v_result->>'retry_without_reload_allowed')::boolean then
    raise exception 'same-state replay was not stopped: %', v_result;
  end if;

  begin
    perform public.register_orotitan_pilotage_attempt(
      v_run,
      'RESEARCH',
      1,
      1,
      repeat('b',64),
      'CONTINUE_STAGE',
      'CONTINUE_STAGE'
    );
  exception when check_violation then
    v_wrong_route_rejected := true;
  end;
  if not v_wrong_route_rejected then
    raise exception 'wrong continuation route was admitted';
  end if;

  begin
    perform public.register_orotitan_pilotage_attempt(
      v_run,
      'RESEARCH',
      2,
      1,
      repeat('c',64),
      'SAVE_DURABLE_CHECKPOINT',
      'SAVE_DURABLE_CHECKPOINT'
    );
  exception when serialization_failure then
    v_stale_rejected := true;
  end;
  if not v_stale_rejected then
    raise exception 'stale run state was admitted';
  end if;

  if (select count(*) from public.orotitan_pilotage_attempts where run_id=v_run) <> 1 then
    raise exception 'rejected continuation attempts mutated the durable ledger';
  end if;

  if (select attempt_count from public.orotitan_pilotage_attempts where run_id=v_run) <> 2 then
    raise exception 'durable anti-loop attempt counter invalid';
  end if;
end $$;

do $$
declare
  v_rls boolean;
  v_policy_count integer;
  v_write_grants integer;
  v_fn regprocedure :=
    'public.register_orotitan_pilotage_attempt(uuid,text,bigint,bigint,text,text,text)'::regprocedure;
  v_prosecdef boolean;
  v_proconfig text[];
begin
  select relrowsecurity
    into v_rls
  from pg_class c
  join pg_namespace n on n.oid=c.relnamespace
  where n.nspname='public'
    and c.relname='orotitan_pilotage_attempts';

  if v_rls is distinct from true then
    raise exception 'anti-loop ledger RLS is not enabled';
  end if;

  select count(*)
    into v_policy_count
  from pg_policies
  where schemaname='public'
    and tablename='orotitan_pilotage_attempts';

  if v_policy_count <> 0 then
    raise exception 'unexpected client policies on anti-loop ledger: %', v_policy_count;
  end if;

  select count(*)
    into v_write_grants
  from information_schema.role_table_grants
  where table_schema='public'
    and table_name='orotitan_pilotage_attempts'
    and grantee in ('anon','authenticated','service_role')
    and privilege_type in ('INSERT','UPDATE','DELETE','TRUNCATE');

  if v_write_grants <> 0 then
    raise exception 'direct anti-loop ledger write grants exist: %', v_write_grants;
  end if;

  select p.prosecdef, p.proconfig
    into v_prosecdef, v_proconfig
  from pg_proc p
  where p.oid=v_fn;

  if not v_prosecdef
     or not (v_proconfig @> array['search_path=pg_catalog, public']::text[]) then
    raise exception 'anti-loop RPC security-definer boundary invalid';
  end if;

  if has_function_privilege('public', v_fn::text, 'EXECUTE')
     or has_function_privilege('anon', v_fn::text, 'EXECUTE')
     or has_function_privilege('authenticated', v_fn::text, 'EXECUTE')
     or not has_function_privilege('service_role', v_fn::text, 'EXECUTE') then
    raise exception 'anti-loop RPC privileges invalid';
  end if;
end $$;

select jsonb_build_object(
  'table', 'orotitan_pilotage_attempts',
  'same_state_same_operation', 'NO_PROGRESS_REPLAY',
  'route_guard', 'PASS',
  'cas_guard', 'PASS',
  'client_write_firewall', 'PASS',
  'result', 'PASS'
) as orotitan_pilotage_anti_loop_v1_verification;
