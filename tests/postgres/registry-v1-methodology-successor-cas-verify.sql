\set ON_ERROR_STOP on

do $$
declare
  v_fn regprocedure := 'public.create_orotitan_methodology_successor_run(text,uuid,text,text,text,date,uuid,uuid,text,text,jsonb,text,text,bigint,text,text,text,uuid,uuid)'::regprocedure;
begin
  if not (select prosecdef from pg_proc where oid=v_fn) then
    raise exception 'successor RPC must be SECURITY DEFINER';
  end if;
  if not ((select proconfig from pg_proc where oid=v_fn) @> array['search_path=pg_catalog, public']::text[]) then
    raise exception 'successor RPC search_path guard missing';
  end if;
  if has_function_privilege('public',v_fn,'EXECUTE')
     or has_function_privilege('anon',v_fn,'EXECUTE')
     or has_function_privilege('authenticated',v_fn,'EXECUTE')
     or not has_function_privilege('service_role',v_fn,'EXECUTE') then
    raise exception 'successor RPC privilege boundary invalid';
  end if;
end;
$$;

do $$
declare
  v_parent_id uuid := '31000000-0000-4000-8000-000000000001';
  v_identity record;
  v_old_pins jsonb;
  v_new_pins jsonb;
  v_new_hash text;
  v_result jsonb;
  v_child public.orotitan_runs%rowtype;
  v_failed boolean;
begin
  select d.issuer_id, s.security_id, d.dossier_id
    into v_identity
  from public.research_dossiers d
  join public.securities s on s.issuer_id=d.issuer_id
  where d.current_snapshot_id is null
  order by d.created_at, s.created_at
  limit 1;

  if not found then
    raise exception 'test fixture requires one dossier without current snapshot';
  end if;

  v_old_pins := jsonb_build_object(
    'process', jsonb_build_object('name','OLD_PROCESS','version','2.0','content_sha256',repeat('1',64))
  );
  v_new_pins := jsonb_build_object(
    'process', jsonb_build_object('name','NEW_PROCESS','version','3.0','content_sha256',repeat('2',64)),
    'dcf_timing', jsonb_build_object('name','OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0','version','1.0','content_sha256',repeat('3',64))
  );
  v_new_hash := public.orotitan_contract_set_sha256(v_new_pins);

  insert into public.orotitan_runs(
    run_id, creation_idempotency_key, issuer_id, security_id, dossier_id,
    parent_run_id, baseline_snapshot_id, entry_path, canonical_mode, run_type,
    run_status, current_stage, data_cutoff, process_version,
    pilotage_contract_version, contract_pins, contract_set_sha256, state_version
  ) values (
    v_parent_id, 'test:parent:dcf-cas', v_identity.issuer_id, v_identity.security_id,
    v_identity.dossier_id, null, null, 'IMPOSED_COMPANY','ANALYZE','INITIAL',
    'ACTIVE','DEEP_DIVE','2026-09-19','2.0','2.0',v_old_pins,repeat('1',64),7
  );

  v_result := public.create_orotitan_methodology_successor_run(
    'test:successor:dcf-cas:ok',
    v_identity.issuer_id,
    'IMPOSED_COMPANY',
    'ANALYZE',
    'INITIAL',
    '2026-09-19',
    v_parent_id,
    null,
    '3.0',
    '3.0',
    v_new_pins,
    v_new_hash,
    repeat('a',64),
    7,
    'ACTIVE',
    'DEEP_DIVE',
    repeat('1',64),
    v_identity.security_id,
    v_identity.dossier_id
  );

  select * into v_child
  from public.orotitan_runs
  where run_id=(v_result->>'run_id')::uuid;

  if v_child.parent_run_id <> v_parent_id
     or v_child.run_type <> 'INITIAL'
     or v_child.baseline_snapshot_id is not null
     or v_child.data_cutoff <> date '2026-09-19'
     or v_child.contract_set_sha256 <> v_new_hash then
    raise exception 'successor child persistence mismatch';
  end if;

  -- Exact idempotent replay returns the same child and does not create another.
  if (public.create_orotitan_methodology_successor_run(
    'test:successor:dcf-cas:ok',
    v_identity.issuer_id,'IMPOSED_COMPANY','ANALYZE','INITIAL','2026-09-19',
    v_parent_id,null,'3.0','3.0',v_new_pins,v_new_hash,repeat('a',64),
    7,'ACTIVE','DEEP_DIVE',repeat('1',64),v_identity.security_id,v_identity.dossier_id
  )->>'run_id')::uuid <> v_child.run_id then
    raise exception 'successor idempotent replay returned different child';
  end if;

  update public.orotitan_runs
  set state_version=8
  where run_id=v_parent_id;

  v_failed := false;
  begin
    perform public.create_orotitan_methodology_successor_run(
      'test:successor:dcf-cas:stale',
      v_identity.issuer_id,'IMPOSED_COMPANY','ANALYZE','INITIAL','2026-09-19',
      v_parent_id,null,'3.0','3.0',v_new_pins,v_new_hash,repeat('b',64),
      7,'ACTIVE','DEEP_DIVE',repeat('1',64),v_identity.security_id,v_identity.dossier_id
    );
  exception when serialization_failure then
    if position('SUCCESSOR_PARENT_STATE_VERSION_MISMATCH' in sqlerrm)>0 then
      v_failed := true;
    else
      raise;
    end if;
  end;
  if not v_failed then raise exception 'stale parent CAS did not fail closed'; end if;

  update public.orotitan_runs
  set run_status='BLOCKED', state_version=9
  where run_id=v_parent_id;

  v_failed := false;
  begin
    perform public.create_orotitan_methodology_successor_run(
      'test:successor:dcf-cas:status',
      v_identity.issuer_id,'IMPOSED_COMPANY','ANALYZE','INITIAL','2026-09-19',
      v_parent_id,null,'3.0','3.0',v_new_pins,v_new_hash,repeat('c',64),
      9,'ACTIVE','DEEP_DIVE',repeat('1',64),v_identity.security_id,v_identity.dossier_id
    );
  exception when check_violation then
    if position('SUCCESSOR_PARENT_STATUS_MISMATCH' in sqlerrm)>0 then
      v_failed := true;
    else
      raise;
    end if;
  end;
  if not v_failed then raise exception 'parent status mismatch did not fail closed'; end if;

  v_failed := false;
  begin
    perform public.create_orotitan_methodology_successor_run(
      'test:successor:dcf-cas:run-type',
      v_identity.issuer_id,'IMPOSED_COMPANY','ANALYZE','REFRESH','2026-09-19',
      v_parent_id,null,'3.0','3.0',v_new_pins,v_new_hash,repeat('d',64),
      9,'BLOCKED','DEEP_DIVE',repeat('1',64),v_identity.security_id,v_identity.dossier_id
    );
  exception when check_violation then
    if position('SUCCESSOR_RUN_TYPE_INVALID' in sqlerrm)>0 then
      v_failed := true;
    else
      raise;
    end if;
  end;
  if not v_failed then raise exception 'illegal successor run_type did not fail closed'; end if;

  v_failed := false;
  begin
    perform public.create_orotitan_run(
      'test:successor:generic-bypass',
      v_identity.issuer_id,
      'IMPOSED_COMPANY',
      'ANALYZE',
      'INITIAL',
      '2026-09-19',
      v_parent_id,
      null,
      '3.0',
      '3.0',
      v_new_pins,
      v_new_hash,
      repeat('e',64)
    );
  exception when check_violation then
    if position('PARENT_LINK_REQUIRES_CONTROLLED_SUCCESSOR_RPC' in sqlerrm)>0 then
      v_failed := true;
    else
      raise;
    end if;
  end;
  if not v_failed then raise exception 'generic parent-link successor bypass did not fail closed'; end if;
end;
$;

select 'methodology successor transactional CAS regression: PASS' as result;
