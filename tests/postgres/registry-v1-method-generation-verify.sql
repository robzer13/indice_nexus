\set ON_ERROR_STOP on

do $$
begin
  if (select replace(pg_get_functiondef(p.oid), E'\n    if v_existing.methodology_generation is distinct from ''METHOD_V1'' then\n      raise exception ''IDEMPOTENCY_CONFLICT: legacy route requires METHOD_V1'' using errcode = ''23514'';\n    end if;', '') is distinct from f.definition
      from pg_proc p join public.method_generation_test_functions f using (oid)
      where p.proname='create_orotitan_methodology_successor_run') then
    raise exception 'legacy successor differs from production predecessor beyond replay guard';
  end if;
  if not exists (select 1 from public.method_generation_test_history) then
    raise exception 'grandfathering requires historical runs';
  end if;
  if exists (
    select 1 from public.method_generation_test_history h
    left join public.orotitan_runs r using (run_id)
    where r.run_id is null or r.methodology_generation <> 'METHOD_V1'
      or r.methodology_authority_sha256 is not null
      or r.ctid::text <> h.physical_row
      or (to_jsonb(r) - 'methodology_generation' - 'methodology_authority_sha256') <> h.original_row
  ) then raise exception 'historical runs changed or physically rewritten'; end if;
  if exists (select 1 from public.method_generation_test_functions f
    left join pg_proc p using (oid) where p.oid is null or p.proacl is distinct from f.proacl)
    or exists (select 1 from public.method_generation_test_security s
      left join pg_class c using (oid) where c.oid is null
        or c.relacl is distinct from s.relacl or c.relrowsecurity <> s.relrowsecurity
        or c.relforcerowsecurity <> s.relforcerowsecurity)
    or exists ((select * from pg_policy) except (select * from public.method_generation_test_policies))
    or exists ((select * from public.method_generation_test_policies) except (select * from pg_policy))
  then raise exception 'existing privilege/RLS boundary changed'; end if;
  if exists (select 1 from public.method_generation_test_snapshots h
    left join public.research_snapshots s using (snapshot_id)
    where s.snapshot_id is null or to_jsonb(s) <> h.original_row)
  then raise exception 'snapshot provenance changed'; end if;
end;
$$;

-- Test adapter only: every supplied field reaches the actual database RPC.
create function pg_temp.method_v2_call(q jsonb, successor boolean default true)
returns jsonb language plpgsql as $$
begin
  if successor then
    return public.create_orotitan_method_v2_successor_run(
      q->>'key', (q->>'issuer')::uuid, q->>'entry', q->>'mode', q->>'type',
      (q->>'cutoff')::date, (q->>'parent')::uuid, (q->>'baseline')::uuid,
      q->>'process', q->>'pilotage', q->'pins', q->>'contract', q->>'fingerprint',
      (q->>'state')::bigint, q->>'status', q->>'stage', q->>'expected_contract',
      (q->>'security')::uuid, (q->>'dossier')::uuid, q->>'authority');
  end if;
  return public.create_orotitan_method_v2_run(
    q->>'key', (q->>'issuer')::uuid, q->>'entry', q->>'mode', q->>'type',
    (q->>'cutoff')::date, (q->>'parent')::uuid, (q->>'baseline')::uuid,
    q->>'process', q->>'pilotage', q->'pins', q->>'contract', q->>'fingerprint', q->>'authority');
end;
$$;
create function pg_temp.method_v2_reject(q jsonb, expected text, successor boolean default true)
returns void language plpgsql as $$
begin
  begin
    perform pg_temp.method_v2_call(q, successor);
  exception when others then
    if position(expected in sqlerrm) > 0 then return; end if;
    raise;
  end;
  raise exception 'unexpected admission: %', expected;
end;
$$;

do $$
declare
  p public.orotitan_runs%rowtype;
  c public.orotitan_runs%rowtype;
  parent_before jsonb;
  q jsonb;
  ordinary jsonb;
  result jsonb;
  mutation jsonb;
  alternative_pins jsonb;
  snapshot uuid;
  fn regprocedure;
  role_name text;
  v2_id uuid;
  legacy_id uuid;
  parent_id uuid := gen_random_uuid();
begin
  -- The existing V1.10 regression creates this active-pin V3 child as METHOD_V1.
  select * into p from public.orotitan_runs
  where creation_idempotency_key = 'test:successor:dcf-cas:ok';
  if not found or p.contract_set_sha256 <> '257c287357c19a5d47a42f140a1eb0377d48701b04b07e1e9e740646797c172c'
     or p.methodology_generation <> 'METHOD_V1' then
    raise exception 'active contract-set historical fixture missing';
  end if;
  insert into public.orotitan_runs(run_id,creation_idempotency_key,issuer_id,security_id,dossier_id,
    entry_path,canonical_mode,run_type,run_status,current_stage,data_cutoff,
    process_version,pilotage_contract_version,contract_pins,contract_set_sha256,state_version)
  values(parent_id,'method-generation:parent',p.issuer_id,p.security_id,p.dossier_id,
    p.entry_path,p.canonical_mode,'INITIAL','ACTIVE','DEEP_DIVE',p.data_cutoff,
    p.process_version,p.pilotage_contract_version,p.contract_pins,p.contract_set_sha256,7);
  select * into p from public.orotitan_runs where run_id=parent_id;
  parent_before := to_jsonb(p);
  q := jsonb_build_object('key','method-generation:successor','issuer',p.issuer_id,
    'entry',p.entry_path,'mode',p.canonical_mode,'type','INITIAL','cutoff',p.data_cutoff,
    'parent',p.run_id,'baseline',null,'process',p.process_version,'pilotage',p.pilotage_contract_version,
    'pins',p.contract_pins,'contract',p.contract_set_sha256,'fingerprint',repeat('a',64),
    'state',7,'status','ACTIVE','stage','DEEP_DIVE','expected_contract',p.contract_set_sha256,
    'security',p.security_id,'dossier',p.dossier_id,
    'authority','1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2');

  foreach mutation in array array[
    jsonb_build_object('contract',repeat('0',64)),
    jsonb_build_object('pins',jsonb_set(p.contract_pins,'{process,version}','"different"')),
    jsonb_build_object('process','different'),jsonb_build_object('pilotage','different')
  ] loop
    perform pg_temp.method_v2_reject(q || mutation,'METHOD_V2_SUCCESSOR_EXECUTION_IDENTITY_MISMATCH');
  end loop;
  -- A self-consistent alternative set is still illegal for an identity-only transition.
  alternative_pins := jsonb_set(p.contract_pins,'{process,version}','"different"');
  perform pg_temp.method_v2_reject(q || jsonb_build_object('pins',alternative_pins,
    'contract',public.orotitan_contract_set_sha256(alternative_pins)),
    'METHOD_V2_SUCCESSOR_EXECUTION_IDENTITY_MISMATCH');
  perform pg_temp.method_v2_reject(q || '{"authority":null}','METHOD_V2_AUTHORITY_MISMATCH');
  perform pg_temp.method_v2_reject(q || jsonb_build_object('authority',repeat('0',64)),'METHOD_V2_AUTHORITY_MISMATCH');
  perform pg_temp.method_v2_reject(q || '{"state":6}','SUCCESSOR_PARENT_STATE_VERSION_MISMATCH');
  perform pg_temp.method_v2_reject(q || '{"state":null}','SUCCESSOR_PARENT_STATE_VERSION_MISMATCH');
  perform pg_temp.method_v2_reject(q || '{"type":"REFRESH"}','SUCCESSOR_RUN_TYPE_INVALID');
  perform pg_temp.method_v2_reject(q || jsonb_build_object('baseline',gen_random_uuid()),'SUCCESSOR_BASELINE_MISMATCH');
  perform pg_temp.method_v2_reject(q || '{"cutoff":"2026-01-01"}','SUCCESSOR_PARENT_CUTOFF_MISMATCH');
  perform pg_temp.method_v2_reject(q || jsonb_build_object('issuer',gen_random_uuid()),'SUCCESSOR_PARENT_IDENTITY_MISMATCH');
  perform pg_temp.method_v2_reject(q || jsonb_build_object('security',gen_random_uuid()),'SUCCESSOR_PARENT_IDENTITY_MISMATCH');
  perform pg_temp.method_v2_reject(q || jsonb_build_object('dossier',gen_random_uuid()),'SUCCESSOR_PARENT_IDENTITY_MISMATCH');
  perform pg_temp.method_v2_reject(q || '{"entry":"DISCOVERY_TO_DECISION"}','SUCCESSOR_PARENT_ROUTING_MISMATCH');
  perform pg_temp.method_v2_reject(q || '{"mode":"REFRESH"}','SUCCESSOR_PARENT_ROUTING_MISMATCH');
  perform pg_temp.method_v2_reject(q || '{"status":"BLOCKED"}','SUCCESSOR_PARENT_STATUS_MISMATCH');
  perform pg_temp.method_v2_reject(q || '{"stage":"RESEARCH"}','SUCCESSOR_PARENT_STAGE_MISMATCH');
  update public.orotitan_runs set run_status='BLOCKED' where run_id=p.run_id;
  perform pg_temp.method_v2_reject(q,'SUCCESSOR_PARENT_STATUS_MISMATCH');
  update public.orotitan_runs set run_status='ACTIVE',current_stage='RESEARCH' where run_id=p.run_id;
  perform pg_temp.method_v2_reject(q,'SUCCESSOR_PARENT_STAGE_MISMATCH');
  update public.orotitan_runs set current_stage='DEEP_DIVE',published_at=now() where run_id=p.run_id;
  perform pg_temp.method_v2_reject(q,'SUCCESSOR_PARENT_TERMINAL');
  update public.orotitan_runs set published_at=null where run_id=p.run_id;

  -- Roll this fixture back so canonical provenance and dossier rows stay intact.
  begin
    snapshot := gen_random_uuid();
    insert into public.research_snapshots(snapshot_id,dossier_id,issuer_id,security_id,report_id,
      execution_mode,data_cutoff,calculation_date,report_version,method_version,calculation_version,
      evidence_ledger_version,canonical_payload)
    values(snapshot,p.dossier_id,p.issuer_id,p.security_id,'method-generation:fence','ANALYZE',
      p.data_cutoff,p.data_cutoff,'test','test','test','test',
      jsonb_build_object('snapshot_id',snapshot,'report_id','method-generation:fence',
        'issuer_id',p.issuer_id,'security_id',p.security_id,'execution_mode','ANALYZE',
        'data_lock',jsonb_build_object('data_cutoff',p.data_cutoff,'calculation_date',p.data_cutoff),
        'versions',jsonb_build_object('report_version','test','method_version','test',
          'calculation_version','test','evidence_ledger_version','test')));
    update public.research_dossiers set current_snapshot_id=snapshot where dossier_id=p.dossier_id;
    perform pg_temp.method_v2_reject(q,'SUCCESSOR_BASELINE_MISMATCH');
    raise exception 'rollback successful snapshot-fence fixture' using errcode='Z0001';
  exception when sqlstate 'Z0001' then null;
  end;

  select to_jsonb(r) into parent_before from public.orotitan_runs r where run_id=p.run_id;
  result := pg_temp.method_v2_call(q);
  select * into c from public.orotitan_runs where run_id=(result->>'run_id')::uuid;
  v2_id := c.run_id;
  if c.methodology_generation <> 'METHOD_V2' or c.methodology_authority_sha256 <> q->>'authority'
     or c.parent_run_id <> p.run_id or c.run_type <> 'INITIAL' or c.baseline_snapshot_id is not null
     or (to_jsonb(c) - array['run_id','creation_idempotency_key','parent_run_id','run_status','current_stage',
       'state_version','created_at','updated_at','methodology_generation','methodology_authority_sha256'])
       <> (parent_before - array['run_id','creation_idempotency_key','parent_run_id','run_status','current_stage',
       'state_version','created_at','updated_at','methodology_generation','methodology_authority_sha256'])
  then raise exception 'identity-only successor did not preserve parent fields'; end if;
  if (select to_jsonb(r) from public.orotitan_runs r where run_id=p.run_id) <> parent_before
  then raise exception 'successor mutated parent'; end if;
  result := pg_temp.method_v2_call(q);
  if result->>'idempotent_replay' <> 'true' or (result->>'run_id')::uuid <> c.run_id
  then raise exception 'exact replay failed'; end if;
  perform pg_temp.method_v2_reject(q || '{"key":"method-generation:fork"}','METHOD_V2_SUCCESSOR_ALREADY_EXISTS');
  perform pg_temp.method_v2_reject(q || '{"process":"changed"}','IDEMPOTENCY_CONFLICT');
  perform pg_temp.method_v2_reject(q || '{"state":8}','IDEMPOTENCY_CONFLICT');
  update public.orotitan_runs set run_status='ACTIVE',current_stage='DEEP_DIVE' where run_id=c.run_id;
  perform pg_temp.method_v2_reject(q || jsonb_build_object('key','method-generation:v2-parent','parent',c.run_id,'state',1),
    'METHOD_V2_SUCCESSOR_REQUIRES_METHOD_V1_PARENT');

  -- Neither caller-controlled fingerprints nor legacy RPCs can replay a V2 row.
  begin
    perform public.create_orotitan_run(q->>'key',p.issuer_id,p.entry_path,p.canonical_mode,'INITIAL',p.data_cutoff,
      p.run_id,null,p.process_version,p.pilotage_contract_version,p.contract_pins,p.contract_set_sha256,repeat('a',64));
    raise exception 'legacy route admitted V2';
  exception when check_violation then
    if position('legacy route requires METHOD_V1' in sqlerrm)=0 then raise; end if;
  end;
  begin
    perform public.create_orotitan_methodology_successor_run(q->>'key',p.issuer_id,p.entry_path,p.canonical_mode,'INITIAL',
      p.data_cutoff,p.run_id,null,p.process_version,p.pilotage_contract_version,p.contract_pins,p.contract_set_sha256,
      repeat('a',64),7,'ACTIVE','DEEP_DIVE',p.contract_set_sha256,p.security_id,p.dossier_id);
    raise exception 'legacy successor admitted V2';
  exception when check_violation then
    if position('legacy route requires METHOD_V1' in sqlerrm)=0 then raise; end if;
  end;

  -- The unique index is also a direct-write backstop, independent of RPC locks.
  begin
    insert into public.orotitan_runs(run_id,creation_idempotency_key,issuer_id,parent_run_id,entry_path,canonical_mode,
      run_type,data_cutoff,process_version,pilotage_contract_version,contract_pins,contract_set_sha256,
      methodology_generation,methodology_authority_sha256)
    values(gen_random_uuid(),'method-generation:direct-fork',p.issuer_id,p.run_id,p.entry_path,p.canonical_mode,
      'INITIAL',p.data_cutoff,p.process_version,p.pilotage_contract_version,p.contract_pins,p.contract_set_sha256,
      'METHOD_V2',q->>'authority');
    raise exception 'unique index did not reject fork';
  exception when unique_violation then
    if position('orotitan_runs_one_method_v2_successor_idx' in sqlerrm)=0 then raise; end if;
  end;
  begin
    update public.orotitan_runs set methodology_generation='METHOD_V2',methodology_authority_sha256=q->>'authority'
    where run_id=p.run_id;
    raise exception 'parent identity mutated';
  exception when check_violation then
    if position('immutable OroTitan methodology identity' in sqlerrm)=0 then raise; end if;
  end;
  begin
    update public.orotitan_runs set methodology_authority_sha256=repeat('0',64) where run_id=c.run_id;
    raise exception 'child authority mutated';
  exception when check_violation then
    if position('immutable OroTitan methodology identity' in sqlerrm)=0 then raise; end if;
  end;
  begin
    update public.orotitan_runs set parent_run_id=c.run_id where run_id=p.run_id;
    raise exception 'parent cycle admitted';
  exception when check_violation then
    if position('immutable OroTitan run lock field' in sqlerrm)=0 then raise; end if;
  end;

  ordinary := q || '{"key":"method-generation:ordinary","parent":null}';
  perform pg_temp.method_v2_reject(ordinary || '{"authority":null}','METHOD_V2_AUTHORITY_MISMATCH',false);
  perform pg_temp.method_v2_reject(ordinary || jsonb_build_object('authority',repeat('0',64)),
    'METHOD_V2_AUTHORITY_MISMATCH',false);
  result := pg_temp.method_v2_call(ordinary,false);
  if (select methodology_generation from public.orotitan_runs where run_id=(result->>'run_id')::uuid) <> 'METHOD_V2'
  then raise exception 'dedicated ordinary create did not materialize V2'; end if;
  if pg_temp.method_v2_call(ordinary,false)->>'idempotent_replay' <> 'true'
  then raise exception 'ordinary V2 replay failed'; end if;
  perform pg_temp.method_v2_reject(ordinary || '{"process":"changed"}','IDEMPOTENCY_CONFLICT',false);
  perform pg_temp.method_v2_reject(q || '{"key":"method-generation:ordinary-parent"}',
    'PARENT_LINK_REQUIRES_CONTROLLED_SUCCESSOR_RPC',false);

  -- Legacy V3 labels still produce METHOD_V1, including service_role execution.
  set local role service_role;
  result := public.create_orotitan_run('method-generation:legacy',p.issuer_id,p.entry_path,p.canonical_mode,'INITIAL',
    p.data_cutoff,null,null,p.process_version,p.pilotage_contract_version,p.contract_pins,p.contract_set_sha256,repeat('b',64));
  legacy_id := (result->>'run_id')::uuid;
  result := public.create_orotitan_run('method-generation:legacy',p.issuer_id,p.entry_path,p.canonical_mode,'INITIAL',
    p.data_cutoff,null,null,p.process_version,p.pilotage_contract_version,p.contract_pins,p.contract_set_sha256,repeat('b',64));
  if result->>'idempotent_replay' <> 'true' then raise exception 'legacy replay failed'; end if;
  reset role;
  if (select methodology_generation <> 'METHOD_V1' or methodology_authority_sha256 is not null
      from public.orotitan_runs where run_id=legacy_id) then raise exception 'legacy create inferred generation'; end if;

  for fn in select oid::regprocedure from pg_proc where pronamespace='public'::regnamespace
    and proname in ('create_orotitan_method_v2_run','create_orotitan_method_v2_successor_run')
  loop
    foreach role_name in array array['public','anon','authenticated','service_role'] loop
      if has_function_privilege(role_name,fn,'EXECUTE') then
        raise exception 'unexpected Method-V2 execution privilege: % %',role_name,fn;
      end if;
    end loop;
  end loop;
  -- Actual denied execution, not only a catalog assertion.
  set local role service_role;
  begin
    perform public.create_orotitan_method_v2_run('denied',null,null,null,null,null,null,null,null,null,null,null,null,null);
    raise exception 'service_role Method-V2 create executed';
  exception when insufficient_privilege then null;
  end;
  begin
    perform public.create_orotitan_method_v2_successor_run('denied',null,null,null,null,null,null,null,null,null,null,null,
      null,null,null,null,null,null,null,null);
    raise exception 'service_role Method-V2 successor executed';
  exception when insufficient_privilege then null;
  end;
  reset role;
end;
$$;

drop table public.method_generation_test_history, public.method_generation_test_functions,
  public.method_generation_test_security, public.method_generation_test_policies, public.method_generation_test_snapshots;
select 'Method-V2 Registry identity adversarial matrix: PASS' as result;
