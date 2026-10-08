\set ON_ERROR_STOP on
begin;
create function pg_temp.assert_v14(ok boolean, label text) returns void language plpgsql as $$
begin if ok is distinct from true then raise exception 'V1.14 FAIL: %',label; end if; end $$;
create function pg_temp.reject_v14(sql text, expected text) returns void language plpgsql as $$
begin
  begin execute sql;
  exception when others then
    if position(expected in sqlerrm)>0 or sqlstate=expected then return; end if;
    raise exception 'Expected %, got % %',expected,sqlstate,sqlerrm;
  end;
  raise exception 'Unexpected admission: %',expected;
end $$;
create function pg_temp.mutate_reject_v14(mutation text, attempt text, expected text) returns void language plpgsql as $$
begin
  begin
    execute mutation;
    perform pg_temp.reject_v14(attempt,expected);
    raise exception 'rollback mutation fixture' using errcode='Z0001';
  exception when sqlstate 'Z0001' then null;
  end;
end $$;
create function pg_temp.call_v14(q jsonb) returns jsonb language sql as $$
 select public.create_orotitan_method_v2_runtime_run(q->>'key',(q->>'issuer')::uuid,(q->>'security')::uuid,
   (q->>'dossier')::uuid,(q->>'snapshot')::uuid,(q->>'cutoff')::date,q->>'fingerprint',q->>'binding')
$$;
create function pg_temp.reject_request_v14(q jsonb, expected text) returns void language plpgsql as $$
begin perform pg_temp.reject_v14(format('select pg_temp.call_v14(%L::jsonb)',q::text),expected); end $$;

-- B36/B44: exact persisted V1.13 rows survive installation without a rewrite.
do $$
declare r public.orotitan_runs%rowtype; manifest uuid; result jsonb;
  hash text:=encode(extensions.digest(convert_to('{}','UTF8'),'sha256'),'hex');
begin
  perform pg_temp.assert_v14(not exists (
    select 1 from public.runtime_firewall_test_history h left join public.orotitan_runs historical_run using(run_id)
    where historical_run.run_id is null or historical_run.runtime_binding_sha256 is not null or historical_run.runtime_commit_sha is not null
      or historical_run.ctid::text <> h.physical_row
      or (to_jsonb(historical_run)-'runtime_binding_sha256'-'runtime_commit_sha') <> h.original_row
  ),'B36 B44 every historical run unchanged; no runtime backfill');
  perform pg_temp.assert_v14((select count(*)=2 from public.orotitan_runs
    where creation_idempotency_key in ('v14:grandfather:1','v14:grandfather:2')
      and methodology_generation='METHOD_V2' and runtime_binding_sha256 is null
      and parent_run_id is null and dossier_id is not null),'actual V1.13 fresh creation fixtures');
  for r in select * from public.orotitan_runs
    where creation_idempotency_key in ('v14:grandfather:1','v14:grandfather:2','method-generation:parent') loop
    perform pg_temp.assert_v14(not exists(select 1 from public.orotitan_artifacts where run_id=r.run_id)
      and not exists(select 1 from public.orotitan_method_v2_challenge_proofs where run_id=r.run_id),
      'historical fixture has no Challenge artifacts/proof');
    -- Integration UPDATE is intercepted even without a lifecycle change for bound runs.
    update public.orotitan_run_stages set state_version=state_version+1
      where run_id=r.run_id and stage_code='INTEGRATION';
    perform pg_temp.assert_v14((select state_version=2 from public.orotitan_run_stages
      where run_id=r.run_id and stage_code='INTEGRATION'),'historical V2/V1 Integration mutation allowed');
    insert into public.orotitan_run_stages(run_id,stage_code,stage_contract_name,stage_contract_version,
      stage_contract_sha256,handoff_gate_name,lifecycle_status,started_at)
    values(r.run_id,'DEEP_DIVE',r.contract_pins #>> '{deep_dive_stage,name}',
      r.contract_pins #>> '{deep_dive_stage,version}',r.contract_pins #>> '{deep_dive_stage,content_sha256}',
      'READY_FOR_INTEGRATION','IN_PROGRESS',now());
    manifest:=gen_random_uuid();
    insert into public.orotitan_artifacts(artifact_id,version,run_id,stage_code,artifact_type,logical_name,
      authority_class,authority_state,media_type,size_bytes,content_sha256,storage_backend,storage_uri,
      supabase_bucket,supabase_object_path)
    values(manifest,1,r.run_id,'DEEP_DIVE','DEEP_DIVE_STAGE_MANIFEST','historical manifest',
      'AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE','application/json',2,hash,'SUPABASE_STORAGE',
      'supabase://orotitan-text-artifacts-v1/'||manifest,'orotitan-text-artifacts-v1',manifest::text);
    update public.orotitan_run_stages set lifecycle_status='COMPLETE',active_manifest_kind='FINAL',
      active_manifest_artifact_id=manifest,active_manifest_version=1,completed_at=now(),handoff_gate_state='YES'
      where run_id=r.run_id and stage_code='DEEP_DIVE';
    perform pg_temp.assert_v14((select lifecycle_status='COMPLETE' and handoff_gate_state='YES'
      from public.orotitan_run_stages where run_id=r.run_id and stage_code='DEEP_DIVE'),
      'historical V2/V1 FINAL mutation needs no V1.14 Challenge proof/edges');
    result:=public.reopen_orotitan_stage(r.run_id,'DEEP_DIVE',
      (select state_version from public.orotitan_runs where run_id=r.run_id),
      (select state_version from public.orotitan_run_stages where run_id=r.run_id and stage_code='DEEP_DIVE'),
      'IN_PROGRESS','{"code":"RECOVERY_REGRESSION"}'::jsonb,'v14:historical:reopen:'||r.run_id,repeat('b',64));
    perform pg_temp.assert_v14(result->>'stage_revision'='2' and exists(
      select 1 from public.orotitan_run_stages where run_id=r.run_id and stage_code='INTEGRATION'
        and lifecycle_status='BLOCKED' and contract_status_code='UPSTREAM_STAGE_REOPENED'
        and handoff_gate_state='NOT_EVALUATED' and active_manifest_artifact_id is null
        and active_manifest_version is null and active_manifest_kind is null and completed_at is null),
      'real METHOD_V1 / grandfathered METHOD_V2 reopen retains inherited downstream invalidation');
    insert into public.orotitan_run_stages(run_id,stage_code,stage_contract_name,stage_contract_version,stage_contract_sha256,
      handoff_gate_name,lifecycle_status,started_at)
    values(r.run_id,'RESEARCH',r.contract_pins #>> '{research_stage,name}',r.contract_pins #>> '{research_stage,version}',
      r.contract_pins #>> '{research_stage,content_sha256}','READY_FOR_DEEP_DIVE','IN_PROGRESS',now());
    manifest:=gen_random_uuid();
    insert into public.orotitan_artifacts(artifact_id,version,run_id,stage_code,artifact_type,logical_name,authority_class,authority_state,
      media_type,size_bytes,content_sha256,storage_backend,storage_uri,supabase_bucket,supabase_object_path)
    values(manifest,1,r.run_id,'RESEARCH','RESEARCH_STAGE_MANIFEST','historical Research','AUTHORITATIVE_STAGE_OUTPUT',
      'AUTHORITATIVE','application/json',2,hash,'SUPABASE_STORAGE','supabase://orotitan-text-artifacts-v1/'||manifest,
      'orotitan-text-artifacts-v1',manifest::text);
    update public.orotitan_run_stages set lifecycle_status='COMPLETE',handoff_gate_state='YES',active_manifest_artifact_id=manifest,
      active_manifest_version=1,active_manifest_kind='FINAL',completed_at=now() where run_id=r.run_id and stage_code='RESEARCH';
    perform public.reopen_orotitan_stage(r.run_id,'RESEARCH',
      (select state_version from public.orotitan_runs where run_id=r.run_id),
      (select state_version from public.orotitan_run_stages where run_id=r.run_id and stage_code='RESEARCH'),
      'IN_PROGRESS','{}'::jsonb,'v14:historical:research:reopen:'||r.run_id,repeat('c',64));
    perform pg_temp.assert_v14((select count(*)=2 from public.orotitan_run_stages
      where run_id=r.run_id and stage_code in ('DEEP_DIVE','INTEGRATION') and lifecycle_status='BLOCKED'
      and contract_status_code='UPSTREAM_STAGE_REOPENED' and handoff_gate_state='NOT_EVALUATED'
      and active_manifest_artifact_id is null and active_manifest_version is null and active_manifest_kind is null
      and completed_at is null and blocker_summary='[{"code":"UPSTREAM_STAGE_REOPENED","summary":"Research was reopened"}]'::jsonb),
      'METHOD_V1 / unbound METHOD_V2 real Research reopen remains grandfathered');
  end loop;
  -- The adjacent index must not constrain two legitimate unbound historical fresh runs.
  update public.orotitan_runs set run_status='ACTIVE',state_version=state_version+1
    where creation_idempotency_key in ('v14:grandfather:1','v14:grandfather:2');
  select * into strict r from public.orotitan_runs where creation_idempotency_key='v14:grandfather:1';
  update public.orotitan_method_v2_runtime_control set admission_mode='ACTIVE_FOR_NEW_RUNS',admission_scope='INITIAL_AND_REFRESH',runtime_commit_sha=repeat('a',40);
  result:=public.create_orotitan_method_v2_runtime_run('v14:alongside-grandfather',r.issuer_id,r.security_id,r.dossier_id,
    (select current_snapshot_id from public.research_dossiers where dossier_id=r.dossier_id),
    greatest(r.data_cutoff,(select s.data_cutoff+1 from public.research_snapshots s join public.research_dossiers d on d.current_snapshot_id=s.snapshot_id where d.dossier_id=r.dossier_id)),repeat('f',64),'0832d3c90afab1e4e044d0b84af3992edcb2961298b379890cd7d978d391ca2c');
  perform pg_temp.assert_v14((select runtime_binding_sha256='0832d3c90afab1e4e044d0b84af3992edcb2961298b379890cd7d978d391ca2c'
    from public.orotitan_runs where run_id=(result->>'run_id')::uuid),'runtime admission excludes unbound historical cohort');
  -- A new bound run has no Integration row yet: INSERT still enforces the runtime firewall.
  perform pg_temp.reject_v14(format('insert into public.orotitan_run_stages(run_id,stage_code,stage_contract_name,stage_contract_version,stage_contract_sha256,handoff_gate_name) values(%L,''INTEGRATION'',''test'',''test'',%L,''READY_TO_PUBLISH'')',
    (result->>'run_id')::uuid,hash),'INTEGRATION_NOT_ADMITTED');
  perform pg_temp.assert_v14(not exists(select 1 from public.runtime_firewall_test_history h
    join public.orotitan_runs historical_run using(run_id) where historical_run.runtime_binding_sha256 is not null),'historical binding remains NULL after mutations');
  update public.orotitan_method_v2_runtime_control set admission_mode='INSTALLED_INACTIVE',admission_scope='INITIAL_ONLY',runtime_commit_sha=null;
end $$;
select 'B36 B44 V1.13-created NULL-binding V2 and V1 grandfathering / no backfill / adjacent index: PASS' as result;

do $$
declare
  c public.orotitan_method_v2_runtime_control%rowtype;
  q jsonb; result jsonb; again jsonb; r public.orotitan_runs%rowtype;
  issuer uuid:=gen_random_uuid(); security uuid:=gen_random_uuid(); dossier uuid:=gen_random_uuid();
  other_issuer uuid:=gen_random_uuid(); other_security uuid:=gen_random_uuid(); sibling_security uuid:=gen_random_uuid();
  baseline uuid:=gen_random_uuid(); legacy jsonb; legacy_pins jsonb; fn regprocedure;
  initial_run uuid; role_name text; mutated jsonb;
begin
  select * into c from public.orotitan_method_v2_runtime_control;
  perform pg_temp.assert_v14((select count(*)=1 from public.orotitan_method_v2_runtime_control),'B01 singleton count');
  perform pg_temp.reject_v14('insert into public.orotitan_method_v2_runtime_control(singleton) values(true)','23505');
  perform pg_temp.reject_v14('insert into public.orotitan_method_v2_runtime_control(singleton) values(false)','23514');
  perform pg_temp.reject_v14('delete from public.orotitan_method_v2_runtime_control','append-preserved');
  perform pg_temp.reject_v14('truncate public.orotitan_method_v2_runtime_control','append-preserved');
  perform pg_temp.reject_v14('update public.orotitan_method_v2_runtime_control set admission_mode=''UNKNOWN''','23514');
  perform pg_temp.assert_v14(c.admission_mode='INSTALLED_INACTIVE' and c.admission_scope='INITIAL_ONLY'
    and c.publication_mode='DISABLED' and c.runtime_commit_sha is null,'B02 B03 inactive defaults');
  perform pg_temp.assert_v14((select count(*)=0 from public.orotitan_method_v2_canary_allowlist),'canary empty default');
  set local role service_role;
  perform pg_temp.assert_v14((select count(*)=1 from public.orotitan_method_v2_runtime_control),'B04 service read');
  perform pg_temp.reject_v14('update public.orotitan_method_v2_runtime_control set admission_mode=''ACTIVE_FOR_NEW_RUNS''','42501');
  perform pg_temp.reject_v14('insert into public.orotitan_method_v2_runtime_control(singleton) values(true)','42501');
  perform pg_temp.reject_v14('delete from public.orotitan_method_v2_runtime_control','42501');
  perform pg_temp.reject_v14('truncate public.orotitan_method_v2_runtime_control','42501');
  perform pg_temp.reject_v14('delete from public.orotitan_method_v2_canary_allowlist','42501');
  perform pg_temp.reject_v14('insert into public.orotitan_method_v2_challenge_proofs(run_id) values(null)','42501');
  reset role;
  foreach role_name in array array['anon','authenticated'] loop
    execute format('set local role %I',role_name);
    perform pg_temp.reject_v14('select * from public.orotitan_method_v2_runtime_control','42501');
    perform pg_temp.reject_v14('select * from public.orotitan_method_v2_canary_allowlist','42501');
    perform pg_temp.reject_v14('delete from public.orotitan_method_v2_canary_allowlist','42501');
    reset role;
  end loop;
  perform pg_temp.assert_v14(not has_table_privilege('public','public.orotitan_method_v2_runtime_control','SELECT'),'B06 PUBLIC denied');
  perform pg_temp.assert_v14(c.methodology_authority_sha256='1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2'
    and c.contract_set_sha256='23b75bf5c2d7448e8270e7e8f3a0223e639d0be3a1406b3c089e6e885dd063ea','B07 B08 exact identities');
  perform pg_temp.assert_v14(c.runtime_binding_sha256='0832d3c90afab1e4e044d0b84af3992edcb2961298b379890cd7d978d391ca2c','B09 binding');
  fn := 'public.create_orotitan_method_v2_runtime_run(text,uuid,uuid,uuid,uuid,date,text,text)'::regprocedure;
  perform pg_temp.assert_v14((select prosecdef and proconfig=array['search_path=pg_catalog, public'] from pg_proc where oid=fn),'B11 safe SECURITY DEFINER');
  foreach role_name in array array['public','anon','authenticated','service_role'] loop
    perform pg_temp.assert_v14(not has_function_privilege(role_name,fn,'EXECUTE'),'B12 no runtime EXECUTE');
  end loop;
  set local role service_role;
  perform pg_temp.reject_v14('select public.create_orotitan_method_v2_runtime_run(null,null,null,null,null,null,null,null)','42501');
  reset role;
  perform pg_temp.assert_v14((select proargnames = array['p_creation_idempotency_key','p_issuer_id','p_security_id','p_dossier_id',
    'p_expected_current_snapshot_id','p_data_cutoff','p_request_fingerprint_sha256','p_runtime_binding_sha256'] from pg_proc where oid=fn),'B14 B16 caller identity limited');
  perform pg_temp.reject_v14('update public.orotitan_method_v2_runtime_control set admission_mode=''CANARY_ONLY''','23514');
  perform pg_temp.reject_v14('update public.orotitan_method_v2_runtime_control set admission_mode=''ACTIVE_FOR_NEW_RUNS''','23514');

  insert into public.issuers(issuer_id,display_name) values(issuer,'Runtime synthetic'),(other_issuer,'Other synthetic');
  insert into public.securities(security_id,issuer_id,ticker,exchange,trading_currency)
    values(security,issuer,'RUNTIME','TEST','EUR'),(other_security,other_issuer,'OTHER','TEST','EUR'),(sibling_security,issuer,'SIBLING','TEST','EUR');
  insert into public.research_dossiers(dossier_id,issuer_id,candidate_episode) values(dossier,issuer,gen_random_uuid());
  q := jsonb_build_object('key','v14:initial','issuer',issuer,'security',security,'dossier',dossier,
    'snapshot',null,'cutoff','2026-10-07','fingerprint',repeat('a',64),'binding',c.runtime_binding_sha256);
  perform pg_temp.reject_request_v14(q || jsonb_build_object('binding',repeat('0',64)),'RUNTIME_BINDING_MISMATCH');
  perform pg_temp.reject_request_v14(q,'RUNTIME_NOT_ADMITTED'); -- B30
  update public.orotitan_method_v2_runtime_control set admission_mode='DISABLED';
  perform pg_temp.reject_request_v14(q,'RUNTIME_NOT_ADMITTED'); -- B31
  update public.orotitan_method_v2_runtime_control set admission_mode='INSTALLED_INACTIVE';
  select contract_pins into legacy_pins from public.orotitan_runs where methodology_generation='METHOD_V1' limit 1;
  legacy := public.create_orotitan_run('v14:legacy',issuer,'IMPOSED_COMPANY','ANALYZE','INITIAL','2026-10-07',null,null,
    legacy_pins #>> '{process,version}',legacy_pins #>> '{pilotage,version}',legacy_pins,public.orotitan_contract_set_sha256(legacy_pins),repeat('b',64));
  perform pg_temp.assert_v14(legacy->>'idempotent_replay'='false','B33 inactive legacy compatibility');
  update public.orotitan_method_v2_runtime_control set admission_mode='CANARY_ONLY',runtime_commit_sha=repeat('a',40);
  perform pg_temp.reject_request_v14(q,'CANARY_SCOPE_MISMATCH'); -- empty allowlist
  insert into public.orotitan_method_v2_canary_allowlist values(dossier,issuer,security);
  perform pg_temp.reject_request_v14(q || jsonb_build_object('security',sibling_security),'CANARY_SCOPE_MISMATCH');
  perform pg_temp.reject_request_v14(q || jsonb_build_object('security',other_security),'SECURITY_IDENTITY_MISMATCH');
  perform pg_temp.reject_request_v14(q || jsonb_build_object('issuer',other_issuer),'DOSSIER_IDENTITY_MISMATCH');
  perform pg_temp.reject_request_v14(q || jsonb_build_object('dossier',gen_random_uuid()),'DOSSIER_IDENTITY_MISMATCH');
  update public.research_dossiers set active=false where dossier_id=dossier;
  perform pg_temp.reject_request_v14(q,'DOSSIER_IDENTITY_MISMATCH');
  update public.research_dossiers set active=true where dossier_id=dossier;
  perform pg_temp.reject_request_v14(q || jsonb_build_object('snapshot',gen_random_uuid()),'STALE_CURRENT_SNAPSHOT');
  result := pg_temp.call_v14(q); initial_run := (result->>'run_id')::uuid;
  select * into r from public.orotitan_runs where run_id=initial_run;
  perform pg_temp.assert_v14(r.run_type='INITIAL' and r.canonical_mode='ANALYZE' and r.baseline_snapshot_id is null,'B20 NULL derives INITIAL');
  perform pg_temp.assert_v14(r.issuer_id=issuer and r.security_id=security and r.dossier_id=dossier and r.parent_run_id is null,'B24 fully bound at birth');
  perform pg_temp.assert_v14(r.contract_pins=public.orotitan_method_v2_runtime_contract_pins()
    and r.process_version=r.contract_pins #>> '{process,version}' and r.pilotage_contract_version=r.contract_pins #>> '{pilotage,version}'
    and (select count(*) from jsonb_object_keys(r.contract_pins))=13,'B15 exact full pins');
  again := pg_temp.call_v14(q);
  perform pg_temp.assert_v14(again->>'run_id'=result->>'run_id' and again->>'idempotent_replay'='true','B26 exact replay');
  perform pg_temp.reject_request_v14(q || jsonb_build_object('fingerprint',repeat('c',64)),'IDEMPOTENCY_CONFLICT');
  perform pg_temp.reject_request_v14(q || jsonb_build_object('cutoff','2026-10-08'),'IDEMPOTENCY_CONFLICT');
  perform pg_temp.reject_request_v14(q || '{"key":"v14:second"}','FRESH_RUN_ALREADY_EXISTS');
  perform pg_temp.reject_v14(format('update public.orotitan_runs set runtime_commit_sha=%L where run_id=%L',repeat('b',40),initial_run),'RUNTIME_IDENTITY_IMMUTABLE');
  perform pg_temp.reject_v14(format('insert into public.orotitan_runs(run_id,creation_idempotency_key,issuer_id,security_id,dossier_id,entry_path,canonical_mode,run_type,data_cutoff,process_version,pilotage_contract_version,contract_pins,contract_set_sha256,methodology_generation,methodology_authority_sha256,runtime_binding_sha256,runtime_commit_sha) select gen_random_uuid(),''v14:index'',issuer_id,security_id,dossier_id,entry_path,canonical_mode,run_type,data_cutoff,process_version,pilotage_contract_version,contract_pins,contract_set_sha256,methodology_generation,methodology_authority_sha256,runtime_binding_sha256,runtime_commit_sha from public.orotitan_runs where run_id=%L',initial_run),'23505');
  perform pg_temp.assert_v14((public.create_orotitan_run('v14:legacy',issuer,'IMPOSED_COMPANY','ANALYZE','INITIAL','2026-10-07',null,null,
    legacy_pins #>> '{process,version}',legacy_pins #>> '{pilotage,version}',legacy_pins,public.orotitan_contract_set_sha256(legacy_pins),repeat('b',64))->>'run_id')=legacy->>'run_id','B34 legacy exact replay after activation');
  perform pg_temp.reject_v14(format('select public.create_orotitan_run(''v14:legacy:new'',%L,''IMPOSED_COMPANY'',''ANALYZE'',''INITIAL'',''2026-10-07'',null,null,%L,%L,%L::jsonb,%L,%L)',issuer,
    legacy_pins #>> '{process,version}',legacy_pins #>> '{pilotage,version}',legacy_pins::text,public.orotitan_contract_set_sha256(legacy_pins),repeat('b',64)),'LEGACY_NEW_RUN_FIREWALL');
  update public.orotitan_runs set run_status='CANCELLED',cancelled_at=now() where run_id=initial_run;
  insert into public.research_snapshots(snapshot_id,dossier_id,issuer_id,security_id,report_id,execution_mode,data_cutoff,calculation_date,
    report_version,method_version,calculation_version,evidence_ledger_version,canonical_payload)
  values(baseline,dossier,issuer,security,'v14:baseline','ANALYZE','2026-10-06','2026-10-06','test','test','test','test',
    jsonb_build_object('snapshot_id',baseline,'report_id','v14:baseline','issuer_id',issuer,'security_id',security,'execution_mode','ANALYZE',
      'data_lock',jsonb_build_object('data_cutoff','2026-10-06','calculation_date','2026-10-06'),
      'versions',jsonb_build_object('report_version','test','method_version','test','calculation_version','test','evidence_ledger_version','test')));
  update public.research_dossiers set current_snapshot_id=baseline where dossier_id=dossier;
  q := q || jsonb_build_object('key','v14:refresh','snapshot',baseline);
  perform pg_temp.reject_request_v14(q,'REFRESH_NOT_ADMITTED'); -- B21, allowlisted canary never overrides scope
  update public.orotitan_method_v2_runtime_control set admission_scope='INITIAL_AND_REFRESH';
  perform pg_temp.reject_request_v14(q || '{"cutoff":"2026-10-06"}','REFRESH_BASELINE_MISMATCH');
  perform pg_temp.reject_request_v14(q || '{"cutoff":"2026-10-05"}','REFRESH_BASELINE_MISMATCH');
  insert into public.orotitan_method_v2_canary_allowlist values(dossier,issuer,sibling_security);
  perform pg_temp.reject_request_v14(q || jsonb_build_object('security',sibling_security),'REFRESH_BASELINE_MISMATCH');
  result := pg_temp.call_v14(q);
  select * into r from public.orotitan_runs where run_id=(result->>'run_id')::uuid;
  perform pg_temp.assert_v14(r.run_type='REFRESH' and r.canonical_mode='REFRESH' and r.baseline_snapshot_id=baseline,'B22 B23 exact REFRESH');
  update public.orotitan_runs set run_status='CANCELLED',cancelled_at=now() where run_id=r.run_id;
  update public.research_dossiers set current_snapshot_id=null where dossier_id=dossier;
  update public.orotitan_method_v2_runtime_control set admission_mode='ACTIVE_FOR_NEW_RUNS',admission_scope='INITIAL_ONLY';
  delete from public.orotitan_method_v2_canary_allowlist;
  q := q || jsonb_build_object('key','v14:active','snapshot',null);
  result := pg_temp.call_v14(q);
  perform pg_temp.assert_v14(result->>'idempotent_replay'='false','B32 ACTIVE initial without allowlist');
  update public.orotitan_method_v2_runtime_control set admission_mode='DISABLED';
  perform pg_temp.assert_v14(pg_temp.call_v14(q)->>'run_id'=result->>'run_id','replay remains exact while disabled');
  mutated := jsonb_set(public.orotitan_method_v2_runtime_contract_pins(),'{process,name}','"tampered"');
  perform pg_temp.assert_v14(public.orotitan_contract_set_sha256(mutated)=c.contract_set_sha256,'digest alone does not bind name');
  perform pg_temp.reject_v14(format('insert into public.orotitan_runs(creation_idempotency_key,issuer_id,security_id,dossier_id,entry_path,canonical_mode,run_type,data_cutoff,process_version,pilotage_contract_version,contract_pins,contract_set_sha256,methodology_generation,methodology_authority_sha256,runtime_binding_sha256,runtime_commit_sha) values(''v14:tampered'',%L,%L,%L,''IMPOSED_COMPANY'',''ANALYZE'',''INITIAL'',''2026-10-07'',''1.0'',''1.0'',%L::jsonb,%L,''METHOD_V2'',%L,%L,%L)',issuer,security,dossier,mutated::text,c.contract_set_sha256,c.methodology_authority_sha256,c.runtime_binding_sha256,repeat('a',40)),'23514');
end $$;
select 'B01-B35 runtime control / exact routing / birth / replay / canary / legacy firewall: PASS' as result;

-- Structural finalization guard matrix. The persisted-byte validator suite proves
-- proof production; owner-only rows here exercise actual PostgreSQL consumers.
do $$
declare
  r public.orotitan_runs%rowtype; manifest uuid:=gen_random_uuid(); a uuid;
  q uuid:=gen_random_uuid(); ch uuid:=gen_random_uuid(); cert uuid:=gen_random_uuid(); f uuid:=gen_random_uuid(); v uuid:=gen_random_uuid();
  hash text:=encode(extensions.digest(convert_to('{}','UTF8'),'sha256'),'hex');
  kind text; kinds text[]:=array['DEEP_DIVE_REPORT','EVIDENCE_LEDGER','CONFLICT_LEDGER','CALCULATION_LEDGER',
    'MATERIAL_ASSUMPTION_REGISTER','ANALYTICAL_BLOCK_OUTPUTS','CROSS_BLOCK_RECONCILIATION_RECORD','RED_TEAM_PREMORTEM_RECORD',
    'VALUATION_ARTIFACT','CERTIFICATION_ARTIFACT','OROTITAN_TERMINAL_GATE_ARTIFACT','READINESS_NEXT_ACTION_ARTIFACT',
    'PRE_CERTIFICATION_QUESTION_LEDGER','PRE_CERTIFICATION_CHALLENGE_REPORT','FUNDAMENTALS_LOCK','VALUATION_LOCK'];
  finalize_sql text; target text; revision integer; mutation text; result jsonb; integration_manifest uuid:=gen_random_uuid();
  integration_before public.orotitan_run_stages%rowtype; deep_dive_before public.orotitan_run_stages%rowtype;
  upstream text; invalidation_summary jsonb; research_manifest uuid:=gen_random_uuid(); research_revision integer;
begin
  select * into r from public.orotitan_runs where creation_idempotency_key='v14:active';
  insert into public.orotitan_run_stages(run_id,stage_code,stage_contract_name,stage_contract_version,stage_contract_sha256,lifecycle_status,handoff_gate_name,started_at)
    values(r.run_id,'DEEP_DIVE',r.contract_pins #>> '{deep_dive_stage,name}',r.contract_pins #>> '{deep_dive_stage,version}',
      r.contract_pins #>> '{deep_dive_stage,content_sha256}','IN_PROGRESS','READY_FOR_INTEGRATION',now());
  perform pg_temp.reject_v14(format('insert into public.orotitan_run_stages(run_id,stage_code,stage_contract_name,stage_contract_version,stage_contract_sha256,handoff_gate_name) values(%L,''INTEGRATION'',''test'',''test'',%L,''READY_TO_PUBLISH'')',r.run_id,hash),'INTEGRATION_NOT_ADMITTED');
  insert into public.orotitan_artifacts(artifact_id,version,run_id,stage_code,artifact_type,logical_name,authority_class,authority_state,media_type,size_bytes,content_sha256,storage_backend,storage_uri,supabase_bucket,supabase_object_path)
    values(manifest,1,r.run_id,'DEEP_DIVE','DEEP_DIVE_STAGE_MANIFEST','manifest','AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE','application/json',2,hash,'SUPABASE_STORAGE','supabase://orotitan-text-artifacts-v1/manifest','orotitan-text-artifacts-v1','manifest');
  finalize_sql := format('update public.orotitan_run_stages set lifecycle_status=''COMPLETE'',active_manifest_kind=''FINAL'',active_manifest_artifact_id=%L,active_manifest_version=1,completed_at=now(),handoff_gate_state=''YES'' where run_id=%L and stage_code=''DEEP_DIVE''',manifest,r.run_id);
  perform pg_temp.reject_v14(finalize_sql,'DEEP_DIVE_REQUIRED_OUTPUT');
  foreach kind in array kinds loop
    a := case kind when 'PRE_CERTIFICATION_QUESTION_LEDGER' then q when 'PRE_CERTIFICATION_CHALLENGE_REPORT' then ch
      when 'CERTIFICATION_ARTIFACT' then cert when 'FUNDAMENTALS_LOCK' then f when 'VALUATION_LOCK' then v else gen_random_uuid() end;
    insert into public.orotitan_artifacts(artifact_id,version,run_id,stage_code,artifact_type,logical_name,authority_class,authority_state,media_type,size_bytes,content_sha256,storage_backend,storage_uri,supabase_bucket,supabase_object_path,manifest_artifact_id,manifest_version)
    values(a,1,r.run_id,'DEEP_DIVE',kind,kind,'AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE','application/json',2,hash,'SUPABASE_STORAGE','supabase://orotitan-text-artifacts-v1/'||a,'orotitan-text-artifacts-v1',a::text,manifest,1);
  end loop;
  perform pg_temp.reject_v14(finalize_sql,'CHALLENGE_PERSISTED_PROOF_REQUIRED');
  insert into public.orotitan_method_v2_challenge_proofs values(r.run_id,1,q,1,hash,ch,1,hash,f,1,hash,v,1,hash,r.runtime_binding_sha256,'verifyPersistedMethodV2Challenge:1.0');
  perform pg_temp.reject_v14(finalize_sql,'CHALLENGE_CERTIFICATION_LINEAGE_MISMATCH');
  insert into public.orotitan_artifact_edges(child_run_id,child_artifact_id,child_version,parent_run_id,parent_artifact_id,parent_version,relation_type)
    values(r.run_id,ch,1,r.run_id,q,1,'CONSUMES'),(r.run_id,ch,1,r.run_id,v,1,'CONSUMES');
  perform pg_temp.reject_v14(finalize_sql,'CHALLENGE_CERTIFICATION_LINEAGE_MISMATCH');
  insert into public.orotitan_artifact_edges(child_run_id,child_artifact_id,child_version,parent_run_id,parent_artifact_id,parent_version,relation_type)
    values(r.run_id,cert,1,r.run_id,ch,1,'CONSUMES');
  -- Each Method-V2 addition must fail individually; locks and proof revision cannot go stale.
  perform pg_temp.mutate_reject_v14(format('update public.orotitan_artifacts set artifact_status=''INVALIDATED'' where artifact_id=%L',q),finalize_sql,'DEEP_DIVE_REQUIRED_OUTPUT');
  perform pg_temp.mutate_reject_v14(format('update public.orotitan_artifacts set artifact_status=''INVALIDATED'' where artifact_id=%L',ch),finalize_sql,'DEEP_DIVE_REQUIRED_OUTPUT');
  perform pg_temp.mutate_reject_v14(format('update public.orotitan_artifacts set artifact_status=''INVALIDATED'' where artifact_id=%L',cert),finalize_sql,'DEEP_DIVE_REQUIRED_OUTPUT');
  perform pg_temp.mutate_reject_v14(format('update public.orotitan_artifacts set authority_state=''SUPERSEDED'' where artifact_id=%L',v),finalize_sql,'CHALLENGE_LOCK_LINEAGE_MISMATCH');
  perform pg_temp.mutate_reject_v14(format('update public.orotitan_run_stages set stage_revision=2 where run_id=%L and stage_code=''DEEP_DIVE''',r.run_id),finalize_sql,'CHALLENGE_PERSISTED_PROOF_REQUIRED');
  execute finalize_sql;
  perform public.start_orotitan_stage(r.run_id,'INTEGRATION',
    (select state_version from public.orotitan_runs where run_id=r.run_id),
    r.contract_pins #>> '{integration_stage,name}',r.contract_pins #>> '{integration_stage,version}',
    r.contract_pins #>> '{integration_stage,content_sha256}','v14:integration:start',repeat('a',64));
  perform pg_temp.mutate_reject_v14(format('update public.orotitan_artifacts set artifact_status=''INVALIDATED'' where artifact_id=%L',ch),
    format('update public.orotitan_run_stages set state_version=state_version+1 where run_id=%L and stage_code=''INTEGRATION''',r.run_id),'DEEP_DIVE_REQUIRED_OUTPUT');
  insert into public.orotitan_run_stages(run_id,stage_code,stage_contract_name,stage_contract_version,stage_contract_sha256,
    lifecycle_status,handoff_gate_name,started_at)
  values(r.run_id,'RESEARCH',r.contract_pins #>> '{research_stage,name}',r.contract_pins #>> '{research_stage,version}',
    r.contract_pins #>> '{research_stage,content_sha256}','IN_PROGRESS','READY_FOR_DEEP_DIVE',now());
  insert into public.orotitan_artifacts(artifact_id,version,run_id,stage_code,artifact_type,logical_name,authority_class,authority_state,
    media_type,size_bytes,content_sha256,storage_backend,storage_uri,supabase_bucket,supabase_object_path)
  values(research_manifest,1,r.run_id,'RESEARCH','RESEARCH_STAGE_MANIFEST','Research final','AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE',
    'application/json',2,hash,'SUPABASE_STORAGE','supabase://orotitan-text-artifacts-v1/'||research_manifest,
    'orotitan-text-artifacts-v1',research_manifest::text);
  update public.orotitan_run_stages set lifecycle_status='COMPLETE',handoff_gate_state='YES',active_manifest_artifact_id=research_manifest,
    active_manifest_version=1,active_manifest_kind='FINAL',completed_at=now() where run_id=r.run_id and stage_code='RESEARCH';
  foreach upstream in array array['DEEP_DIVE','RESEARCH'] loop
   invalidation_summary:=jsonb_build_array(jsonb_build_object('code','UPSTREAM_STAGE_REOPENED',
     'summary',case when upstream='RESEARCH' then 'Research was reopened' else 'Deep Dive was reopened' end));
   foreach target in array array['IN_PROGRESS','BLOCKED'] loop
    if target='BLOCKED' then
      integration_manifest:=gen_random_uuid();
      insert into public.orotitan_artifacts(artifact_id,version,run_id,stage_code,artifact_type,logical_name,authority_class,authority_state,
        media_type,size_bytes,content_sha256,storage_backend,storage_uri,supabase_bucket,supabase_object_path)
      values(integration_manifest,1,r.run_id,'INTEGRATION','INTEGRATION_STAGE_MANIFEST','Integration final','AUTHORITATIVE_STAGE_OUTPUT',
        'AUTHORITATIVE','application/json',2,hash,'SUPABASE_STORAGE','supabase://orotitan-text-artifacts-v1/'||integration_manifest,
        'orotitan-text-artifacts-v1',integration_manifest::text);
      update public.orotitan_run_stages set lifecycle_status='COMPLETE',handoff_gate_state='YES',active_manifest_artifact_id=integration_manifest,
        active_manifest_version=1,active_manifest_kind='FINAL',completed_at=now(),state_version=state_version+1
        where run_id=r.run_id and stage_code='INTEGRATION';
    end if;
    select * into strict integration_before from public.orotitan_run_stages where run_id=r.run_id and stage_code='INTEGRATION';
    select * into strict deep_dive_before from public.orotitan_run_stages where run_id=r.run_id and stage_code='DEEP_DIVE';
    result:=public.reopen_orotitan_stage(r.run_id,upstream,
      (select state_version from public.orotitan_runs where run_id=r.run_id),
      (select state_version from public.orotitan_run_stages where run_id=r.run_id and stage_code=upstream),
      target,'{"code":"RECOVERY_REGRESSION"}'::jsonb,'v14:reopen:'||upstream||':'||target,repeat('b',64));
    perform pg_temp.assert_v14((select lifecycle_status=target and stage_revision=(result->>'stage_revision')::integer
      from public.orotitan_run_stages where run_id=r.run_id and stage_code=upstream),'real upstream reopen requested lifecycle');
    select stage_revision into revision from public.orotitan_run_stages where run_id=r.run_id and stage_code='DEEP_DIVE';
    perform pg_temp.assert_v14((select lifecycle_status=target and stage_revision=revision
      and handoff_gate_state='NOT_EVALUATED' and active_manifest_artifact_id is null
      from public.orotitan_run_stages where run_id=r.run_id and stage_code='DEEP_DIVE') or upstream='RESEARCH',
      'real runtime-bound Deep Dive reopen succeeds with requested lifecycle');
    if upstream='RESEARCH' then
      perform pg_temp.assert_v14((select lifecycle_status='BLOCKED' and contract_status_code='UPSTREAM_STAGE_REOPENED'
        and handoff_gate_state='NOT_EVALUATED' and active_manifest_artifact_id is null and active_manifest_version is null
        and active_manifest_kind is null and completed_at is null and blocker_summary=invalidation_summary
        and state_version=deep_dive_before.state_version+1
        and stage_revision=deep_dive_before.stage_revision+case when deep_dive_before.lifecycle_status='COMPLETE' then 1 else 0 end
        from public.orotitan_run_stages where run_id=r.run_id and stage_code='DEEP_DIVE'),'exact Research downstream Deep Dive invalidation');
    end if;
    perform pg_temp.assert_v14((select lifecycle_status='BLOCKED' and contract_status_code='UPSTREAM_STAGE_REOPENED'
      and handoff_gate_state='NOT_EVALUATED' and active_manifest_artifact_id is null and active_manifest_version is null
      and active_manifest_kind is null and completed_at is null
      and blocker_summary=invalidation_summary
      and state_version=integration_before.state_version+1
      and stage_revision=integration_before.stage_revision+case when integration_before.lifecycle_status='COMPLETE' then 1 else 0 end
      from public.orotitan_run_stages where run_id=r.run_id and stage_code='INTEGRATION'),
      'Integration becomes exact inert invalidation while Deep Dive is reopened');
    perform pg_temp.reject_v14(format('select public.resume_orotitan_stage(%L,''INTEGRATION'',%s,%s,''v14:resume:reject:%s'',%L)',
      r.run_id,(select state_version from public.orotitan_runs where run_id=r.run_id),
      (select state_version from public.orotitan_run_stages where run_id=r.run_id and stage_code='INTEGRATION'),upstream||':'||target,repeat('c',64)),
      'METHOD_V2_INTEGRATION_NOT_ADMITTED');
    -- Even a fresh INSERT copying the safe shape cannot use the UPDATE exception.
    perform pg_temp.reject_v14(format('insert into public.orotitan_run_stages(run_id,stage_code,stage_contract_name,stage_contract_version,stage_contract_sha256,handoff_gate_name,lifecycle_status,contract_status_code,blocker_summary,started_at) values(%L,''INTEGRATION'',''test'',''test'',%L,''READY_TO_PUBLISH'',''BLOCKED'',''UPSTREAM_STAGE_REOPENED'',%L::jsonb,now())',
      r.run_id,hash,invalidation_summary::text),
      'METHOD_V2_INTEGRATION_NOT_ADMITTED');
    foreach mutation in array array[
      'lifecycle_status=''IN_PROGRESS''', 'handoff_gate_state=''YES''', 'contract_status_code=''READY_TO_PUBLISH''',
      format('active_manifest_artifact_id=%L',manifest), 'active_manifest_version=1', 'active_manifest_kind=''FINAL''',
      'completed_at=now()', 'blocker_summary=''[]''::jsonb', 'stage_revision=stage_revision+1',
      'blocker_summary=''[{"code":"UPSTREAM_STAGE_REOPENED","summary":"arbitrary reopen"}]''::jsonb',
      'blocker_summary=''[{"code":"UPSTREAM_STAGE_REOPENED"}]''::jsonb',
      'blocker_summary=''[{"code":"UPSTREAM_STAGE_REOPENED","summary":"Research was reopened","extra":true}]''::jsonb',
      'state_version=state_version',
      format('lifecycle_status=''COMPLETE'',handoff_gate_state=''YES'',active_manifest_artifact_id=%L,active_manifest_version=1,active_manifest_kind=''FINAL'',completed_at=now()',manifest)
    ] loop
      perform pg_temp.reject_v14(format('update public.orotitan_run_stages set %s where run_id=%L and stage_code=''INTEGRATION''',
        case when mutation='state_version=state_version' then mutation else 'state_version=state_version+1,'||mutation end,r.run_id),
        'METHOD_V2_INTEGRATION_NOT_ADMITTED');
    end loop;
    if upstream='RESEARCH' then
      select stage_revision into research_revision from public.orotitan_run_stages where run_id=r.run_id and stage_code='RESEARCH';
      insert into public.orotitan_artifacts(artifact_id,version,run_id,stage_code,artifact_type,logical_name,authority_class,authority_state,
        media_type,size_bytes,content_sha256,storage_backend,storage_uri,supabase_bucket,supabase_object_path)
      select artifact_id,research_revision,run_id,stage_code,artifact_type,logical_name,authority_class,'AUTHORITATIVE',
        media_type,size_bytes,content_sha256,storage_backend,storage_uri,supabase_bucket,supabase_object_path
      from public.orotitan_artifacts where artifact_id=research_manifest and version=1;
      update public.orotitan_run_stages set lifecycle_status='COMPLETE',handoff_gate_state='YES',active_manifest_artifact_id=research_manifest,
        active_manifest_version=research_revision,active_manifest_kind='FINAL',completed_at=now(),contract_status_code=null,
        blocker_summary='[]'::jsonb,state_version=state_version+1 where run_id=r.run_id and stage_code='RESEARCH';
      perform pg_temp.reject_v14(format('update public.orotitan_run_stages set lifecycle_status=''IN_PROGRESS'',state_version=state_version+1 where run_id=%L and stage_code=''INTEGRATION''',r.run_id),
        'METHOD_V2_INTEGRATION_NOT_ADMITTED');
      perform public.resume_orotitan_stage(r.run_id,'DEEP_DIVE',
        (select state_version from public.orotitan_runs where run_id=r.run_id),
        (select state_version from public.orotitan_run_stages where run_id=r.run_id and stage_code='DEEP_DIVE'),
        'v14:dd:resume:after:research:'||target,repeat('e',64));
    end if;
    -- New immutable versions and proof bind the reopened revision; old bundles stay superseded.
    insert into public.orotitan_artifacts(artifact_id,version,run_id,stage_code,artifact_type,logical_name,authority_class,authority_state,
      media_type,size_bytes,content_sha256,storage_backend,storage_uri,supabase_bucket,supabase_object_path,manifest_artifact_id,manifest_version)
    select artifact_id,revision,run_id,stage_code,artifact_type,logical_name,authority_class,'AUTHORITATIVE',media_type,size_bytes,
      content_sha256,storage_backend,storage_uri,supabase_bucket,supabase_object_path,manifest_artifact_id,
      case when manifest_artifact_id is not null then revision end
    from public.orotitan_artifacts where run_id=r.run_id and stage_code='DEEP_DIVE' and version=1;
    insert into public.orotitan_method_v2_challenge_proofs values(r.run_id,revision,q,revision,hash,ch,revision,hash,
      f,revision,hash,v,revision,hash,r.runtime_binding_sha256,'verifyPersistedMethodV2Challenge:1.0');
    perform pg_temp.reject_v14(format('update public.orotitan_run_stages set lifecycle_status=''COMPLETE'',active_manifest_kind=''FINAL'',active_manifest_artifact_id=%L,active_manifest_version=%s,completed_at=now(),handoff_gate_state=''YES'' where run_id=%L and stage_code=''DEEP_DIVE''',
      manifest,revision,r.run_id),'CHALLENGE_CERTIFICATION_LINEAGE_MISMATCH');
    insert into public.orotitan_artifact_edges(child_run_id,child_artifact_id,child_version,parent_run_id,parent_artifact_id,parent_version,relation_type)
      values(r.run_id,ch,revision,r.run_id,q,revision,'CONSUMES'),(r.run_id,ch,revision,r.run_id,v,revision,'CONSUMES'),
        (r.run_id,cert,revision,r.run_id,ch,revision,'CONSUMES');
    update public.orotitan_run_stages set lifecycle_status='COMPLETE',active_manifest_kind='FINAL',active_manifest_artifact_id=manifest,
      active_manifest_version=revision,completed_at=now(),handoff_gate_state='YES',contract_status_code=null,blocker_summary='[]'::jsonb,state_version=state_version+1
      where run_id=r.run_id and stage_code='DEEP_DIVE';
    result:=public.resume_orotitan_stage(r.run_id,'INTEGRATION',
      (select state_version from public.orotitan_runs where run_id=r.run_id),
      (select state_version from public.orotitan_run_stages where run_id=r.run_id and stage_code='INTEGRATION'),
      'v14:resume:allowed:'||upstream||':'||target,repeat('d',64));
    perform pg_temp.assert_v14((select lifecycle_status='IN_PROGRESS' from public.orotitan_run_stages
      where run_id=r.run_id and stage_code='INTEGRATION'),'Integration resumes only after current revision is validly re-finalized');
  end loop;
  end loop;
end $$;
select 'B42 B43 Method-V2 FINAL DEEP_DIVE current output/proof/CONSUMES lineage and Integration bypass rejection: PASS' as result;
select 'Real Deep Dive reopen / exact downstream invalidation / spoof and INSERT rejection / current-lineage readmission: PASS' as result;
select 'Real Research reopen IN_PROGRESS/BLOCKED / exact downstream invalidation / upstream re-finalization/readmission: PASS' as result;
rollback;
select 'B36 B44 historical V1.6-V1.13 regression already passed; all synthetic identity/control changes rolled back: PASS' as result;
