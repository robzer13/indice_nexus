-- Issue #427. RUN ON AN ISOLATED THROWAWAY TEST DATABASE ONLY.
-- Prerequisites: I1, I3 and Registry migrations, then controlled cancellation migration.
\set ON_ERROR_STOP on
begin;
do $guard$
begin
  if current_database() !~ '^orotitan_registry_v1_[0-9]+$' then
    raise exception 'REFUSE_NON_ISOLATED_DATABASE: %',current_database();
  end if;
end;
$guard$;

do $acl$
declare
  v_sig regprocedure :=
    'public.cancel_orotitan_redundant_run_v1(uuid,bigint,uuid,text,text,text)'::regprocedure;
begin
  if not (select prosecdef from pg_proc where oid=v_sig)
     or not (select proconfig @> array['search_path=pg_catalog, public']::text[]
             from pg_proc where oid=v_sig)
     or has_function_privilege('public',v_sig,'EXECUTE')
     or has_function_privilege('anon',v_sig,'EXECUTE')
     or has_function_privilege('authenticated',v_sig,'EXECUTE')
     or not has_function_privilege('service_role',v_sig,'EXECUTE')
  then raise exception 'INVALID_FUNCTION_ACL'; end if;
end;
$acl$;

insert into public.issuers (issuer_id,display_name) values
 ('d37ccfaa-4dad-47c3-9b44-a752522cd7d5','Test Qualys'),
 ('b7dd4445-ae1e-4f33-985b-216d83e3fbde','Test Veolia');
insert into public.securities (security_id,issuer_id,ticker,exchange,trading_currency) values
 ('5ca6f4a2-a796-55fd-58f8-0798a4b04fbe','d37ccfaa-4dad-47c3-9b44-a752522cd7d5','TQLYS','TEST','USD'),
 ('71bc5f4a-662e-4628-872f-c86d292a81be','b7dd4445-ae1e-4f33-985b-216d83e3fbde','TVIE','TEST','EUR');
insert into public.research_dossiers (dossier_id,issuer_id,candidate_episode,current_snapshot_id) values
 ('17f57723-a218-8327-841a-04c5f5513174','d37ccfaa-4dad-47c3-9b44-a752522cd7d5',gen_random_uuid(),null),
 ('59a08ccd-ceb4-42c2-ab04-aabec5cdc330','b7dd4445-ae1e-4f33-985b-216d83e3fbde',gen_random_uuid(),null);

-- Synthetic V1-schema snapshots satisfy I3 relational locks. The real V2
-- publication is checked by production admission before the RPC is called.
insert into public.research_snapshots
 (snapshot_id,dossier_id,issuer_id,security_id,report_id,execution_mode,
  data_cutoff,calculation_date,report_version,method_version,
  calculation_version,evidence_ledger_version,contract_version,schema_version,canonical_payload)
select sid,'17f57723-a218-8327-841a-04c5f5513174'::uuid,
       'd37ccfaa-4dad-47c3-9b44-a752522cd7d5'::uuid,
       '5ca6f4a2-a796-55fd-58f8-0798a4b04fbe'::uuid,
       'test-report-'||i,'REFRESH',date '2026-09-15',date '2026-09-15',
       'v1','v1','v1','v1','04_SCREENER_SCHEMA_V1','1.0.0',
       jsonb_build_object(
         'snapshot_id',sid,'report_id','test-report-'||i,
         'issuer_id','d37ccfaa-4dad-47c3-9b44-a752522cd7d5',
         'security_id','5ca6f4a2-a796-55fd-58f8-0798a4b04fbe',
         'execution_mode','REFRESH',
         'data_lock',jsonb_build_object('data_cutoff','2026-09-15','calculation_date','2026-09-15'),
         'versions',jsonb_build_object('report_version','v1','method_version','v1',
           'calculation_version','v1','evidence_ledger_version','v1'))
from (values ('960fa825-adad-4bea-bd43-c45847c4c78c'::uuid,1),
             ('e42a1396-357a-42f5-ae0c-5e750f321180'::uuid,2)) x(sid,i);

update public.research_dossiers
  set current_snapshot_id='e42a1396-357a-42f5-ae0c-5e750f321180'
  where dossier_id='17f57723-a218-8327-841a-04c5f5513174';

-- V1.5 requires all 13 contract-pin records even in isolated fixtures.
-- Synthetic immutable locators are test-only, never analytical authority.
create function pg_temp.test_registry_contract_pins()
returns jsonb language sql stable as $pins$
  select jsonb_object_agg(k,jsonb_build_object(
    'name','TEST_'||k,
    'version','1.0',
    'content_sha256',repeat('f',64),
    'locator',jsonb_build_object(
      'backend','HASH_ADDRESSED_PRIVATE_STORE',
      'storage_uri','test://frozen-pins/'||k)))
  from unnest(array[
    'process','pilotage','research_stage','deep_dive_stage','integration_stage',
    'analysis_standard','master_prompt','investment_policy','execution_patch',
    'integration_spec','screener_schema','i2','i3b'
  ]) as t(k);
$pins$;

insert into public.orotitan_runs
 (run_id,creation_idempotency_key,issuer_id,security_id,dossier_id,parent_run_id,baseline_snapshot_id,
  entry_path,canonical_mode,run_type,run_status,current_stage,data_cutoff,
  process_version,pilotage_contract_version,contract_pins,contract_set_sha256,
  state_version,published_at)
values
 ('d5cd803b-f959-4e41-ba55-5fbfc2a7d245','test:cancel:qualys:v1',
  'd37ccfaa-4dad-47c3-9b44-a752522cd7d5','5ca6f4a2-a796-55fd-58f8-0798a4b04fbe','17f57723-a218-8327-841a-04c5f5513174',
  null,'960fa825-adad-4bea-bd43-c45847c4c78c','IMPOSED_COMPANY','REFRESH','REFRESH','ACTIVE','RESEARCH',
  '2026-09-15','1.0','1.0.1',pg_temp.test_registry_contract_pins(),repeat('1',64),3,null),
 ('e41a8fe0-29cb-45e3-8bca-b1f19ac67d12','test:cancel:qualys:v2',
  'd37ccfaa-4dad-47c3-9b44-a752522cd7d5','5ca6f4a2-a796-55fd-58f8-0798a4b04fbe','17f57723-a218-8327-841a-04c5f5513174',
  null,'960fa825-adad-4bea-bd43-c45847c4c78c','IMPOSED_COMPANY','ANALYZE','REFRESH','PUBLISHED','INTEGRATION',
  '2026-09-15','2.0','2.0',pg_temp.test_registry_contract_pins(),repeat('2',64),10,now()),
 ('76ee7eb1-8658-4038-83bf-e2c951db731a','test:cancel:veolia:parent',
  'b7dd4445-ae1e-4f33-985b-216d83e3fbde','71bc5f4a-662e-4628-872f-c86d292a81be','59a08ccd-ceb4-42c2-ab04-aabec5cdc330',
  null,null,'IMPOSED_COMPANY','ANALYZE','INITIAL','BLOCKED','DEEP_DIVE',
  '2026-09-22','3.0','3.0',pg_temp.test_registry_contract_pins(),repeat('3',64),8,null),
 ('d56a0bc9-4e80-48b3-b326-74d1fea15e63','test:cancel:veolia:empty',
  'b7dd4445-ae1e-4f33-985b-216d83e3fbde','71bc5f4a-662e-4628-872f-c86d292a81be','59a08ccd-ceb4-42c2-ab04-aabec5cdc330',
  '76ee7eb1-8658-4038-83bf-e2c951db731a',null,
  'IMPOSED_COMPANY','ANALYZE','INITIAL','ACTIVE','RESEARCH',
  '2026-09-22','3.1','3.1',pg_temp.test_registry_contract_pins(),
  '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2',2,null),
 ('a4cf9002-b52d-4440-8902-adc70bd777dd','test:cancel:veolia:sibling',
  'b7dd4445-ae1e-4f33-985b-216d83e3fbde','71bc5f4a-662e-4628-872f-c86d292a81be','59a08ccd-ceb4-42c2-ab04-aabec5cdc330',
  '76ee7eb1-8658-4038-83bf-e2c951db731a',null,
  'IMPOSED_COMPANY','ANALYZE','INITIAL','BLOCKED','DEEP_DIVE',
  '2026-09-22','3.1','3.1',pg_temp.test_registry_contract_pins(),
  '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2',7,null);

insert into public.orotitan_run_stages
 (run_id,stage_code,stage_contract_name,stage_contract_version,stage_contract_sha256,
  lifecycle_status,handoff_gate_name,handoff_gate_state,started_at,completed_at)
values
 ('d5cd803b-f959-4e41-ba55-5fbfc2a7d245','RESEARCH','TEST','1',repeat('a',64),'COMPLETE','READY_FOR_DEEP_DIVE','YES',now(),now()),
 ('d56a0bc9-4e80-48b3-b326-74d1fea15e63','RESEARCH','TEST','1',repeat('a',64),'IN_PROGRESS','READY_FOR_DEEP_DIVE','NOT_EVALUATED',now(),null),
 ('a4cf9002-b52d-4440-8902-adc70bd777dd','DEEP_DIVE','TEST','1',repeat('a',64),'BLOCKED','READY_FOR_INTEGRATION','NOT_EVALUATED',now(),null);

insert into public.orotitan_artifacts
 (version,run_id,stage_code,artifact_type,logical_name,authority_class,authority_state,media_type,
  size_bytes,content_sha256,storage_backend,storage_uri,supabase_bucket,supabase_object_path)
select 1,'d5cd803b-f959-4e41-ba55-5fbfc2a7d245','RESEARCH','TEST_ARTIFACT',
       'test-qualys-'||i,'CHECKPOINT_STAGE_OUTPUT','CHECKPOINT','application/json',
       10,repeat('c',64),'SUPABASE_STORAGE','test://qualys/'||i,'test','qualys/'||i
from generate_series(1,8) i;
insert into public.orotitan_artifacts
 (version,run_id,stage_code,artifact_type,logical_name,authority_class,authority_state,media_type,
  size_bytes,content_sha256,storage_backend,storage_uri,supabase_bucket,supabase_object_path)
values
 (1,'a4cf9002-b52d-4440-8902-adc70bd777dd','DEEP_DIVE','TEST_ARTIFACT','test-veolia-sibling',
  'CHECKPOINT_STAGE_OUTPUT','CHECKPOINT','application/json',
  10,repeat('c',64),'SUPABASE_STORAGE','test://veolia/sibling','test','veolia/sibling');

-- Mutations below execute ONLY inside this local transaction, rolled back at end.
do $tests$
declare
  q uuid := 'd5cd803b-f959-4e41-ba55-5fbfc2a7d245';
  qr uuid := 'e41a8fe0-29cb-45e3-8bca-b1f19ac67d12';
  v uuid := 'd56a0bc9-4e80-48b3-b326-74d1fea15e63';
  vr uuid := 'a4cf9002-b52d-4440-8902-adc70bd777dd';
  output jsonb;
  rejected boolean;
begin
  rejected := false;
  begin
    perform public.cancel_orotitan_redundant_run_v1(
      '00000000-0000-4000-8000-000000000111',1,qr,
      'REDUNDANT_HISTORICAL_REFRESH_ALREADY_PUBLISHED_V2','test:unknown',repeat('e',64));
  exception when check_violation then rejected := true;
  end;
  if not rejected then raise exception 'UNKNOWN_RUN_WAS_ADMITTED'; end if;

  rejected := false;
  begin
    perform public.cancel_orotitan_redundant_run_v1(
      q,3,vr,'REDUNDANT_HISTORICAL_REFRESH_ALREADY_PUBLISHED_V2','test:wrongpair',repeat('e',64));
  exception when check_violation then rejected := true;
  end;
  if not rejected then raise exception 'WRONG_REPLACEMENT_WAS_ADMITTED'; end if;

  output := public.cancel_orotitan_redundant_run_v1(
    q,3,qr,'REDUNDANT_HISTORICAL_REFRESH_ALREADY_PUBLISHED_V2',
    'test:qualys:cancel',repeat('e',64));
  if output->>'run_status' <> 'CANCELLED' or output->>'idempotent_replay' <> 'false'
  then raise exception 'QUALYS_CANCELLATION_FAILED'; end if;

  output := public.cancel_orotitan_redundant_run_v1(
    q,3,qr,'REDUNDANT_HISTORICAL_REFRESH_ALREADY_PUBLISHED_V2',
    'test:qualys:cancel',repeat('e',64));
  if output->>'idempotent_replay' <> 'true' then raise exception 'QUALYS_IDEMPOTENCY_FAILED'; end if;

  rejected := false;
  begin
    perform public.cancel_orotitan_redundant_run_v1(
      q,3,qr,'REDUNDANT_HISTORICAL_REFRESH_ALREADY_PUBLISHED_V2',
      'test:qualys:cancel',repeat('f',64));
  exception when check_violation then rejected := true;
  end;
  if not rejected then raise exception 'REPLAY_FINGERPRINT_CONFLICT_NOT_REJECTED'; end if;

  output := public.cancel_orotitan_redundant_run_v1(
    v,2,vr,'EMPTY_PARALLEL_METHOD_V3_1_SUCCESSOR_REDUNDANT',
    'test:veolia:cancel',repeat('d',64));
  if output->>'run_status' <> 'CANCELLED' then raise exception 'VEOLIA_CANCELLATION_FAILED'; end if;

  if (select count(*) from public.orotitan_runs
      where run_id in (q,v) and run_status='CANCELLED' and cancelled_at is not null) <> 2
     or (select count(*) from public.orotitan_run_events
         where run_id in (q,v) and event_type='RUN_CANCELLED') <> 2
     or (select state_version from public.orotitan_runs where run_id=q) <> 4
     or (select state_version from public.orotitan_runs where run_id=v) <> 3
     or (select run_status from public.orotitan_runs where run_id=qr) <> 'PUBLISHED'
     or (select run_status from public.orotitan_runs where run_id=vr) <> 'BLOCKED'
     or (select count(*) from public.orotitan_artifacts where run_id=q) <> 8
     or (select current_snapshot_id from public.research_dossiers
         where dossier_id='17f57723-a218-8327-841a-04c5f5513174') is distinct from
         'e42a1396-357a-42f5-ae0c-5e750f321180'::uuid
  then raise exception 'CANCELLATION_POSTCONDITIONS_NOT_MET'; end if;
end;
$tests$;
rollback;
