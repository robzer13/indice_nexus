-- J2 transaction qualification against the frozen V3.1 pin pack.
-- psql variable j2_pins is supplied by the Registry runner from the checked-in JSON.
\set ON_ERROR_STOP on
\o /dev/null
select set_config('j2.contract_pins', :'j2_pins', false);
\o
do $$
declare
  pins jsonb := current_setting('j2.contract_pins')::jsonb;
  issuer uuid; security uuid; dossier uuid; run uuid;
  created jsonb; started jsonb; replay jsonb; bad jsonb;
  fingerprint text := repeat('a',64);
  idempotency text := 'orotitan:j2:research:start:v1:synthetic-443';
  research jsonb;
  n integer;
begin
  if (select count(*) from jsonb_object_keys(pins)) <> 15 then raise exception 'J2_EXPECTED_15_PINS'; end if;
  if public.orotitan_contract_set_sha256(pins) <> '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2' then raise exception 'J2_CONTRACT_SET_SHA_MISMATCH'; end if;
  research := pins->'research_stage';
  if research->>'version' <> '2.0' or research->>'content_sha256' <> 'cdd62f087ff8f8dc97f0c16586d634001f2cef1c44f9e4540e087778cdaa7a01' then raise exception 'J2_RESEARCH_PIN_MISMATCH'; end if;
  select issuer_id, security_id, dossier_id into issuer, security, dossier from public.legacy_company_identity_map order by legacy_company_id limit 1;
  if issuer is null or security is null or dossier is null then raise exception 'J2_SYNTHETIC_IDENTITY_FIXTURE_MISSING'; end if;
  created := public.create_orotitan_run('j2:443:create',issuer,'IMPOSED_COMPANY','ANALYZE','INITIAL',date '2026-11-07',null,null,'3.1','3.1',pins,'3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2',repeat('b',64));
  run := (created->>'run_id')::uuid;
  perform public.bind_orotitan_run_identity(run,1,security,dossier,null,null,'j2:443:bind',repeat('c',64));
  if not exists(select 1 from public.orotitan_runs where run_id=run and run_status='CREATED' and state_version=2) then raise exception 'J2_INITIAL_RUN_READBACK_FAILED'; end if;
  started := public.start_orotitan_stage(run,'RESEARCH',2,research->>'name',research->>'version',research->>'content_sha256',idempotency,fingerprint,'PILOTAGE');
  if started->>'idempotent_replay' <> 'false' then raise exception 'J2_FIRST_CALL_NOT_NEW'; end if;
  replay := public.start_orotitan_stage(run,'RESEARCH',2,research->>'name',research->>'version',research->>'content_sha256',idempotency,fingerprint,'PILOTAGE');
  if replay->>'idempotent_replay' <> 'true' then raise exception 'J2_REPLAY_NOT_IDEMPOTENT'; end if;
  select count(*) into n from public.orotitan_run_stages where run_id=run and stage_code='RESEARCH' and lifecycle_status='IN_PROGRESS' and stage_revision=1 and state_version=1;
  if n <> 1 then raise exception 'J2_STAGE_READBACK_INVALID'; end if;
  select count(*) into n from public.orotitan_run_events where run_id=run and event_type='STAGE_STARTED' and idempotency_key=idempotency and request_fingerprint_sha256=fingerprint;
  if n <> 1 then raise exception 'J2_EVENT_READBACK_INVALID'; end if;
  if not exists(select 1 from public.orotitan_runs where run_id=run and run_status='ACTIVE' and current_stage='RESEARCH' and state_version=3) then raise exception 'J2_FINAL_RUN_READBACK_INVALID'; end if;
  begin
    perform public.start_orotitan_stage(run,'RESEARCH',2,research->>'name',research->>'version',research->>'content_sha256','j2:443:stale-cas',repeat('d',64),'PILOTAGE');
    raise exception 'J2_STALE_CAS_ACCEPTED';
  exception when serialization_failure then null; end;
  begin
    perform public.start_orotitan_stage(run,'RESEARCH',3,research->>'name',research->>'version',repeat('0',64),'j2:443:bad-pin',repeat('e',64),'PILOTAGE');
    raise exception 'J2_BAD_PIN_ACCEPTED';
  exception when check_violation then null; end;
  raise notice 'J2_LOCAL_POSTGRES_PASS run_id=% start=% replay=% stage_count=% stage_events=%', run, started, replay, 1, 1;
end $$;
select r.run_id,r.run_status,r.current_stage,r.state_version,s.stage_code,s.lifecycle_status,s.stage_revision,s.state_version as stage_state_version,e.event_type,e.idempotency_key,e.request_fingerprint_sha256
from public.orotitan_runs r join public.orotitan_run_stages s using(run_id) join public.orotitan_run_events e using(run_id)
where r.creation_idempotency_key='j2:443:create' and s.stage_code='RESEARCH' and e.event_type='STAGE_STARTED';
