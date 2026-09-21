\set ON_ERROR_STOP on

-- OroTitan Registry V1.11 persistence-attestation authority regression.
-- Verifies exact Registry reconciliation, stable logical idempotency identity,
-- reachable fingerprint conflict, and zero authority promotion by attestation.

create or replace function pg_temp.pa_pins()
returns jsonb
language plpgsql
as $$
declare
  v_pins jsonb := '{}'::jsonb;
  v_key text;
  v_name text;
  v_version text;
  v_hash text;
begin
  foreach v_key in array array[
    'process','pilotage','research_stage','deep_dive_stage','integration_stage',
    'analysis_standard','master_prompt','investment_policy','execution_patch',
    'integration_spec','screener_schema','i2','i3b'
  ] loop
    v_name := case v_key
      when 'research_stage' then 'RESEARCH'
      when 'deep_dive_stage' then 'DEEP_DIVE'
      when 'integration_stage' then 'INTEGRATION'
      else upper(v_key)
    end;
    v_version := case v_key
      when 'pilotage' then '1.0.1'
      when 'investment_policy' then '1.0.0'
      when 'execution_patch' then '1.0.1'
      when 'screener_schema' then '1.0.0'
      else '1.0'
    end;
    v_hash := encode(
      extensions.digest(convert_to('pa:' || v_key,'UTF8'),'sha256'),
      'hex'
    );
    v_pins := v_pins || jsonb_build_object(
      v_key,
      jsonb_build_object(
        'name',v_name,
        'version',v_version,
        'content_sha256',v_hash,
        'locator',jsonb_build_object(
          'backend','GITHUB_IMMUTABLE',
          'repository','robzer13/indice_nexus',
          'path','contracts/orotitan-equity/v1/execution/' || v_key || '.md',
          'commit_sha',repeat('a',40),
          'blob_sha',repeat('b',40)
        )
      )
    );
  end loop;
  return v_pins;
end;
$$;

create or replace function pg_temp.pa_fixture(p_tag text)
returns jsonb
language plpgsql
as $$
declare
  v_run uuid := gen_random_uuid();
  v_artifact uuid := gen_random_uuid();
  v_issuer uuid;
  v_security uuid;
  v_dossier uuid;
  v_pins jsonb := pg_temp.pa_pins();
  v_contract_hash text := public.orotitan_contract_set_sha256(v_pins);
  v_size bigint := 321;
  v_sha text := encode(
    extensions.digest(convert_to('pa-content:'||p_tag,'UTF8'),'sha256'),
    'hex'
  );
  v_blob text := encode(
    extensions.digest(convert_to('pa-blob:'||p_tag,'UTF8'),'sha1'),
    'hex'
  );
  v_commit text := encode(
    extensions.digest(convert_to('pa-commit:'||p_tag,'UTF8'),'sha1'),
    'hex'
  );
  v_path text := 'research/'||v_run::text||'/evidence-ledger.json';
  v_uri text;
  v_candidate jsonb;
  v_payload jsonb;
begin
  select m.issuer_id,m.security_id,m.dossier_id
    into v_issuer,v_security,v_dossier
  from public.legacy_company_identity_map m
  order by m.legacy_company_id
  limit 1;

  if v_issuer is null or v_security is null or v_dossier is null then
    raise exception 'PA identity fixture unavailable';
  end if;

  insert into public.orotitan_runs(
    run_id,creation_idempotency_key,run_scope,
    issuer_id,security_id,dossier_id,
    entry_path,canonical_mode,run_type,
    run_status,current_stage,data_cutoff,
    process_version,pilotage_contract_version,
    contract_pins,contract_set_sha256,state_version
  ) values (
    v_run,'pa:'||p_tag||':'||v_run::text,'COMPANY_ANALYSIS',
    v_issuer,v_security,v_dossier,
    'IMPOSED_COMPANY','ANALYZE','INITIAL',
    'ACTIVE','RESEARCH',date '2026-09-19',
    v_pins->'process'->>'version',v_pins->'pilotage'->>'version',
    v_pins,v_contract_hash,1
  );

  insert into public.orotitan_run_stages(
    run_id,stage_code,stage_revision,
    stage_contract_name,stage_contract_version,stage_contract_sha256,
    lifecycle_status,contract_status_code,
    handoff_gate_name,handoff_gate_state,
    state_version,started_at
  ) values (
    v_run,'RESEARCH',1,
    v_pins->'research_stage'->>'name',
    v_pins->'research_stage'->>'version',
    v_pins->'research_stage'->>'content_sha256',
    'IN_PROGRESS','RESEARCH_IN_PROGRESS',
    'READY_FOR_DEEP_DIVE','NOT_EVALUATED',
    1,'2026-09-21T00:00:00Z'
  );

  v_uri := 'github://robzer13/real-orotitan@'||v_commit||'/'||v_path;

  v_candidate := jsonb_build_object(
    'artifact_id',v_artifact,
    'version',1,
    'artifact_type','EVIDENCE_LEDGER',
    'logical_name','evidence_ledger',
    'authority_class','AUTHORITATIVE_STAGE_OUTPUT',
    'authority_state','AUTHORITATIVE',
    'artifact_status','SEALED',
    'availability_state','AVAILABLE',
    'media_type','application/json',
    'size_bytes',v_size,
    'content_sha256',v_sha,
    'storage_backend','PRIVATE_GITHUB',
    'storage_uri',v_uri,
    'github_repository','robzer13/real-orotitan',
    'github_path',v_path,
    'github_commit_sha',v_commit,
    'github_blob_sha',v_blob
  );

  perform public.stage_orotitan_persistence_candidate(
    v_run,'RESEARCH',v_candidate
  );

  if not exists (
    select 1 from public.orotitan_artifacts
    where run_id=v_run
      and stage_code='RESEARCH'
      and artifact_id=v_artifact
      and version=1
      and authority_state='NON_AUTHORITATIVE'
      and artifact_status='SEALED'
      and availability_state='AVAILABLE'
  ) then
    raise exception 'PA staged Registry authority row missing';
  end if;

  v_payload := jsonb_build_object(
    'attestation_schema_version','1.0',
    'verification_method','GITHUB_CONNECTOR_PRIVATE_REREAD_V1',
    'trust_boundary','SUPABASE_MANAGEMENT_PLANE',
    'storage_backend','PRIVATE_GITHUB',
    'run_id',v_run,
    'stage_code','RESEARCH',
    'artifact_id',v_artifact,
    'version',1,
    'artifact_type','EVIDENCE_LEDGER',
    'github_repository','robzer13/real-orotitan',
    'github_path',v_path,
    'github_commit_sha',v_commit,
    'github_blob_sha',v_blob,
    'storage_uri',v_uri,
    'size_bytes',v_size,
    'content_sha256',v_sha,
    'commit_path_resolved',true,
    'verified_at','2026-09-21T00:00:01Z'
  );

  return jsonb_build_object(
    'run_id',v_run,
    'artifact_id',v_artifact,
    'candidate',v_candidate,
    'payload',v_payload
  );
end;
$$;

create or replace function pg_temp.pa_expect_reject(
  p_run_id uuid,
  p_payload jsonb,
  p_expected text
)
returns void
language plpgsql
as $$
declare
  v_rejected boolean := false;
  v_message text;
  v_before bigint;
begin
  select count(*) into v_before
  from public.orotitan_run_events
  where run_id=p_run_id and event_type='PERSISTENCE_ATTESTED';

  begin
    perform public.attest_orotitan_persistence_locator(
      p_run_id,'RESEARCH',p_payload
    );
  exception when others then
    v_rejected := true;
    v_message := sqlerrm;
  end;

  if not v_rejected then
    raise exception 'PA expected rejection was accepted: %',p_expected;
  end if;

  if v_message not like '%'||p_expected||'%' then
    raise exception 'PA rejection mismatch. expected %, got %',p_expected,v_message;
  end if;

  if (select count(*) from public.orotitan_run_events
      where run_id=p_run_id and event_type='PERSISTENCE_ATTESTED') <> v_before then
    raise exception 'PA rejected request inserted persistence event';
  end if;
end;
$$;

-- Positive exact authority receipt, identical replay, and reachable fingerprint conflict.
do $$
declare
  f jsonb := pg_temp.pa_fixture('positive');
  r uuid := (f->>'run_id')::uuid;
  p jsonb := f->'payload';
  e1 uuid;
  e2 uuid;
  p_conflict jsonb;
begin
  e1 := public.attest_orotitan_persistence_locator(r,'RESEARCH',p);
  e2 := public.attest_orotitan_persistence_locator(r,'RESEARCH',p);

  if e1 is null or e2 is distinct from e1 then
    raise exception 'PA identical replay did not return same canonical event UUID';
  end if;

  if (select count(*) from public.orotitan_run_events
      where run_id=r and event_type='PERSISTENCE_ATTESTED') <> 1 then
    raise exception 'PA identical replay created duplicate event';
  end if;

  if (select authority_state from public.orotitan_artifacts
      where run_id=r and artifact_id=(f->>'artifact_id')::uuid and version=1)
     <> 'NON_AUTHORITATIVE' then
    raise exception 'PA attestation improperly promoted artifact authority';
  end if;

  p_conflict := jsonb_set(
    p,'{verified_at}',to_jsonb('2026-09-21T00:00:02Z'::text)
  );
  perform pg_temp.pa_expect_reject(r,p_conflict,'IDEMPOTENCY_CONFLICT');
end;
$$;

-- Negative and wrong-positive size.
do $$
declare
  f jsonb;
  r uuid;
  p jsonb;
begin
  f := pg_temp.pa_fixture('negative-size');
  r := (f->>'run_id')::uuid;
  p := jsonb_set(f->'payload','{size_bytes}',to_jsonb((-1)::bigint));
  perform pg_temp.pa_expect_reject(r,p,'PERSISTENCE_ATTESTATION_INVALID');

  f := pg_temp.pa_fixture('wrong-size');
  r := (f->>'run_id')::uuid;
  p := jsonb_set(
    f->'payload','{size_bytes}',
    to_jsonb(((f->'payload'->>'size_bytes')::bigint+1))
  );
  perform pg_temp.pa_expect_reject(r,p,'PERSISTENCE_ATTESTATION_AUTHORITY_MISMATCH');
end;
$$;

-- SHA-256 syntax and exact Registry equality.
do $$
declare
  f jsonb;
  r uuid;
  p jsonb;
begin
  f := pg_temp.pa_fixture('malformed-sha');
  r := (f->>'run_id')::uuid;
  p := jsonb_set(f->'payload','{content_sha256}',to_jsonb('xyz'::text));
  perform pg_temp.pa_expect_reject(r,p,'PERSISTENCE_ATTESTATION_INVALID');

  f := pg_temp.pa_fixture('wrong-sha');
  r := (f->>'run_id')::uuid;
  p := jsonb_set(f->'payload','{content_sha256}',to_jsonb(repeat('0',64)));
  perform pg_temp.pa_expect_reject(r,p,'PERSISTENCE_ATTESTATION_AUTHORITY_MISMATCH');
end;
$$;

-- Git blob SHA syntax and exact Registry equality.
do $$
declare
  f jsonb;
  r uuid;
  p jsonb;
begin
  f := pg_temp.pa_fixture('malformed-blob');
  r := (f->>'run_id')::uuid;
  p := jsonb_set(f->'payload','{github_blob_sha}',to_jsonb('bad'::text));
  perform pg_temp.pa_expect_reject(r,p,'PERSISTENCE_ATTESTATION_INVALID');

  f := pg_temp.pa_fixture('wrong-blob');
  r := (f->>'run_id')::uuid;
  p := jsonb_set(f->'payload','{github_blob_sha}',to_jsonb(repeat('0',40)));
  perform pg_temp.pa_expect_reject(r,p,'PERSISTENCE_ATTESTATION_AUTHORITY_MISMATCH');
end;
$$;

-- Repository, immutable path, and artifact identity authority mismatch.
do $$
declare
  f jsonb;
  r uuid;
  p jsonb;
  bad_path text;
begin
  f := pg_temp.pa_fixture('wrong-repository');
  r := (f->>'run_id')::uuid;
  p := jsonb_set(
    f->'payload','{github_repository}',
    to_jsonb('robzer13/not-authoritative'::text)
  );
  perform pg_temp.pa_expect_reject(r,p,'PERSISTENCE_ATTESTATION_INVALID');

  f := pg_temp.pa_fixture('wrong-path');
  r := (f->>'run_id')::uuid;
  bad_path := 'research/'||r::text||'/syntactically-valid-but-nonexistent.json';
  p := jsonb_set(f->'payload','{github_path}',to_jsonb(bad_path));
  p := jsonb_set(
    p,'{storage_uri}',
    to_jsonb(
      'github://robzer13/real-orotitan@'||
      (p->>'github_commit_sha')||'/'||bad_path
    )
  );
  perform pg_temp.pa_expect_reject(r,p,'PERSISTENCE_ATTESTATION_AUTHORITY_MISMATCH');

  f := pg_temp.pa_fixture('artifact-identity');
  r := (f->>'run_id')::uuid;
  p := jsonb_set(
    f->'payload','{artifact_id}',to_jsonb(gen_random_uuid()::text)
  );
  perform pg_temp.pa_expect_reject(r,p,'PERSISTENCE_ATTESTATION_ARTIFACT_NOT_REGISTERED');
end;
$$;

-- Commit and storage URI are also authority-bound; protect against self-consistent
-- locator substitution outside the explicit minimum matrix.
do $$
declare
  f jsonb;
  r uuid;
  p jsonb;
  bad_commit text := repeat('f',40);
begin
  f := pg_temp.pa_fixture('wrong-commit');
  r := (f->>'run_id')::uuid;
  p := jsonb_set(f->'payload','{github_commit_sha}',to_jsonb(bad_commit));
  p := jsonb_set(
    p,'{storage_uri}',
    to_jsonb(
      'github://robzer13/real-orotitan@'||bad_commit||'/'||(p->>'github_path')
    )
  );
  perform pg_temp.pa_expect_reject(r,p,'PERSISTENCE_ATTESTATION_AUTHORITY_MISMATCH');
end;
$$;

select jsonb_build_object(
  'valid_exact_authority_receipt','PASS',
  'identical_replay_same_event_uuid','PASS',
  'fingerprint_conflict','REJECT',
  'negative_size','REJECT',
  'wrong_positive_size','REJECT',
  'malformed_sha256','REJECT',
  'valid_format_wrong_sha256','REJECT',
  'malformed_git_blob_sha','REJECT',
  'valid_format_wrong_git_blob_sha','REJECT',
  'wrong_repository','REJECT',
  'wrong_nonexistent_immutable_locator','REJECT',
  'artifact_identity_mismatch','REJECT',
  'wrong_commit','REJECT',
  'attestation_authority_promotion','NONE',
  'result','PASS_ALL'
) as orotitan_registry_v1_11_persistence_attestation_authority_regression;
