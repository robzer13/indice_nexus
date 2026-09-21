\set ON_ERROR_STOP on

-- OroTitan Registry bundle-wide persistence integrity regression.
-- Runs only after the forward migration. Existing V1.6/V1.7/V1.8 matrices
-- are executed at the historical V1.8 frontier before this file.

create or replace function pg_temp.bpi_pins()
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
    v_hash := encode(extensions.digest(convert_to('bpi:' || v_key,'UTF8'),'sha256'),'hex');
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

create or replace function pg_temp.bpi_create_run(p_tag text)
returns table(run_id uuid, issuer_id uuid, security_id uuid, dossier_id uuid)
language plpgsql
as $$
declare
  v_run uuid := gen_random_uuid();
  v_issuer uuid;
  v_security uuid;
  v_dossier uuid;
  v_pins jsonb := pg_temp.bpi_pins();
  v_contract_hash text := public.orotitan_contract_set_sha256(v_pins);
begin
  select m.issuer_id,m.security_id,m.dossier_id
    into v_issuer,v_security,v_dossier
  from public.legacy_company_identity_map m
  order by m.legacy_company_id
  limit 1;

  if v_issuer is null or v_security is null or v_dossier is null then
    raise exception 'BPI identity fixture unavailable';
  end if;

  insert into public.orotitan_runs(
    run_id,creation_idempotency_key,run_scope,
    issuer_id,security_id,dossier_id,
    entry_path,canonical_mode,run_type,
    run_status,current_stage,data_cutoff,
    process_version,pilotage_contract_version,
    contract_pins,contract_set_sha256,state_version
  ) values (
    v_run,'bpi:'||p_tag||':'||v_run::text,'COMPANY_ANALYSIS',
    v_issuer,v_security,v_dossier,
    'IMPOSED_COMPANY','ANALYZE','INITIAL',
    'ACTIVE','INTEGRATION',date '2026-09-19',
    v_pins->'process'->>'version',v_pins->'pilotage'->>'version',
    v_pins,v_contract_hash,1
  );

  -- Visa-shaped upstream admission: Deep Dive is complete and says
  -- READY_FOR_INTEGRATION=YES. No baseline canonical snapshot is bound.
  insert into public.orotitan_run_stages(
    run_id,stage_code,stage_revision,
    stage_contract_name,stage_contract_version,stage_contract_sha256,
    lifecycle_status,contract_status_code,
    handoff_gate_name,handoff_gate_state,
    state_version,started_at,completed_at
  ) values (
    v_run,'DEEP_DIVE',1,
    v_pins->'deep_dive_stage'->>'name',
    v_pins->'deep_dive_stage'->>'version',
    v_pins->'deep_dive_stage'->>'content_sha256',
    'COMPLETE','COMPLETE',
    'READY_FOR_INTEGRATION','YES',
    1,'2026-09-20T10:00:00Z','2026-09-20T11:00:00Z'
  );

  insert into public.orotitan_run_stages(
    run_id,stage_code,stage_revision,
    stage_contract_name,stage_contract_version,stage_contract_sha256,
    lifecycle_status,contract_status_code,
    handoff_gate_name,handoff_gate_state,
    state_version,started_at
  ) values (
    v_run,'INTEGRATION',1,
    v_pins->'integration_stage'->>'name',
    v_pins->'integration_stage'->>'version',
    v_pins->'integration_stage'->>'content_sha256',
    'IN_PROGRESS','INTEGRATION_IN_PROGRESS',
    'READY_TO_PUBLISH','NOT_EVALUATED',
    1,'2026-09-20T12:00:00Z'
  );

  run_id := v_run;
  issuer_id := v_issuer;
  security_id := v_security;
  dossier_id := v_dossier;
  return next;
end;
$$;

create or replace function pg_temp.bpi_attested_registration(
  p_run_id uuid,
  p_stage text,
  p_artifact_id uuid,
  p_version integer,
  p_artifact_type text,
  p_logical_name text,
  p_content_text text,
  p_path text,
  p_authority_class text,
  p_authority_state text,
  p_expected_json jsonb default null,
  p_repository text default 'robzer13/real-orotitan',
  p_commit text default null
)
returns jsonb
language plpgsql
as $$
declare
  v_bytes bytea := convert_to(p_content_text,'UTF8');
  v_size bigint := octet_length(v_bytes);
  v_sha text := encode(extensions.digest(v_bytes,'sha256'),'hex');
  v_blob text;
  v_commit text := coalesce(p_commit,repeat('c',40));
  v_uri text;
  v_verified_at timestamptz := clock_timestamp();
  v_event_id uuid := gen_random_uuid();
  v_payload jsonb;
  v_fingerprint text;
  v_registration jsonb;
begin
  v_blob := encode(
    extensions.digest(
      convert_to('blob ' || v_size::text,'UTF8') || decode('00','hex') || v_bytes,
      'sha1'
    ),
    'hex'
  );
  v_uri := 'github://' || p_repository || '@' || v_commit || '/' || p_path;

  v_payload := jsonb_build_object(
    'attestation_schema_version','1.0',
    'verification_method','GITHUB_CONNECTOR_PRIVATE_REREAD_V1',
    'trust_boundary','SUPABASE_MANAGEMENT_PLANE',
    'storage_backend','PRIVATE_GITHUB',
    'run_id',p_run_id,
    'stage_code',p_stage,
    'artifact_id',p_artifact_id,
    'version',p_version,
    'artifact_type',p_artifact_type,
    'github_repository',p_repository,
    'github_path',p_path,
    'github_commit_sha',v_commit,
    'github_blob_sha',v_blob,
    'storage_uri',v_uri,
    'size_bytes',v_size,
    'content_sha256',v_sha,
    'commit_path_resolved',true,
    'verified_at',v_verified_at
  );
  -- Exercise the production management-plane attestation helper rather than
  -- synthesizing the event directly. This catches runtime expression defects
  -- in the attestation boundary itself.
  v_event_id := public.attest_orotitan_persistence_locator(
    p_run_id,
    p_stage,
    v_payload
  );

  v_registration := jsonb_build_object(
    'artifact_id',p_artifact_id,
    'version',p_version,
    'artifact_type',p_artifact_type,
    'logical_name',p_logical_name,
    'authority_class',p_authority_class,
    'artifact_status','SEALED',
    'authority_state',p_authority_state,
    'availability_state','AVAILABLE',
    'media_type','application/json',
    'size_bytes',v_size,
    'content_sha256',v_sha,
    'storage_backend','PRIVATE_GITHUB',
    'storage_uri',v_uri,
    'github_repository',p_repository,
    'github_path',p_path,
    'github_commit_sha',v_commit,
    'github_blob_sha',v_blob,
    'persistence_receipt',jsonb_build_object(
      'receipt_schema_version','1.1',
      'verification_method','PRIVATE_GITHUB_ATTESTED_REREAD_EXACT_BYTES_V1',
      'storage_backend','PRIVATE_GITHUB',
      'run_id',p_run_id,
      'stage_code',p_stage,
      'artifact_id',p_artifact_id,
      'version',p_version,
      'artifact_type',p_artifact_type,
      'github_repository',p_repository,
      'github_path',p_path,
      'github_commit_sha',v_commit,
      'github_blob_sha',v_blob,
      'attestation_event_id',v_event_id,
      'commit_path_resolved',true,
      'verified_content_base64',encode(v_bytes,'base64'),
      'verified_at',v_verified_at
    )
  );

  if p_expected_json is not null then
    v_registration := v_registration || jsonb_build_object('canonical_json_content',p_expected_json);
  end if;

  return v_registration;
end;
$$;

create or replace function pg_temp.bpi_ref(p_registration jsonb)
returns jsonb
language sql
as $$
  select jsonb_build_object(
    'artifact_id',p_registration->>'artifact_id',
    'version',(p_registration->>'version')::integer,
    'artifact_type',p_registration->>'artifact_type',
    'content_sha256',p_registration->>'content_sha256',
    'authority_class',p_registration->>'authority_class',
    'media_type',p_registration->>'media_type',
    'size_bytes',(p_registration->>'size_bytes')::bigint,
    'storage_ref',jsonb_build_object(
      'backend','PRIVATE_GITHUB',
      'repository',p_registration->>'github_repository',
      'path',p_registration->>'github_path',
      'commit_sha',p_registration->>'github_commit_sha',
      'blob_sha',p_registration->>'github_blob_sha',
      'storage_uri',p_registration->>'storage_uri'
    )
  );
$$;

create or replace function pg_temp.bpi_fixture(p_tag text)
returns jsonb
language plpgsql
as $$
declare
  v_run uuid;
  v_issuer uuid;
  v_security uuid;
  v_dossier uuid;
  v_manifest_id uuid := gen_random_uuid();
  v_output_id uuid;
  v_types text[] := array[
    'CANONICAL_SNAPSHOT_CANDIDATE',
    'INTEGRATION_MAPPING_RECORD',
    'SCHEMA_VALIDATION_REPORT',
    'I2_RECONCILIATION_REPORT',
    'HISTORY_TRANSITION_VALIDATION_REPORT',
    'I3B_ADMISSION_REPORT',
    'PRE_PUBLICATION_CONTROL_CARD'
  ];
  v_type text;
  v_i integer := 0;
  v_raw text;
  v_json jsonb;
  v_reg jsonb;
  v_outputs jsonb := '[]'::jsonb;
  v_refs jsonb := '[]'::jsonb;
  v_manifest jsonb;
  v_manifest_reg jsonb;
  v_edges jsonb := '[]'::jsonb;
  v_stage public.orotitan_run_stages%rowtype;
  v_runrow public.orotitan_runs%rowtype;
begin
  select run_id,issuer_id,security_id,dossier_id
    into v_run,v_issuer,v_security,v_dossier
  from pg_temp.bpi_create_run(p_tag);

  foreach v_type in array v_types
  loop
    v_i := v_i + 1;
    v_output_id := gen_random_uuid();
    if v_i = 1 then
      -- Deliberately non-canonical whitespace/order: byte identity must use the
      -- exact persisted bytes while semantic JSON equality still passes.
      v_raw := E'{\n  "items": [1, 2],\n  "ordinal": 1,\n  "artifact_type": "' || v_type || E'"\n}\n';
    else
      v_raw := jsonb_build_object(
        'artifact_type',v_type,
        'ordinal',v_i,
        'items',jsonb_build_array(1,2)
      )::text;
    end if;
    v_json := v_raw::jsonb;
    v_reg := pg_temp.bpi_attested_registration(
      v_run,'INTEGRATION',v_output_id,1,v_type,lower(v_type),
      v_raw,'integration/'||v_run::text||'/'||lower(v_type)||'.json',
      'AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE',v_json
    );
    v_outputs := v_outputs || jsonb_build_array(v_reg);
    v_refs := v_refs || jsonb_build_array(pg_temp.bpi_ref(v_reg));
  end loop;

  select * into v_runrow from public.orotitan_runs where run_id=v_run;
  select * into v_stage from public.orotitan_run_stages
    where run_id=v_run and stage_code='INTEGRATION';

  v_manifest := jsonb_build_object(
    'manifest_schema_version','1.0.0',
    'manifest_id',v_manifest_id,
    'manifest_kind','FINAL',
    'run_id',v_run,
    'stage','INTEGRATION',
    'stage_revision',1,
    'issuer_id',v_issuer,
    'security_id',v_security,
    'dossier_id',v_dossier,
    'canonical_mode',v_runrow.canonical_mode,
    'run_type',v_runrow.run_type,
    'data_cutoff',v_runrow.data_cutoff::text,
    'baseline_snapshot_id',null,
    'process_version',v_runrow.process_version,
    'pilotage_contract_version',v_runrow.pilotage_contract_version,
    'contract_pins',v_runrow.contract_pins,
    'stage_contract',v_runrow.contract_pins->'integration_stage',
    'contract_set_sha256',v_runrow.contract_set_sha256,
    'input_artifacts','[]'::jsonb,
    'output_artifacts',v_refs,
    'stage_status','COMPLETE',
    'contract_status_code','COMPLETE_READY_TO_PUBLISH',
    'handoff_gate',jsonb_build_object('name','READY_TO_PUBLISH','state','YES'),
    'critical_blockers','[]'::jsonb,
    'open_material_limitations','[]'::jsonb,
    'parent_manifests','[]'::jsonb,
    'started_at','2026-09-20T12:00:00Z',
    'completed_at','2026-09-20T13:00:00Z'
  );

  -- Pretty bytes prove formatting changes are accepted only when exact
  -- persisted bytes are represented by the receipt and JSON value is equal.
  v_manifest_reg := pg_temp.bpi_attested_registration(
    v_run,'INTEGRATION',v_manifest_id,1,'INTEGRATION_STAGE_MANIFEST',
    'integration_stage_manifest',jsonb_pretty(v_manifest),
    'integration/'||v_run::text||'/stage-manifest.json',
    'AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE',v_manifest
  );

  select coalesce(jsonb_agg(jsonb_build_object(
    'child_run_id',v_run,
    'child_artifact_id',v_manifest_id,
    'child_version',1,
    'parent_run_id',v_run,
    'parent_artifact_id',value->>'artifact_id',
    'parent_version',(value->>'version')::integer,
    'relation_type','CONSUMES'
  ) order by ordinality),'[]'::jsonb)
    into v_edges
  from jsonb_array_elements(v_outputs) with ordinality;

  return jsonb_build_object(
    'run_id',v_run,
    'manifest',v_manifest,
    'manifest_registration',v_manifest_reg,
    'output_artifacts',v_outputs,
    'edges',v_edges
  );
end;
$$;

create or replace function pg_temp.bpi_expect_reject(p_bundle jsonb, p_tag text)
returns void
language plpgsql
as $$
declare
  v_run uuid := (p_bundle->>'run_id')::uuid;
  v_run_state bigint;
  v_stage_state bigint;
  v_artifacts_before bigint;
  v_active_before uuid;
  v_rejected boolean := false;
begin
  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  select state_version,active_manifest_artifact_id
    into v_stage_state,v_active_before
  from public.orotitan_run_stages where run_id=v_run and stage_code='INTEGRATION';
  select count(*) into v_artifacts_before
  from public.orotitan_artifacts where run_id=v_run and stage_code='INTEGRATION';

  begin
    perform public.finalize_orotitan_stage(
      v_run,'INTEGRATION',v_run_state,v_stage_state,
      p_bundle->'manifest',
      p_bundle->'manifest_registration',
      p_bundle->'output_artifacts',
      p_bundle->'edges',
      'bpi:reject:'||p_tag||':'||v_run::text,
      encode(extensions.digest(convert_to('bpi:'||p_tag||':'||v_run::text,'UTF8'),'sha256'),'hex'),
      'SYSTEM'
    );
  exception when others then
    v_rejected := true;
  end;

  if not v_rejected then
    raise exception 'BPI % unexpectedly accepted',p_tag;
  end if;

  if (select state_version from public.orotitan_runs where run_id=v_run) <> v_run_state
     or (select state_version from public.orotitan_run_stages where run_id=v_run and stage_code='INTEGRATION') <> v_stage_state
     or (select count(*) from public.orotitan_artifacts where run_id=v_run and stage_code='INTEGRATION') <> v_artifacts_before
     or (select active_manifest_artifact_id from public.orotitan_run_stages where run_id=v_run and stage_code='INTEGRATION') is distinct from v_active_before then
    raise exception 'BPI % violated atomic rejection',p_tag;
  end if;
end;
$$;

-- Positive control: Visa-shaped FINAL Integration bundle closes 8/8 receipts,
-- preserves semantic JSON formatting tolerance, registers lineage, and finalizes.
do $$
declare
  b jsonb := pg_temp.bpi_fixture('positive-visa-shape');
  r uuid := (b->>'run_id')::uuid;
  run_v bigint;
  stage_v bigint;
  res1 jsonb;
  res2 jsonb;
  rejected boolean := false;
begin
  select state_version into run_v from public.orotitan_runs where run_id=r;
  select state_version into stage_v from public.orotitan_run_stages where run_id=r and stage_code='INTEGRATION';

  res1 := public.finalize_orotitan_stage(
    r,'INTEGRATION',run_v,stage_v,
    b->'manifest',b->'manifest_registration',b->'output_artifacts',b->'edges',
    'bpi:positive:'||r::text,repeat('1',64),'SYSTEM'
  );

  if (select count(*) from public.orotitan_artifacts
      where run_id=r and stage_code='INTEGRATION'
        and authority_state='AUTHORITATIVE' and availability_state='AVAILABLE') <> 8 then
    raise exception 'BPI positive bundle did not close 8/8 authoritative objects';
  end if;
  if (select active_manifest_artifact_id from public.orotitan_run_stages
      where run_id=r and stage_code='INTEGRATION') is distinct from ((b->'manifest'->>'manifest_id')::uuid)
     or (select run_status from public.orotitan_runs where run_id=r) <> 'READY_TO_PUBLISH'
     or (select count(*) from public.orotitan_artifact_edges where child_run_id=r) <> 7 then
    raise exception 'BPI positive finalization/lineage state mismatch';
  end if;

  -- Same idempotency key/fingerprint returns replay without a second transition.
  res2 := public.finalize_orotitan_stage(
    r,'INTEGRATION',run_v,stage_v,
    b->'manifest',b->'manifest_registration',b->'output_artifacts',b->'edges',
    'bpi:positive:'||r::text,repeat('1',64),'SYSTEM'
  );
  if coalesce((res2->>'idempotent_replay')::boolean,false) is not true then
    raise exception 'BPI idempotent replay failed';
  end if;

  -- Same key with another fingerprint must conflict.
  begin
    perform public.finalize_orotitan_stage(
      r,'INTEGRATION',run_v,stage_v,
      b->'manifest',b->'manifest_registration',b->'output_artifacts',b->'edges',
      'bpi:positive:'||r::text,repeat('2',64),'SYSTEM'
    );
  exception when others then rejected := true; end;
  if not rejected then raise exception 'BPI fingerprint conflict accepted'; end if;
end;
$$;

-- Output byte/identity/version/manifest-reference and locator-authority matrix.
do $$
declare
  b jsonb;
  out0 jsonb;
  out1 jsonb;
  bad text;
begin
  b := pg_temp.bpi_fixture('wrong-sha');
  b := jsonb_set(b,'{output_artifacts,0,content_sha256}',to_jsonb(repeat('0',64)));
  b := jsonb_set(b,'{manifest,output_artifacts,0,content_sha256}',to_jsonb(repeat('0',64)));
  perform pg_temp.bpi_expect_reject(b,'OUTPUT_WRONG_SHA');

  b := pg_temp.bpi_fixture('wrong-size');
  b := jsonb_set(b,'{output_artifacts,0,size_bytes}',to_jsonb(((b->'output_artifacts'->0->>'size_bytes')::bigint+1)));
  b := jsonb_set(b,'{manifest,output_artifacts,0,size_bytes}',to_jsonb(((b->'manifest'->'output_artifacts'->0->>'size_bytes')::bigint+1)));
  perform pg_temp.bpi_expect_reject(b,'OUTPUT_WRONG_SIZE');

  b := pg_temp.bpi_fixture('wrong-blob');
  bad := repeat('0',40);
  b := jsonb_set(b,'{output_artifacts,0,github_blob_sha}',to_jsonb(bad));
  b := jsonb_set(b,'{output_artifacts,0,persistence_receipt,github_blob_sha}',to_jsonb(bad));
  b := jsonb_set(b,'{manifest,output_artifacts,0,storage_ref,blob_sha}',to_jsonb(bad));
  perform pg_temp.bpi_expect_reject(b,'OUTPUT_WRONG_GIT_BLOB');

  b := pg_temp.bpi_fixture('artifact-id-mismatch');
  out0 := b->'output_artifacts'->0;
  b := jsonb_set(b,'{output_artifacts,1,persistence_receipt}',out0->'persistence_receipt');
  perform pg_temp.bpi_expect_reject(b,'OUTPUT_ARTIFACT_ID_MISMATCH');

  b := pg_temp.bpi_fixture('version-mismatch');
  b := jsonb_set(b,'{output_artifacts,0,version}','2'::jsonb);
  b := jsonb_set(b,'{manifest,output_artifacts,0,version}','2'::jsonb);
  perform pg_temp.bpi_expect_reject(b,'OUTPUT_VERSION_MISMATCH');

  b := pg_temp.bpi_fixture('manifest-ref-mismatch');
  b := jsonb_set(b,'{manifest,output_artifacts,0,content_sha256}',to_jsonb(repeat('f',64)));
  perform pg_temp.bpi_expect_reject(b,'OUTPUT_MANIFEST_REFERENCE_MISMATCH');

  b := pg_temp.bpi_fixture('wrong-repository');
  b := jsonb_set(b,'{output_artifacts,0,github_repository}',to_jsonb('robzer13/not-authoritative'::text));
  b := jsonb_set(b,'{output_artifacts,0,persistence_receipt,github_repository}',to_jsonb('robzer13/not-authoritative'::text));
  b := jsonb_set(b,'{manifest,output_artifacts,0,storage_ref,repository}',to_jsonb('robzer13/not-authoritative'::text));
  b := jsonb_set(b,'{output_artifacts,0,storage_uri}',to_jsonb(
    'github://robzer13/not-authoritative@'||(b->'output_artifacts'->0->>'github_commit_sha')||'/'||(b->'output_artifacts'->0->>'github_path')
  ));
  b := jsonb_set(b,'{manifest,output_artifacts,0,storage_ref,storage_uri}',b->'output_artifacts'->0->'storage_uri');
  perform pg_temp.bpi_expect_reject(b,'SELF_CONSISTENT_WRONG_REPOSITORY');

  b := pg_temp.bpi_fixture('wrong-path');
  bad := 'integration/self-consistent-wrong-path.json';
  b := jsonb_set(b,'{output_artifacts,0,github_path}',to_jsonb(bad));
  b := jsonb_set(b,'{output_artifacts,0,persistence_receipt,github_path}',to_jsonb(bad));
  b := jsonb_set(b,'{manifest,output_artifacts,0,storage_ref,path}',to_jsonb(bad));
  b := jsonb_set(b,'{output_artifacts,0,storage_uri}',to_jsonb(
    'github://'||(b->'output_artifacts'->0->>'github_repository')||'@'||(b->'output_artifacts'->0->>'github_commit_sha')||'/'||bad
  ));
  b := jsonb_set(b,'{manifest,output_artifacts,0,storage_ref,storage_uri}',b->'output_artifacts'->0->'storage_uri');
  perform pg_temp.bpi_expect_reject(b,'SELF_CONSISTENT_WRONG_PATH');

  b := pg_temp.bpi_fixture('wrong-commit');
  bad := repeat('d',40);
  b := jsonb_set(b,'{output_artifacts,0,github_commit_sha}',to_jsonb(bad));
  b := jsonb_set(b,'{output_artifacts,0,persistence_receipt,github_commit_sha}',to_jsonb(bad));
  b := jsonb_set(b,'{manifest,output_artifacts,0,storage_ref,commit_sha}',to_jsonb(bad));
  b := jsonb_set(b,'{output_artifacts,0,storage_uri}',to_jsonb(
    'github://'||(b->'output_artifacts'->0->>'github_repository')||'@'||bad||'/'||(b->'output_artifacts'->0->>'github_path')
  ));
  b := jsonb_set(b,'{manifest,output_artifacts,0,storage_ref,storage_uri}',b->'output_artifacts'->0->'storage_uri');
  perform pg_temp.bpi_expect_reject(b,'SELF_CONSISTENT_WRONG_COMMIT');

  -- Pair the valid blob with a false path while keeping all caller claims
  -- self-consistent: the independent attestation must reject path/blob membership.
  b := pg_temp.bpi_fixture('path-blob-membership');
  bad := 'integration/path-does-not-own-valid-blob.json';
  b := jsonb_set(b,'{output_artifacts,0,github_path}',to_jsonb(bad));
  b := jsonb_set(b,'{output_artifacts,0,persistence_receipt,github_path}',to_jsonb(bad));
  b := jsonb_set(b,'{manifest,output_artifacts,0,storage_ref,path}',to_jsonb(bad));
  b := jsonb_set(b,'{output_artifacts,0,storage_uri}',to_jsonb(
    'github://'||(b->'output_artifacts'->0->>'github_repository')||'@'||(b->'output_artifacts'->0->>'github_commit_sha')||'/'||bad
  ));
  b := jsonb_set(b,'{manifest,output_artifacts,0,storage_ref,storage_uri}',b->'output_artifacts'->0->'storage_uri');
  perform pg_temp.bpi_expect_reject(b,'PATH_BLOB_MEMBERSHIP');
end;
$$;

-- JSON semantic equality matrix on exact persisted bytes.
do $$
declare
  b jsonb;
  original jsonb;
begin
  b := pg_temp.bpi_fixture('json-value');
  b := jsonb_set(b,'{output_artifacts,0,canonical_json_content}','{"artifact_type":"CANONICAL_SNAPSHOT_CANDIDATE","ordinal":999,"items":[1,2]}'::jsonb);
  perform pg_temp.bpi_expect_reject(b,'DIFFERENT_JSON_VALUE');

  b := pg_temp.bpi_fixture('json-array-order');
  original := b->'output_artifacts'->0->'canonical_json_content';
  b := jsonb_set(b,'{output_artifacts,0,canonical_json_content}',jsonb_set(original,'{items}','[2,1]'::jsonb));
  perform pg_temp.bpi_expect_reject(b,'ARRAY_ORDER');

  b := pg_temp.bpi_fixture('json-missing-field');
  original := (b->'output_artifacts'->0->'canonical_json_content') - 'ordinal';
  b := jsonb_set(b,'{output_artifacts,0,canonical_json_content}',original);
  perform pg_temp.bpi_expect_reject(b,'MISSING_JSON_FIELD');

  b := pg_temp.bpi_fixture('json-extra-field');
  original := (b->'output_artifacts'->0->'canonical_json_content') || '{"extra":true}'::jsonb;
  b := jsonb_set(b,'{output_artifacts,0,canonical_json_content}',original);
  perform pg_temp.bpi_expect_reject(b,'EXTRA_JSON_FIELD');
end;
$$;

-- Closure and mixed-validity atomicity.
do $$
declare
  b jsonb;
begin
  b := pg_temp.bpi_fixture('missing-one-output-receipt');
  b := jsonb_set(b,'{output_artifacts,6}',(b->'output_artifacts'->6)-'persistence_receipt');
  perform pg_temp.bpi_expect_reject(b,'MISSING_ONE_OUTPUT_RECEIPT');

  b := pg_temp.bpi_fixture('mixed-validity');
  b := jsonb_set(b,'{output_artifacts,3,canonical_json_content}','{"mixed":"invalid"}'::jsonb);
  perform pg_temp.bpi_expect_reject(b,'MIXED_VALIDITY_ZERO_PARTIAL_AUTHORITY');
end;
$$;

-- Stage Manifest V1.8 controls under the stronger attested authority model.
do $$
declare
  b jsonb;
  bad text;
  first_ref jsonb;
  second_ref jsonb;
begin
  b := pg_temp.bpi_fixture('manifest-version');
  b := jsonb_set(b,'{manifest_registration,version}','2'::jsonb);
  b := jsonb_set(b,'{manifest_registration,persistence_receipt,version}','2'::jsonb);
  perform pg_temp.bpi_expect_reject(b,'WRONG_MANIFEST_VERSION_CLAIM');

  b := pg_temp.bpi_fixture('manifest-repo');
  b := jsonb_set(b,'{manifest_registration,github_repository}',to_jsonb('robzer13/not-authoritative'::text));
  b := jsonb_set(b,'{manifest_registration,persistence_receipt,github_repository}',to_jsonb('robzer13/not-authoritative'::text));
  b := jsonb_set(b,'{manifest_registration,storage_uri}',to_jsonb(
    'github://robzer13/not-authoritative@'||(b->'manifest_registration'->>'github_commit_sha')||'/'||(b->'manifest_registration'->>'github_path')
  ));
  perform pg_temp.bpi_expect_reject(b,'MANIFEST_SELF_CONSISTENT_WRONG_REPOSITORY');

  b := pg_temp.bpi_fixture('manifest-path');
  bad := 'integration/self-consistent-wrong-manifest.json';
  b := jsonb_set(b,'{manifest_registration,github_path}',to_jsonb(bad));
  b := jsonb_set(b,'{manifest_registration,persistence_receipt,github_path}',to_jsonb(bad));
  b := jsonb_set(b,'{manifest_registration,storage_uri}',to_jsonb(
    'github://'||(b->'manifest_registration'->>'github_repository')||'@'||(b->'manifest_registration'->>'github_commit_sha')||'/'||bad
  ));
  perform pg_temp.bpi_expect_reject(b,'MANIFEST_SELF_CONSISTENT_WRONG_PATH');

  b := pg_temp.bpi_fixture('manifest-commit');
  bad := repeat('e',40);
  b := jsonb_set(b,'{manifest_registration,github_commit_sha}',to_jsonb(bad));
  b := jsonb_set(b,'{manifest_registration,persistence_receipt,github_commit_sha}',to_jsonb(bad));
  b := jsonb_set(b,'{manifest_registration,storage_uri}',to_jsonb(
    'github://'||(b->'manifest_registration'->>'github_repository')||'@'||bad||'/'||(b->'manifest_registration'->>'github_path')
  ));
  perform pg_temp.bpi_expect_reject(b,'MANIFEST_SELF_CONSISTENT_WRONG_COMMIT');

  b := pg_temp.bpi_fixture('manifest-different-value');
  b := jsonb_set(b,'{manifest,contract_status_code}',to_jsonb('DIFFERENT_VALUE'::text));
  perform pg_temp.bpi_expect_reject(b,'MANIFEST_DIFFERENT_JSON_VALUE');

  b := pg_temp.bpi_fixture('manifest-array-order');
  first_ref := b->'manifest'->'output_artifacts'->0;
  second_ref := b->'manifest'->'output_artifacts'->1;
  b := jsonb_set(b,'{manifest,output_artifacts,0}',second_ref);
  b := jsonb_set(b,'{manifest,output_artifacts,1}',first_ref);
  perform pg_temp.bpi_expect_reject(b,'MANIFEST_ARRAY_ORDER');

  b := pg_temp.bpi_fixture('manifest-missing-field');
  b := jsonb_set(b,'{manifest}',(b->'manifest')-'open_material_limitations');
  perform pg_temp.bpi_expect_reject(b,'MANIFEST_MISSING_FIELD');

  b := pg_temp.bpi_fixture('manifest-extra-field');
  b := jsonb_set(b,'{manifest}',(b->'manifest')||'{"extra_probe":true}'::jsonb);
  perform pg_temp.bpi_expect_reject(b,'MANIFEST_EXTRA_FIELD');

  b := pg_temp.bpi_fixture('manifest-missing-receipt');
  b := jsonb_set(b,'{manifest_registration}',(b->'manifest_registration')-'persistence_receipt');
  perform pg_temp.bpi_expect_reject(b,'MANIFEST_MISSING_RECEIPT');
end;
$$;

-- CAS rejection remains before any authority transition.
do $$
declare
  b jsonb := pg_temp.bpi_fixture('stale-cas');
  r uuid := (b->>'run_id')::uuid;
  run_v bigint;
  stage_v bigint;
  rejected boolean := false;
begin
  select state_version into run_v from public.orotitan_runs where run_id=r;
  select state_version into stage_v from public.orotitan_run_stages where run_id=r and stage_code='INTEGRATION';
  begin
    perform public.finalize_orotitan_stage(
      r,'INTEGRATION',run_v+1,stage_v,
      b->'manifest',b->'manifest_registration',b->'output_artifacts',b->'edges',
      'bpi:stale-cas:'||r::text,repeat('9',64),'SYSTEM'
    );
  exception when others then rejected := true; end;
  if not rejected then raise exception 'BPI stale CAS accepted'; end if;
  if exists (select 1 from public.orotitan_artifacts where run_id=r and stage_code='INTEGRATION')
     or (select active_manifest_artifact_id from public.orotitan_run_stages where run_id=r and stage_code='INTEGRATION') is not null then
    raise exception 'BPI stale CAS caused partial authority transition';
  end if;
end;
$$;

select jsonb_build_object(
  'output_receipt_closure','PASS_ALL',
  'bundle_receipt_closure','PASS_ALL_8_OF_8',
  'output_wrong_sha','REJECT',
  'output_wrong_size','REJECT',
  'output_wrong_git_blob','REJECT',
  'output_artifact_id_mismatch','REJECT',
  'output_version_mismatch','REJECT',
  'output_manifest_reference_mismatch','REJECT',
  'wrong_repository','REJECT',
  'wrong_path','REJECT',
  'wrong_commit','REJECT',
  'path_blob_membership','REJECT',
  'semantic_formatting','PASS',
  'different_json_value','REJECT',
  'array_order','REJECT',
  'missing_extra_field','REJECT',
  'missing_one_output_receipt','REJECT_ATOMIC',
  'mixed_validity','REJECT_ZERO_PARTIAL_AUTHORITY',
  'manifest_attested_locator_authority','PASS',
  'visa_shaped_final_bundle','PASS_8_OF_8',
  'cas','PASS',
  'idempotency','PASS',
  'fingerprint_conflict','PASS',
  'result','PASS'
) as orotitan_bundle_persistence_integrity_regression;
