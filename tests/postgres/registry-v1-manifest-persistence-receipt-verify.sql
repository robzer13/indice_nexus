\set ON_ERROR_STOP on

-- OroTitan Registry V1.7 manifest persistence-receipt regression.
-- Covers exact-byte SHA/size/Git-blob verification and atomic rejection.

create or replace function pg_temp.mpr_contract_pins()
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
    v_hash := encode(extensions.digest(convert_to(v_key,'UTF8'),'sha256'),'hex');
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

create or replace function pg_temp.mpr_output_registration(
  p_artifact_id uuid,
  p_hash_char text,
  p_path text
)
returns jsonb
language sql
as $$
  select jsonb_build_object(
    'artifact_id',p_artifact_id,
    'version',1,
    'artifact_type','MPR_TEST_OUTPUT',
    'logical_name','mpr_test_output',
    'authority_class','CHECKPOINT_STAGE_OUTPUT',
    'artifact_status','SEALED',
    'authority_state','CHECKPOINT',
    'availability_state','AVAILABLE',
    'media_type','application/json',
    'size_bytes',2,
    'content_sha256',repeat(p_hash_char,64),
    'storage_backend','SUPABASE_STORAGE',
    'storage_uri','supabase://orotitan-private/' || p_path,
    'supabase_bucket','orotitan-private',
    'supabase_object_path',p_path
  );
$$;

create or replace function pg_temp.mpr_ref(p_registration jsonb)
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
    'size_bytes',(p_registration->>'size_bytes')::bigint
  );
$$;

create or replace function pg_temp.mpr_manifest_registration(
  p_manifest_id uuid,
  p_version integer,
  p_manifest jsonb,
  p_path text
)
returns jsonb
language plpgsql
as $$
declare
  v_text text := p_manifest::text;
  v_bytes bytea := convert_to(v_text,'UTF8');
  v_size bigint := octet_length(v_bytes);
  v_sha text := encode(extensions.digest(v_bytes,'sha256'),'hex');
  v_blob text;
  v_commit text := repeat('c',40);
  v_repo text := 'robzer13/real-orotitan';
begin
  v_blob := encode(
    extensions.digest(
      convert_to('blob ' || v_size::text,'UTF8') || decode('00','hex') || v_bytes,
      'sha1'
    ),
    'hex'
  );
  return jsonb_build_object(
    'artifact_id',p_manifest_id,
    'version',p_version,
    'artifact_type','DEEP_DIVE_STAGE_MANIFEST',
    'logical_name','deep_dive_stage_manifest',
    'authority_class','CHECKPOINT_STAGE_OUTPUT',
    'artifact_status','SEALED',
    'authority_state','CHECKPOINT',
    'availability_state','AVAILABLE',
    'media_type','application/json',
    'size_bytes',v_size,
    'content_sha256',v_sha,
    'storage_backend','PRIVATE_GITHUB',
    'storage_uri','github://' || v_repo || '@' || v_commit || '/' || p_path,
    'github_repository',v_repo,
    'github_path',p_path,
    'github_commit_sha',v_commit,
    'github_blob_sha',v_blob,
    'persistence_receipt',jsonb_build_object(
      'receipt_schema_version','1.0',
      'verification_method','PRIVATE_GITHUB_REREAD_EXACT_BYTES_V1',
      'storage_backend','PRIVATE_GITHUB',
      'github_repository',v_repo,
      'github_path',p_path,
      'github_commit_sha',v_commit,
      'github_blob_sha',v_blob,
      'commit_path_resolved',true,
      'verified_content_base64',encode(v_bytes,'base64'),
      'verified_at','2026-09-20T12:00:00Z'
    )
  );
end;
$$;

do $$
declare
  v_issuer uuid;
  v_security uuid;
  v_dossier uuid;
  v_run uuid := gen_random_uuid();
  v_manifest_id uuid := gen_random_uuid();
  v_output_id uuid := gen_random_uuid();
  v_pins jsonb := pg_temp.mpr_contract_pins();
  v_contract_hash text;
  v_output jsonb;
  v_outputs jsonb;
  v_refs jsonb;
  v_manifest jsonb;
  v_reg jsonb;
  v_bad jsonb;
  v_run_state bigint;
  v_stage_state bigint;
  v_events_before bigint;
  v_artifacts_before bigint;
  v_rejected boolean;
begin
  select m.issuer_id,m.security_id,m.dossier_id
    into v_issuer,v_security,v_dossier
  from public.legacy_company_identity_map m
  order by m.legacy_company_id
  limit 1;

  if v_issuer is null or v_security is null or v_dossier is null then
    raise exception 'MPR identity fixture unavailable';
  end if;

  v_contract_hash := public.orotitan_contract_set_sha256(v_pins);

  insert into public.orotitan_runs(
    run_id,creation_idempotency_key,run_scope,
    issuer_id,security_id,dossier_id,
    entry_path,canonical_mode,run_type,
    run_status,current_stage,data_cutoff,
    process_version,pilotage_contract_version,
    contract_pins,contract_set_sha256,state_version
  ) values (
    v_run,'mpr:'||v_run::text,'COMPANY_ANALYSIS',
    v_issuer,v_security,v_dossier,
    'IMPOSED_COMPANY','ANALYZE','INITIAL',
    'ACTIVE','DEEP_DIVE',date '2026-09-20',
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
    v_run,'DEEP_DIVE',1,
    v_pins->'deep_dive_stage'->>'name',
    v_pins->'deep_dive_stage'->>'version',
    v_pins->'deep_dive_stage'->>'content_sha256',
    'IN_PROGRESS','VALUATION_LOCKED_READY_FOR_CERTIFICATION',
    'READY_FOR_INTEGRATION','NOT_EVALUATED',
    1,now()
  );

  v_output := pg_temp.mpr_output_registration(
    v_output_id,'1','deep_dive/mpr-output.json'
  );
  v_outputs := jsonb_build_array(v_output);
  v_refs := jsonb_build_array(pg_temp.mpr_ref(v_output));

  v_manifest := jsonb_build_object(
    'manifest_schema_version','1.0.0',
    'manifest_id',v_manifest_id,
    'manifest_kind','CHECKPOINT',
    'run_id',v_run,
    'stage','DEEP_DIVE',
    'stage_revision',1,
    'issuer_id',v_issuer,
    'security_id',v_security,
    'dossier_id',v_dossier,
    'canonical_mode','ANALYZE',
    'run_type','INITIAL',
    'data_cutoff','2026-09-20',
    'baseline_snapshot_id',null,
    'process_version',v_pins->'process'->>'version',
    'pilotage_contract_version',v_pins->'pilotage'->>'version',
    'contract_pins',v_pins,
    'stage_contract',v_pins->'deep_dive_stage',
    'contract_set_sha256',v_contract_hash,
    'input_artifacts','[]'::jsonb,
    'output_artifacts',v_refs,
    'stage_status','IN_PROGRESS',
    'contract_status_code','VALUATION_LOCKED_READY_FOR_CERTIFICATION',
    'handoff_gate',jsonb_build_object('name','READY_FOR_INTEGRATION','state','NOT_EVALUATED'),
    'critical_blockers','[]'::jsonb,
    'open_material_limitations','[]'::jsonb,
    'parent_manifests','[]'::jsonb,
    'started_at','2026-09-20T12:00:00Z',
    'completed_at',null
  );
  v_reg := pg_temp.mpr_manifest_registration(
    v_manifest_id,1,v_manifest,'deep_dive/mpr-manifest.json'
  );

  select state_version into v_run_state
  from public.orotitan_runs where run_id=v_run;
  select state_version into v_stage_state
  from public.orotitan_run_stages where run_id=v_run and stage_code='DEEP_DIVE';
  select count(*) into v_events_before
  from public.orotitan_run_events where run_id=v_run;
  select count(*) into v_artifacts_before
  from public.orotitan_artifacts where run_id=v_run;

  -- TEST 16: supplied content SHA differs from verified persisted bytes.
  v_bad := jsonb_set(v_reg,'{content_sha256}',to_jsonb(repeat('0',64)));
  v_rejected := false;
  begin
    perform public.checkpoint_orotitan_stage(
      v_run,'DEEP_DIVE',v_run_state,v_stage_state,
      v_manifest,v_bad,v_outputs,'[]'::jsonb,
      'IN_PROGRESS','mpr:bad-sha',repeat('1',64),'DEEP_DIVE_WORKER'
    );
  exception when others then
    v_rejected := true;
  end;
  if not v_rejected then raise exception 'TEST16 bad manifest SHA accepted'; end if;
  if (select state_version from public.orotitan_runs where run_id=v_run) <> v_run_state
     or (select state_version from public.orotitan_run_stages where run_id=v_run and stage_code='DEEP_DIVE') <> v_stage_state
     or (select count(*) from public.orotitan_run_events where run_id=v_run) <> v_events_before
     or (select count(*) from public.orotitan_artifacts where run_id=v_run) <> v_artifacts_before
     or (select active_manifest_artifact_id from public.orotitan_run_stages where run_id=v_run and stage_code='DEEP_DIVE') is not null then
    raise exception 'TEST16 atomic rejection violated';
  end if;

  -- TEST 17: supplied size differs from verified persisted bytes.
  v_bad := jsonb_set(v_reg,'{size_bytes}',to_jsonb(((v_reg->>'size_bytes')::bigint + 1)));
  v_rejected := false;
  begin
    perform public.checkpoint_orotitan_stage(
      v_run,'DEEP_DIVE',v_run_state,v_stage_state,
      v_manifest,v_bad,v_outputs,'[]'::jsonb,
      'IN_PROGRESS','mpr:bad-size',repeat('2',64),'DEEP_DIVE_WORKER'
    );
  exception when others then
    v_rejected := true;
  end;
  if not v_rejected then raise exception 'TEST17 bad manifest size accepted'; end if;
  if (select state_version from public.orotitan_runs where run_id=v_run) <> v_run_state
     or (select state_version from public.orotitan_run_stages where run_id=v_run and stage_code='DEEP_DIVE') <> v_stage_state
     or (select count(*) from public.orotitan_run_events where run_id=v_run) <> v_events_before
     or (select count(*) from public.orotitan_artifacts where run_id=v_run) <> v_artifacts_before then
    raise exception 'TEST17 atomic rejection violated';
  end if;

  -- Git blob mismatch also fails atomically.
  v_bad := jsonb_set(
    jsonb_set(v_reg,'{github_blob_sha}',to_jsonb(repeat('0',40))),
    '{persistence_receipt,github_blob_sha}',to_jsonb(repeat('0',40))
  );
  v_rejected := false;
  begin
    perform public.checkpoint_orotitan_stage(
      v_run,'DEEP_DIVE',v_run_state,v_stage_state,
      v_manifest,v_bad,v_outputs,'[]'::jsonb,
      'IN_PROGRESS','mpr:bad-blob',repeat('3',64),'DEEP_DIVE_WORKER'
    );
  exception when others then
    v_rejected := true;
  end;
  if not v_rejected then raise exception 'bad Git blob accepted'; end if;

  -- Missing receipt cannot reach activation.
  v_bad := v_reg - 'persistence_receipt';
  v_rejected := false;
  begin
    perform public.checkpoint_orotitan_stage(
      v_run,'DEEP_DIVE',v_run_state,v_stage_state,
      v_manifest,v_bad,v_outputs,'[]'::jsonb,
      'IN_PROGRESS','mpr:no-receipt',repeat('4',64),'DEEP_DIVE_WORKER'
    );
  exception when others then
    v_rejected := true;
  end;
  if not v_rejected then raise exception 'missing persistence receipt accepted'; end if;

  -- NULL discriminator fields must fail closed (SQL NULL must never bypass validation).
  v_bad := jsonb_set(v_reg,'{persistence_receipt,receipt_schema_version}','null'::jsonb);
  v_rejected := false;
  begin
    perform public.checkpoint_orotitan_stage(
      v_run,'DEEP_DIVE',v_run_state,v_stage_state,
      v_manifest,v_bad,v_outputs,'[]'::jsonb,
      'IN_PROGRESS','mpr:null-schema-version',repeat('6',64),'DEEP_DIVE_WORKER'
    );
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'null receipt_schema_version accepted'; end if;

  v_bad := jsonb_set(v_reg,'{persistence_receipt,verification_method}','null'::jsonb);
  v_rejected := false;
  begin
    perform public.checkpoint_orotitan_stage(
      v_run,'DEEP_DIVE',v_run_state,v_stage_state,
      v_manifest,v_bad,v_outputs,'[]'::jsonb,
      'IN_PROGRESS','mpr:null-method',repeat('7',64),'DEEP_DIVE_WORKER'
    );
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'null verification_method accepted'; end if;

  v_bad := jsonb_set(v_reg,'{persistence_receipt,storage_backend}','null'::jsonb);
  v_rejected := false;
  begin
    perform public.checkpoint_orotitan_stage(
      v_run,'DEEP_DIVE',v_run_state,v_stage_state,
      v_manifest,v_bad,v_outputs,'[]'::jsonb,
      'IN_PROGRESS','mpr:null-backend',repeat('8',64),'DEEP_DIVE_WORKER'
    );
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'null receipt storage_backend accepted'; end if;

  -- Wrong commit/path relationship must fail closed.
  v_bad := jsonb_set(v_reg,'{persistence_receipt,github_path}',to_jsonb('deep_dive/not-the-persisted-manifest.json'::text));
  v_rejected := false;
  begin
    perform public.checkpoint_orotitan_stage(
      v_run,'DEEP_DIVE',v_run_state,v_stage_state,
      v_manifest,v_bad,v_outputs,'[]'::jsonb,
      'IN_PROGRESS','mpr:wrong-path',repeat('9',64),'DEEP_DIVE_WORKER'
    );
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'wrong receipt path accepted'; end if;

  v_bad := jsonb_set(v_reg,'{persistence_receipt,github_commit_sha}',to_jsonb(repeat('d',40)));
  v_rejected := false;
  begin
    perform public.checkpoint_orotitan_stage(
      v_run,'DEEP_DIVE',v_run_state,v_stage_state,
      v_manifest,v_bad,v_outputs,'[]'::jsonb,
      'IN_PROGRESS','mpr:wrong-commit',repeat('a',64),'DEEP_DIVE_WORKER'
    );
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'wrong receipt commit accepted'; end if;

  -- Exact receipt succeeds.
  perform public.checkpoint_orotitan_stage(
    v_run,'DEEP_DIVE',v_run_state,v_stage_state,
    v_manifest,v_reg,v_outputs,'[]'::jsonb,
    'IN_PROGRESS','mpr:valid',repeat('5',64),'DEEP_DIVE_WORKER'
  );

  if not exists (
    select 1 from public.orotitan_run_stages
    where run_id=v_run and stage_code='DEEP_DIVE'
      and active_manifest_artifact_id=v_manifest_id
      and active_manifest_version=1
      and active_manifest_kind='CHECKPOINT'
  ) then
    raise exception 'valid verified manifest was not activated';
  end if;
  if not exists (
    select 1 from public.orotitan_artifacts
    where run_id=v_run and artifact_id=v_manifest_id and version=1
      and content_sha256=v_reg->>'content_sha256'
      and size_bytes=(v_reg->>'size_bytes')::bigint
      and github_blob_sha=v_reg->>'github_blob_sha'
  ) then
    raise exception 'valid verified manifest Registry row mismatch';
  end if;
end;
$$;

select jsonb_build_object(
  'manifest_verified_sha256_reject','PASS',
  'manifest_verified_size_reject','PASS',
  'manifest_verified_git_blob_reject','PASS',
  'missing_receipt_reject','PASS',
  'null_discriminator_reject','PASS',
  'wrong_path_reject','PASS',
  'wrong_commit_reject','PASS',
  'atomic_rejection','PASS',
  'valid_exact_receipt_activation','PASS',
  'result','PASS'
) as orotitan_manifest_persistence_receipt_regression;
