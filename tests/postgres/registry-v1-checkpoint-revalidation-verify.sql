\set ON_ERROR_STOP on

create or replace function pg_temp.reval_artifact_registration(
  p_artifact_id uuid,
  p_version integer,
  p_artifact_type text,
  p_hash_char text,
  p_path text
)
returns jsonb
language sql
as $$
  select jsonb_build_object(
    'artifact_id', p_artifact_id,
    'version', p_version,
    'artifact_type', p_artifact_type,
    'logical_name', lower(p_artifact_type),
    'authority_class', 'CHECKPOINT_STAGE_OUTPUT',
    'artifact_status', 'SEALED',
    'authority_state', 'CHECKPOINT',
    'availability_state', 'AVAILABLE',
    'media_type', 'application/json',
    'size_bytes', 2,
    'content_sha256', repeat(p_hash_char, 64),
    'storage_backend', 'SUPABASE_STORAGE',
    'storage_uri', 'supabase://orotitan-private/' || p_path,
    'supabase_bucket', 'orotitan-private',
    'supabase_object_path', p_path
  );
$$;

create or replace function pg_temp.reval_artifact_ref(p_registration jsonb)
returns jsonb
language sql
as $$
  select jsonb_build_object(
    'artifact_id', p_registration->>'artifact_id',
    'version', (p_registration->>'version')::integer,
    'artifact_type', p_registration->>'artifact_type',
    'content_sha256', p_registration->>'content_sha256',
    'authority_class', p_registration->>'authority_class',
    'media_type', p_registration->>'media_type',
    'size_bytes', (p_registration->>'size_bytes')::bigint
  );
$$;

create or replace function pg_temp.reval_manifest_registration(
  p_artifact_id uuid,
  p_version integer,
  p_manifest jsonb,
  p_path text
)
returns jsonb
language plpgsql
as $$
declare
  v_text text := p_manifest::text;
  v_bytes bytea := convert_to(v_text, 'UTF8');
  v_size bigint := octet_length(v_bytes);
  v_sha256 text := encode(extensions.digest(v_bytes, 'sha256'), 'hex');
  v_blob text;
  v_commit text := repeat('c', 40);
  v_repo text := 'robzer13/real-orotitan';
begin
  v_blob := encode(
    extensions.digest(
      convert_to('blob ' || v_size::text, 'UTF8') || decode('00','hex') || v_bytes,
      'sha1'
    ),
    'hex'
  );
  return jsonb_build_object(
    'artifact_id', p_artifact_id,
    'version', p_version,
    'artifact_type', 'DEEP_DIVE_STAGE_MANIFEST',
    'logical_name', 'deep_dive_stage_manifest',
    'authority_class', 'CHECKPOINT_STAGE_OUTPUT',
    'artifact_status', 'SEALED',
    'authority_state', 'CHECKPOINT',
    'availability_state', 'AVAILABLE',
    'media_type', 'application/json',
    'size_bytes', v_size,
    'content_sha256', v_sha256,
    'storage_backend', 'PRIVATE_GITHUB',
    'storage_uri', 'github://' || v_repo || '@' || v_commit || '/' || p_path,
    'github_repository', v_repo,
    'github_path', p_path,
    'github_commit_sha', v_commit,
    'github_blob_sha', v_blob,
    'persistence_receipt', jsonb_build_object(
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

create temporary table reval_manifest_receipts (
  run_id uuid not null,
  manifest_id uuid not null,
  manifest_version integer not null,
  registered_sha256 text not null,
  actual_sha256 text not null,
  receipt jsonb not null,
  primary key (run_id, manifest_id, manifest_version)
);

create or replace function pg_temp.reval_contract_pins()
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
    v_hash := encode(extensions.digest(convert_to(v_key, 'UTF8'), 'sha256'), 'hex');
    v_pins := v_pins || jsonb_build_object(
      v_key,
      jsonb_build_object(
        'name', v_name,
        'version', v_version,
        'content_sha256', v_hash,
        'locator', jsonb_build_object(
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

create or replace function pg_temp.reval_create_dd_run(p_tag text)
returns table(
  run_id uuid,
  issuer_id uuid,
  security_id uuid,
  dossier_id uuid,
  pins jsonb,
  contract_hash text
)
language plpgsql
as $$
declare
  v_issuer uuid;
  v_security uuid;
  v_dossier uuid;
  v_pins jsonb := pg_temp.reval_contract_pins();
  v_run uuid := gen_random_uuid();
  v_contract_hash text := public.orotitan_contract_set_sha256(v_pins);
begin
  select m.issuer_id, m.security_id, m.dossier_id
    into v_issuer, v_security, v_dossier
  from public.legacy_company_identity_map m
  order by m.legacy_company_id
  limit 1;

  insert into public.orotitan_runs (
    run_id, creation_idempotency_key, run_scope,
    issuer_id, security_id, dossier_id,
    entry_path, canonical_mode, run_type,
    run_status, current_stage, data_cutoff,
    process_version, pilotage_contract_version,
    contract_pins, contract_set_sha256, state_version
  ) values (
    v_run, 'checkpoint-revalidation-test:' || p_tag || ':' || v_run::text, 'COMPANY_ANALYSIS',
    v_issuer, v_security, v_dossier,
    'IMPOSED_COMPANY', 'ANALYZE', 'INITIAL',
    'ACTIVE', 'DEEP_DIVE', date '2026-09-20',
    v_pins->'process'->>'version', v_pins->'pilotage'->>'version',
    v_pins, v_contract_hash, 1
  );

  insert into public.orotitan_run_stages (
    run_id, stage_code, stage_revision,
    stage_contract_name, stage_contract_version, stage_contract_sha256,
    lifecycle_status, contract_status_code,
    handoff_gate_name, handoff_gate_state,
    state_version, started_at
  ) values (
    v_run, 'DEEP_DIVE', 1,
    v_pins->'deep_dive_stage'->>'name',
    v_pins->'deep_dive_stage'->>'version',
    v_pins->'deep_dive_stage'->>'content_sha256',
    'IN_PROGRESS', 'FUNDAMENTALS_IN_PROGRESS',
    'READY_FOR_INTEGRATION', 'NOT_EVALUATED',
    1, now()
  );

  run_id := v_run;
  issuer_id := v_issuer;
  security_id := v_security;
  dossier_id := v_dossier;
  pins := v_pins;
  contract_hash := v_contract_hash;
  return next;
end;
$$;

create or replace function pg_temp.reval_checkpoint(
  p_run_id uuid,
  p_issuer_id uuid,
  p_security_id uuid,
  p_dossier_id uuid,
  p_pins jsonb,
  p_contract_hash text,
  p_manifest_id uuid,
  p_manifest_version integer,
  p_output_regs jsonb,
  p_predecessor_version integer,
  p_idempotency_key text,
  p_fingerprint text
)
returns void
language plpgsql
as $$
declare
  v_refs jsonb;
  v_edges jsonb;
  v_manifest jsonb;
  v_manifest_reg jsonb;
  v_run_state bigint;
  v_stage_state bigint;
begin
  select jsonb_agg(pg_temp.reval_artifact_ref(value) order by ordinality)
    into v_refs
  from jsonb_array_elements(p_output_regs) with ordinality;

  select coalesce(jsonb_agg(
    jsonb_build_object(
      'child_run_id',p_run_id,
      'child_artifact_id',p_manifest_id,
      'child_version',p_manifest_version,
      'parent_run_id',p_run_id,
      'parent_artifact_id',value->>'artifact_id',
      'parent_version',(value->>'version')::integer,
      'relation_type','CONSUMES'
    ) order by ordinality
  ), '[]'::jsonb)
  into v_edges
  from jsonb_array_elements(v_refs) with ordinality;

  if p_predecessor_version is not null then
    v_edges := v_edges || jsonb_build_array(jsonb_build_object(
      'child_run_id',p_run_id,
      'child_artifact_id',p_manifest_id,
      'child_version',p_manifest_version,
      'parent_run_id',p_run_id,
      'parent_artifact_id',p_manifest_id,
      'parent_version',p_predecessor_version,
      'relation_type','SUPERSEDES'
    ));
  end if;

  v_manifest := jsonb_build_object(
    'manifest_schema_version','1.0.0',
    'manifest_id',p_manifest_id,
    'manifest_kind','CHECKPOINT',
    'run_id',p_run_id,
    'stage','DEEP_DIVE',
    'stage_revision',1,
    'issuer_id',p_issuer_id,
    'security_id',p_security_id,
    'dossier_id',p_dossier_id,
    'canonical_mode','ANALYZE',
    'run_type','INITIAL',
    'data_cutoff','2026-09-20',
    'baseline_snapshot_id',null,
    'process_version',p_pins->'process'->>'version',
    'pilotage_contract_version',p_pins->'pilotage'->>'version',
    'contract_pins',p_pins,
    'stage_contract',p_pins->'deep_dive_stage',
    'contract_set_sha256',p_contract_hash,
    'input_artifacts','[]'::jsonb,
    'output_artifacts',v_refs,
    'stage_status','IN_PROGRESS',
    'contract_status_code','FUNDAMENTALS_LOCKED_READY_FOR_VALUATION',
    'handoff_gate',jsonb_build_object('name','READY_FOR_INTEGRATION','state','NOT_EVALUATED'),
    'critical_blockers','[]'::jsonb,
    'open_material_limitations','[]'::jsonb,
    'parent_manifests',case when p_predecessor_version is null then '[]'::jsonb else jsonb_build_array(jsonb_build_object('manifest_id',p_manifest_id,'version',p_predecessor_version)) end,
    'started_at','2026-09-20T10:00:00Z',
    'completed_at',null
  );

  v_manifest_reg := pg_temp.reval_manifest_registration(
    p_manifest_id,
    p_manifest_version,
    v_manifest,
    'deep_dive/checkpoint-manifest-v' || p_manifest_version || '.json'
  );

  insert into reval_manifest_receipts(
    run_id,manifest_id,manifest_version,registered_sha256,actual_sha256,receipt
  ) values (
    p_run_id,p_manifest_id,p_manifest_version,
    v_manifest_reg->>'content_sha256',
    v_manifest_reg->>'content_sha256',
    v_manifest_reg->'persistence_receipt'
  )
  on conflict (run_id,manifest_id,manifest_version) do update
    set registered_sha256=excluded.registered_sha256,
        actual_sha256=excluded.actual_sha256,
        receipt=excluded.receipt;

  select state_version into v_run_state from public.orotitan_runs where run_id=p_run_id;
  select state_version into v_stage_state from public.orotitan_run_stages where run_id=p_run_id and stage_code='DEEP_DIVE';

  perform public.checkpoint_orotitan_stage(
    p_run_id,'DEEP_DIVE',v_run_state,v_stage_state,
    v_manifest,v_manifest_reg,p_output_regs,v_edges,
    'IN_PROGRESS',p_idempotency_key,p_fingerprint,'DEEP_DIVE_WORKER'
  );
end;
$$;

create or replace function pg_temp.revalidate_outputs(
  p_run_id uuid,
  p_stage_code text,
  p_expected_run_state_version bigint,
  p_expected_stage_state_version bigint,
  p_expected_manifest_artifact_id uuid,
  p_expected_manifest_version integer,
  p_expected_manifest_sha256 text,
  p_target_artifacts jsonb,
  p_idempotency_key text,
  p_request_fingerprint_sha256 text
)
returns jsonb
language plpgsql
as $$
declare
  v_actual_sha text;
  v_receipt jsonb;
begin
  select actual_sha256,receipt
    into v_actual_sha,v_receipt
  from reval_manifest_receipts
  where run_id=p_run_id
    and manifest_id=p_expected_manifest_artifact_id
    and manifest_version=p_expected_manifest_version;

  -- Negative tests for wrong run/stage/manifest must reach the canonical RPC,
  -- not fail in this test-only receipt adapter.
  if not found then
    select actual_sha256,receipt
      into v_actual_sha,v_receipt
    from reval_manifest_receipts
    order by manifest_version desc
    limit 1;
  end if;
  if not found then
    raise exception 'test manifest receipt missing';
  end if;
  return public.revalidate_orotitan_checkpoint_outputs(
    p_run_id,
    p_stage_code,
    p_expected_run_state_version,
    p_expected_stage_state_version,
    p_expected_manifest_artifact_id,
    p_expected_manifest_version,
    p_expected_manifest_sha256,
    v_actual_sha,
    v_receipt,
    p_target_artifacts,
    p_idempotency_key,
    p_request_fingerprint_sha256
  );
end;
$$;

create or replace function pg_temp.reval_output_refs(
  p_run_id uuid,
  p_manifest_id uuid,
  p_manifest_version integer
)
returns jsonb
language sql
as $$
  select coalesce(jsonb_agg(
    jsonb_build_object(
      'artifact_id',a.artifact_id,
      'version',a.version,
      'artifact_type',a.artifact_type,
      'content_sha256',a.content_sha256,
      'authority_class',a.authority_class,
      'media_type',a.media_type,
      'size_bytes',a.size_bytes
    ) order by a.artifact_id,a.version
  ), '[]'::jsonb)
  from public.orotitan_artifact_edges e
  join public.orotitan_artifacts a
    on a.run_id=e.parent_run_id and a.artifact_id=e.parent_artifact_id and a.version=e.parent_version
  where e.child_run_id=p_run_id
    and e.child_artifact_id=p_manifest_id
    and e.child_version=p_manifest_version
    and e.relation_type='CONSUMES'
    and a.stage_code='DEEP_DIVE'
    and a.authority_class='CHECKPOINT_STAGE_OUTPUT'
    and not (a.artifact_id=p_manifest_id and a.version=p_manifest_version);
$$;

create or replace function pg_temp.reval_prepare_defect(
  p_tag text,
  p_output_count integer default 2
)
returns table(
  run_id uuid,
  manifest_id uuid,
  output_refs jsonb,
  manifest_sha256 text
)
language plpgsql
as $$
declare
  v_run uuid;
  v_issuer uuid;
  v_security uuid;
  v_dossier uuid;
  v_pins jsonb;
  v_contract_hash text;
  v_manifest uuid := gen_random_uuid();
  v_regs jsonb := '[]'::jsonb;
  v_reg jsonb;
  v_i integer;
begin
  select r.run_id,r.issuer_id,r.security_id,r.dossier_id,r.pins,r.contract_hash
    into v_run,v_issuer,v_security,v_dossier,v_pins,v_contract_hash
  from pg_temp.reval_create_dd_run(p_tag) r;

  for v_i in 1..p_output_count loop
    v_reg := pg_temp.reval_artifact_registration(
      gen_random_uuid(),1,'TEST_OUTPUT_' || v_i,
      substr('123456789abcdef',v_i,1),
      'deep_dive/' || lower(p_tag) || '-output-' || v_i || '.json'
    );
    v_regs := v_regs || jsonb_build_array(v_reg);
  end loop;

  perform pg_temp.reval_checkpoint(
    v_run,v_issuer,v_security,v_dossier,v_pins,v_contract_hash,
    v_manifest,1,v_regs,null,
    'checkpoint:' || p_tag || ':v1',repeat('1',64)
  );
  perform pg_temp.reval_checkpoint(
    v_run,v_issuer,v_security,v_dossier,v_pins,v_contract_hash,
    v_manifest,2,v_regs,1,
    'checkpoint:' || p_tag || ':v2',repeat('2',64)
  );

  update public.orotitan_artifacts a
  set authority_state='SUPERSEDED',
      manifest_artifact_id=v_manifest,
      manifest_version=1
  from jsonb_array_elements(v_regs) r
  where a.run_id=v_run
    and a.artifact_id=(r->>'artifact_id')::uuid
    and a.version=(r->>'version')::integer;

  run_id := v_run;
  manifest_id := v_manifest;
  output_refs := pg_temp.reval_output_refs(v_run,v_manifest,2);
  select content_sha256 into manifest_sha256
  from public.orotitan_artifacts
  where artifact_id=v_manifest and version=2;
  return next;
end;
$$;

do $$
declare
  v_oid oid;
  v_signature text := 'revalidate_orotitan_checkpoint_outputs(uuid,text,bigint,bigint,uuid,integer,text,text,jsonb,jsonb,text,text)';
  v_prosecdef boolean;
  v_config text[];
begin
  v_oid := to_regprocedure('public.' || v_signature);
  if v_oid is null then raise exception 'revalidation RPC missing'; end if;
  select prosecdef,proconfig into v_prosecdef,v_config from pg_proc where oid=v_oid;
  if not v_prosecdef or not (v_config @> array['search_path=pg_catalog, public']::text[]) then
    raise exception 'revalidation RPC security-definer/search_path boundary invalid';
  end if;
  if has_function_privilege('public','public.' || v_signature,'EXECUTE')
     or has_function_privilege('anon','public.' || v_signature,'EXECUTE')
     or has_function_privilege('authenticated','public.' || v_signature,'EXECUTE')
     or not has_function_privilege('service_role','public.' || v_signature,'EXECUTE') then
    raise exception 'revalidation RPC privileges invalid';
  end if;
end $$;

-- A. predecessor checkpoint -> successor checkpoint with entirely new outputs.
do $$
declare
  v_run uuid; v_issuer uuid; v_security uuid; v_dossier uuid; v_pins jsonb; v_hash text;
  v_manifest uuid := gen_random_uuid(); v_old uuid := gen_random_uuid(); v_new uuid := gen_random_uuid();
  v_regs jsonb;
begin
  select r.run_id,r.issuer_id,r.security_id,r.dossier_id,r.pins,r.contract_hash
    into v_run,v_issuer,v_security,v_dossier,v_pins,v_hash from pg_temp.reval_create_dd_run('A') r;
  v_regs := jsonb_build_array(pg_temp.reval_artifact_registration(v_old,1,'A_OLD','1','deep_dive/a-old.json'));
  perform pg_temp.reval_checkpoint(v_run,v_issuer,v_security,v_dossier,v_pins,v_hash,v_manifest,1,v_regs,null,'A:v1',repeat('1',64));
  v_regs := jsonb_build_array(pg_temp.reval_artifact_registration(v_new,1,'A_NEW','2','deep_dive/a-new.json'));
  perform pg_temp.reval_checkpoint(v_run,v_issuer,v_security,v_dossier,v_pins,v_hash,v_manifest,2,v_regs,1,'A:v2',repeat('2',64));
  if not exists (select 1 from public.orotitan_artifacts where artifact_id=v_old and version=1 and authority_state='SUPERSEDED' and manifest_version=1) then raise exception 'A predecessor-only output not superseded'; end if;
  if not exists (select 1 from public.orotitan_artifacts where artifact_id=v_new and version=1 and authority_state='CHECKPOINT' and manifest_version=2) then raise exception 'A successor output not checkpoint'; end if;
end $$;

-- B + D. predecessor -> successor reusing all outputs; immutable mismatch rejects.
do $$
declare
  v_run uuid; v_issuer uuid; v_security uuid; v_dossier uuid; v_pins jsonb; v_hash text;
  v_manifest uuid := gen_random_uuid(); v_o1 uuid := gen_random_uuid(); v_o2 uuid := gen_random_uuid();
  v_regs jsonb; v_bad jsonb; v_rejected boolean := false;
begin
  select r.run_id,r.issuer_id,r.security_id,r.dossier_id,r.pins,r.contract_hash
    into v_run,v_issuer,v_security,v_dossier,v_pins,v_hash from pg_temp.reval_create_dd_run('B') r;
  v_regs := jsonb_build_array(
    pg_temp.reval_artifact_registration(v_o1,1,'B_ONE','3','deep_dive/b-one.json'),
    pg_temp.reval_artifact_registration(v_o2,1,'B_TWO','4','deep_dive/b-two.json')
  );
  perform pg_temp.reval_checkpoint(v_run,v_issuer,v_security,v_dossier,v_pins,v_hash,v_manifest,1,v_regs,null,'B:v1',repeat('3',64));
  v_bad := jsonb_set(v_regs,'{0,content_sha256}',to_jsonb(repeat('0',64)));
  begin
    perform pg_temp.reval_checkpoint(v_run,v_issuer,v_security,v_dossier,v_pins,v_hash,v_manifest,2,v_bad,1,'B:v2:bad',repeat('4',64));
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'D reused immutable hash mismatch accepted'; end if;
  perform pg_temp.reval_checkpoint(v_run,v_issuer,v_security,v_dossier,v_pins,v_hash,v_manifest,2,v_regs,1,'B:v2',repeat('5',64));
  if (select count(*) from public.orotitan_artifacts where artifact_id in (v_o1,v_o2) and authority_state='CHECKPOINT' and manifest_artifact_id=v_manifest and manifest_version=2) <> 2 then raise exception 'B reused outputs not preserved/rebound'; end if;
  if not exists (select 1 from public.orotitan_artifacts where artifact_id=v_manifest and version=1 and authority_state='SUPERSEDED') then raise exception 'B predecessor manifest not superseded'; end if;
end $$;

-- C. successor reuses a subset: intersection remains current; predecessor-only set is superseded.
do $$
declare
  v_run uuid; v_issuer uuid; v_security uuid; v_dossier uuid; v_pins jsonb; v_hash text;
  v_manifest uuid := gen_random_uuid(); v_keep uuid := gen_random_uuid(); v_drop uuid := gen_random_uuid(); v_add uuid := gen_random_uuid();
  v_regs1 jsonb; v_regs2 jsonb;
begin
  select r.run_id,r.issuer_id,r.security_id,r.dossier_id,r.pins,r.contract_hash
    into v_run,v_issuer,v_security,v_dossier,v_pins,v_hash from pg_temp.reval_create_dd_run('C') r;
  v_regs1 := jsonb_build_array(
    pg_temp.reval_artifact_registration(v_keep,1,'C_KEEP','5','deep_dive/c-keep.json'),
    pg_temp.reval_artifact_registration(v_drop,1,'C_DROP','6','deep_dive/c-drop.json')
  );
  perform pg_temp.reval_checkpoint(v_run,v_issuer,v_security,v_dossier,v_pins,v_hash,v_manifest,1,v_regs1,null,'C:v1',repeat('6',64));
  v_regs2 := jsonb_build_array(
    pg_temp.reval_artifact_registration(v_keep,1,'C_KEEP','5','deep_dive/c-keep.json'),
    pg_temp.reval_artifact_registration(v_add,1,'C_ADD','7','deep_dive/c-add.json')
  );
  perform pg_temp.reval_checkpoint(v_run,v_issuer,v_security,v_dossier,v_pins,v_hash,v_manifest,2,v_regs2,1,'C:v2',repeat('7',64));
  if not exists (select 1 from public.orotitan_artifacts where artifact_id=v_keep and authority_state='CHECKPOINT' and manifest_version=2) then raise exception 'C reused intersection not current'; end if;
  if not exists (select 1 from public.orotitan_artifacts where artifact_id=v_drop and authority_state='SUPERSEDED' and manifest_version=1) then raise exception 'C predecessor-only difference not superseded'; end if;
  if not exists (select 1 from public.orotitan_artifacts where artifact_id=v_add and authority_state='CHECKPOINT' and manifest_version=2) then raise exception 'C successor-only output not current'; end if;
end $$;

-- E-J + M-N-O. Repair admission, CAS, idempotency and history preservation.
do $$
declare
  v_run uuid; v_manifest uuid; v_refs jsonb; v_manifest_hash text;
  v_run_state bigint; v_stage_state bigint; v_edges_before bigint; v_events_before bigint;
  v_old_event uuid; v_extra jsonb; v_bad_refs jsonb; v_result jsonb; v_replay jsonb;
  v_rejected boolean;
begin
  select d.run_id,d.manifest_id,d.output_refs,d.manifest_sha256
    into v_run,v_manifest,v_refs,v_manifest_hash from pg_temp.reval_prepare_defect('M',2) d;
  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  select state_version into v_stage_state from public.orotitan_run_stages where run_id=v_run and stage_code='DEEP_DIVE';
  select count(*) into v_edges_before from public.orotitan_artifact_edges where child_run_id=v_run or parent_run_id=v_run;
  select count(*) into v_events_before from public.orotitan_run_events where run_id=v_run;
  select event_id into v_old_event from public.orotitan_run_events where run_id=v_run and event_type='STAGE_CHECKPOINTED' and payload->>'manifest_version'='1';

  -- E: artifact absent from successor manifest/output set.
  v_extra := pg_temp.reval_artifact_registration(gen_random_uuid(),1,'E_EXTRA','8','deep_dive/e-extra.json');
  perform public.orotitan_insert_artifact_registration(v_run,'DEEP_DIVE',v_extra,null,null);
  v_bad_refs := jsonb_build_array(v_refs->0,pg_temp.reval_artifact_ref(v_extra));
  v_rejected := false;
  begin
    perform pg_temp.revalidate_outputs(v_run,'DEEP_DIVE',v_run_state,v_stage_state,v_manifest,2,v_manifest_hash,v_bad_refs,'reval:E',repeat('8',64));
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'E absent successor artifact accepted'; end if;

  -- F: wrong run.
  v_rejected := false;
  begin
    perform pg_temp.revalidate_outputs(gen_random_uuid(),'DEEP_DIVE',v_run_state,v_stage_state,v_manifest,2,v_manifest_hash,v_refs,'reval:F',repeat('9',64));
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'F wrong run accepted'; end if;

  -- G: wrong stage.
  v_rejected := false;
  begin
    perform pg_temp.revalidate_outputs(v_run,'RESEARCH',v_run_state,v_stage_state,v_manifest,2,v_manifest_hash,v_refs,'reval:G',repeat('a',64));
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'G wrong stage accepted'; end if;

  -- H: wrong active manifest.
  v_rejected := false;
  begin
    perform pg_temp.revalidate_outputs(v_run,'DEEP_DIVE',v_run_state,v_stage_state,gen_random_uuid(),2,v_manifest_hash,v_refs,'reval:H',repeat('b',64));
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'H wrong active manifest accepted'; end if;

  -- I: stale run state version.
  v_rejected := false;
  begin
    perform pg_temp.revalidate_outputs(v_run,'DEEP_DIVE',v_run_state-1,v_stage_state,v_manifest,2,v_manifest_hash,v_refs,'reval:I',repeat('c',64));
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'I stale run state accepted'; end if;

  -- J: stale stage state version.
  v_rejected := false;
  begin
    perform pg_temp.revalidate_outputs(v_run,'DEEP_DIVE',v_run_state,v_stage_state-1,v_manifest,2,v_manifest_hash,v_refs,'reval:J',repeat('d',64));
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'J stale stage state accepted'; end if;

  -- M: exact legal repair succeeds once.
  v_result := pg_temp.revalidate_outputs(
    v_run,'DEEP_DIVE',v_run_state,v_stage_state,v_manifest,2,v_manifest_hash,v_refs,
    'reval:M',repeat('e',64)
  );
  if (v_result->>'idempotent_replay')::boolean or (v_result->>'artifact_count')::integer <> 2 then raise exception 'M first repair result invalid: %',v_result; end if;
  if (select count(*) from public.orotitan_artifacts a join jsonb_array_elements(v_refs) r on a.artifact_id=(r->>'artifact_id')::uuid and a.version=(r->>'version')::integer where a.authority_state='CHECKPOINT' and a.manifest_artifact_id=v_manifest and a.manifest_version=2) <> 2 then raise exception 'M repaired artifacts not current'; end if;
  if (select state_version from public.orotitan_runs where run_id=v_run) <> v_run_state+1 then raise exception 'M run state version not incremented exactly once'; end if;
  if (select state_version from public.orotitan_run_stages where run_id=v_run and stage_code='DEEP_DIVE') <> v_stage_state+1 then raise exception 'M stage state version not incremented exactly once'; end if;

  -- M replay: same request is idempotent even with original CAS values.
  v_replay := pg_temp.revalidate_outputs(
    v_run,'DEEP_DIVE',v_run_state,v_stage_state,v_manifest,2,v_manifest_hash,v_refs,
    'reval:M',repeat('e',64)
  );
  if not (v_replay->>'idempotent_replay')::boolean or v_replay->>'event_id' <> v_result->>'event_id' then raise exception 'M idempotent replay failed: %',v_replay; end if;
  if (select count(*) from public.orotitan_run_events where run_id=v_run and idempotency_key='reval:M') <> 1 then raise exception 'M duplicate repair event created'; end if;

  -- N: same idempotency key, different fingerprint fails.
  v_rejected := false;
  begin
    perform pg_temp.revalidate_outputs(v_run,'DEEP_DIVE',v_run_state,v_stage_state,v_manifest,2,v_manifest_hash,v_refs,'reval:M',repeat('f',64));
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'N conflicting idempotency fingerprint accepted'; end if;

  -- O: prior manifest/event history and lineage are preserved.
  if not exists (select 1 from public.orotitan_artifacts where artifact_id=v_manifest and version=1 and authority_state='SUPERSEDED') then raise exception 'O predecessor manifest history lost'; end if;
  if not exists (select 1 from public.orotitan_run_events where event_id=v_old_event and event_type='STAGE_CHECKPOINTED') then raise exception 'O predecessor checkpoint event history lost'; end if;
  if (select count(*) from public.orotitan_artifact_edges where child_run_id=v_run or parent_run_id=v_run) <> v_edges_before then raise exception 'O lineage changed during metadata repair'; end if;
  if (select count(*) from public.orotitan_run_events where run_id=v_run) <> v_events_before+1 then raise exception 'O unexpected event count after repair'; end if;
end $$;

-- K. INVALIDATED artifact fails closed.
do $$
declare
  v_run uuid; v_manifest uuid; v_refs jsonb; v_manifest_hash text; v_run_state bigint; v_stage_state bigint; v_rejected boolean := false;
begin
  select d.run_id,d.manifest_id,d.output_refs,d.manifest_sha256 into v_run,v_manifest,v_refs,v_manifest_hash from pg_temp.reval_prepare_defect('K',1) d;
  update public.orotitan_artifacts set artifact_status='INVALIDATED' where artifact_id=(v_refs->0->>'artifact_id')::uuid and version=(v_refs->0->>'version')::integer;
  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  select state_version into v_stage_state from public.orotitan_run_stages where run_id=v_run and stage_code='DEEP_DIVE';
  begin
    perform pg_temp.revalidate_outputs(v_run,'DEEP_DIVE',v_run_state,v_stage_state,v_manifest,2,v_manifest_hash,v_refs,'reval:K',repeat('1',64));
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'K INVALIDATED target accepted'; end if;
end $$;

-- L. NON_AUTHORITATIVE artifact fails closed.
do $$
declare
  v_run uuid; v_manifest uuid; v_refs jsonb; v_manifest_hash text; v_run_state bigint; v_stage_state bigint; v_rejected boolean := false;
begin
  select d.run_id,d.manifest_id,d.output_refs,d.manifest_sha256 into v_run,v_manifest,v_refs,v_manifest_hash from pg_temp.reval_prepare_defect('L',1) d;
  update public.orotitan_artifacts set authority_state='NON_AUTHORITATIVE' where artifact_id=(v_refs->0->>'artifact_id')::uuid and version=(v_refs->0->>'version')::integer;
  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  select state_version into v_stage_state from public.orotitan_run_stages where run_id=v_run and stage_code='DEEP_DIVE';
  begin
    perform pg_temp.revalidate_outputs(v_run,'DEEP_DIVE',v_run_state,v_stage_state,v_manifest,2,v_manifest_hash,v_refs,'reval:L',repeat('2',64));
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'L NON_AUTHORITATIVE target accepted'; end if;
end $$;


-- P. Exact existing ID/version with size mismatch fails closed.
do $$
declare
  v_run uuid; v_manifest uuid; v_refs jsonb; v_manifest_hash text;
  v_run_state bigint; v_stage_state bigint; v_bad_refs jsonb; v_rejected boolean := false;
begin
  select d.run_id,d.manifest_id,d.output_refs,d.manifest_sha256
    into v_run,v_manifest,v_refs,v_manifest_hash
  from pg_temp.reval_prepare_defect('P',1) d;
  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  select state_version into v_stage_state from public.orotitan_run_stages where run_id=v_run and stage_code='DEEP_DIVE';
  v_bad_refs := jsonb_set(v_refs,'{0,size_bytes}',to_jsonb(((v_refs->0->>'size_bytes')::bigint + 1)));
  begin
    perform pg_temp.revalidate_outputs(
      v_run,'DEEP_DIVE',v_run_state,v_stage_state,v_manifest,2,v_manifest_hash,v_bad_refs,
      'reval:P',repeat('3',64)
    );
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'P exact reused artifact size mismatch accepted'; end if;
end $$;

-- Q. Unavailable artifact fails closed.
do $$
declare
  v_run uuid; v_manifest uuid; v_refs jsonb; v_manifest_hash text;
  v_run_state bigint; v_stage_state bigint; v_rejected boolean := false;
begin
  select d.run_id,d.manifest_id,d.output_refs,d.manifest_sha256
    into v_run,v_manifest,v_refs,v_manifest_hash
  from pg_temp.reval_prepare_defect('Q',1) d;
  update public.orotitan_artifacts
     set availability_state='MISSING'
   where artifact_id=(v_refs->0->>'artifact_id')::uuid
     and version=(v_refs->0->>'version')::integer;
  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  select state_version into v_stage_state from public.orotitan_run_stages where run_id=v_run and stage_code='DEEP_DIVE';
  begin
    perform pg_temp.revalidate_outputs(
      v_run,'DEEP_DIVE',v_run_state,v_stage_state,v_manifest,2,v_manifest_hash,v_refs,
      'reval:Q',repeat('4',64)
    );
  exception when others then v_rejected := true; end;
  if not v_rejected then raise exception 'Q unavailable target accepted'; end if;
end $$;

-- R. TotalEnergies defect-shape regression:
-- predecessor CHECKPOINT -> successor CHECKPOINT, 6 intended successor outputs,
-- 4 exact reused predecessor outputs + 2 new/current outputs.
do $
declare
  v_run uuid; v_issuer uuid; v_security uuid; v_dossier uuid; v_pins jsonb; v_hash text;
  v_manifest uuid := gen_random_uuid();
  r1 jsonb; r2 jsonb; r3 jsonb; r4 jsonb;
  drop1 jsonb; drop2 jsonb; new1 jsonb; new2 jsonb;
  regs1 jsonb; regs2 jsonb; v_refs jsonb; v_manifest_hash text;
  v_run_state bigint; v_stage_state bigint;
begin
  select x.run_id,x.issuer_id,x.security_id,x.dossier_id,x.pins,x.contract_hash
    into v_run,v_issuer,v_security,v_dossier,v_pins,v_hash
  from pg_temp.reval_create_dd_run('TOTALENERGIES_SHAPE') x;

  r1 := pg_temp.reval_artifact_registration(gen_random_uuid(),1,'EVIDENCE_LEDGER','1','deep_dive/te/evidence-ledger.json');
  r2 := pg_temp.reval_artifact_registration(gen_random_uuid(),1,'CALCULATION_LEDGER','2','deep_dive/te/calculation-ledger.json');
  r3 := pg_temp.reval_artifact_registration(gen_random_uuid(),1,'MATERIAL_ASSUMPTION_REGISTER','3','deep_dive/te/material-assumptions.json');
  r4 := pg_temp.reval_artifact_registration(gen_random_uuid(),1,'VALUATION_ARTIFACT','4','deep_dive/te/valuation-artifact.json');
  drop1 := pg_temp.reval_artifact_registration(gen_random_uuid(),1,'PREDECESSOR_ONLY_A','5','deep_dive/te/predecessor-only-a.json');
  drop2 := pg_temp.reval_artifact_registration(gen_random_uuid(),1,'PREDECESSOR_ONLY_B','6','deep_dive/te/predecessor-only-b.json');
  regs1 := jsonb_build_array(r1,r2,r3,r4,drop1,drop2);

  perform pg_temp.reval_checkpoint(
    v_run,v_issuer,v_security,v_dossier,v_pins,v_hash,
    v_manifest,1,regs1,null,
    'te-shape:m1',repeat('1',64)
  );

  new1 := pg_temp.reval_artifact_registration(gen_random_uuid(),1,'VALUATION_LOCK','7','deep_dive/te/valuation-lock.json');
  new2 := pg_temp.reval_artifact_registration(gen_random_uuid(),1,'VALUATION_FULL_PRECISION_INPUTS','8','deep_dive/te/full-precision-inputs.json');
  regs2 := jsonb_build_array(r1,r2,r3,r4,new1,new2);

  perform pg_temp.reval_checkpoint(
    v_run,v_issuer,v_security,v_dossier,v_pins,v_hash,
    v_manifest,2,regs2,1,
    'te-shape:m2',repeat('2',64)
  );

  if (select count(*) from public.orotitan_artifacts a
      join jsonb_array_elements(regs2) x
        on a.artifact_id=(x->>'artifact_id')::uuid and a.version=(x->>'version')::integer
      where a.run_id=v_run and a.authority_state='CHECKPOINT'
        and a.manifest_artifact_id=v_manifest and a.manifest_version=2) <> 6 then
    raise exception 'TOTALENERGIES shape canonical successor did not bind 6/6';
  end if;

  -- Reproduce the historical defect: only the four exact reused outputs become
  -- predecessor-bound/superseded while the two new outputs remain current.
  update public.orotitan_artifacts
     set authority_state='SUPERSEDED',
         manifest_artifact_id=v_manifest,
         manifest_version=1
   where run_id=v_run
     and artifact_id in (
       (r1->>'artifact_id')::uuid,(r2->>'artifact_id')::uuid,
       (r3->>'artifact_id')::uuid,(r4->>'artifact_id')::uuid
     );

  v_refs := pg_temp.reval_output_refs(v_run,v_manifest,2);
  if jsonb_array_length(v_refs) <> 6 then
    raise exception 'TOTALENERGIES shape active manifest output refs are not exactly 6';
  end if;

  select content_sha256 into v_manifest_hash
  from public.orotitan_artifacts
  where artifact_id=v_manifest and version=2;
  select state_version into v_run_state
  from public.orotitan_runs where run_id=v_run;
  select state_version into v_stage_state
  from public.orotitan_run_stages where run_id=v_run and stage_code='DEEP_DIVE';

  perform pg_temp.revalidate_outputs(
    v_run,'DEEP_DIVE',v_run_state,v_stage_state,
    v_manifest,2,v_manifest_hash,v_refs,
    'te-shape:repair',repeat('3',64)
  );

  if (select count(*) from public.orotitan_artifacts a
      join jsonb_array_elements(v_refs) x
        on a.artifact_id=(x->>'artifact_id')::uuid and a.version=(x->>'version')::integer
      where a.run_id=v_run
        and a.authority_state='CHECKPOINT'
        and a.manifest_artifact_id=v_manifest
        and a.manifest_version=2) <> 6 then
    raise exception 'TOTALENERGIES shape repair authority closure is not 6/6';
  end if;

  if exists (
    select 1 from public.orotitan_artifacts a
    join jsonb_array_elements(v_refs) x
      on a.artifact_id=(x->>'artifact_id')::uuid and a.version=(x->>'version')::integer
    where a.authority_state='SUPERSEDED'
  ) then
    raise exception 'TOTALENERGIES shape intended successor output remains SUPERSEDED';
  end if;

  if (select count(*) from public.orotitan_artifacts
      where artifact_id in ((drop1->>'artifact_id')::uuid,(drop2->>'artifact_id')::uuid)
        and authority_state='SUPERSEDED'
        and manifest_artifact_id=v_manifest
        and manifest_version=1) <> 2 then
    raise exception 'TOTALENERGIES shape predecessor-only outputs not superseded';
  end if;

  if (select count(*) from public.orotitan_artifact_edges
      where child_run_id=v_run and child_artifact_id=v_manifest and child_version=2
        and relation_type='CONSUMES') <> 6 then
    raise exception 'TOTALENERGIES shape successor manifest refs not exact';
  end if;
end $;

select jsonb_build_object(
  'A_new_outputs','PASS',
  'B_reuse_all','PASS',
  'C_reuse_subset','PASS',
  'D_hash_mismatch_reject','PASS',
  'E_absent_successor_reject','PASS',
  'F_wrong_run_reject','PASS',
  'G_wrong_stage_reject','PASS',
  'H_wrong_manifest_reject','PASS',
  'I_stale_run_version_reject','PASS',
  'J_stale_stage_version_reject','PASS',
  'K_invalidated_reject','PASS',
  'L_non_authoritative_reject','PASS',
  'M_idempotent_replay','PASS',
  'N_idempotency_conflict_reject','PASS',
  'O_history_preserved','PASS',
  'P_size_mismatch_reject','PASS',
  'Q_unavailable_reject','PASS',
  'R_totalenergies_shape','PASS',
  'critical_invariant','PASS',
  'result','PASS'
) as orotitan_checkpoint_revalidation_regression;
