\set ON_ERROR_STOP on

-- OroTitan Registry V1.7 successor-output rebinding regression matrix.
-- Covers TEST 1 through TEST 20 from the governed recovery plan.

create or replace function pg_temp.v17_artifact(
  p_artifact_id uuid,
  p_version integer,
  p_artifact_type text,
  p_hash_char text,
  p_path text,
  p_authority_class text default 'CHECKPOINT_STAGE_OUTPUT',
  p_authority_state text default 'CHECKPOINT',
  p_artifact_status text default 'SEALED',
  p_availability_state text default 'AVAILABLE'
)
returns jsonb
language sql
as $$
  select jsonb_build_object(
    'artifact_id',p_artifact_id,
    'version',p_version,
    'artifact_type',p_artifact_type,
    'logical_name',lower(p_artifact_type),
    'authority_class',p_authority_class,
    'artifact_status',p_artifact_status,
    'authority_state',p_authority_state,
    'availability_state',p_availability_state,
    'media_type','application/json',
    'size_bytes',2,
    'content_sha256',repeat(p_hash_char,64),
    'storage_backend','SUPABASE_STORAGE',
    'storage_uri','supabase://orotitan-private/' || p_path,
    'supabase_bucket','orotitan-private',
    'supabase_object_path',p_path
  );
$$;

create or replace function pg_temp.v17_ref(p_registration jsonb)
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

create or replace function pg_temp.v17_manifest_registration(
  p_manifest jsonb,
  p_manifest_id uuid,
  p_version integer,
  p_artifact_type text,
  p_path text,
  p_authority_class text,
  p_authority_state text
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
    'artifact_type',p_artifact_type,
    'logical_name',lower(p_artifact_type),
    'authority_class',p_authority_class,
    'artifact_status','SEALED',
    'authority_state',p_authority_state,
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

create or replace function pg_temp.v17_pins()
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
    v_hash := encode(extensions.digest(convert_to('v17:' || v_key,'UTF8'),'sha256'),'hex');
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

create or replace function pg_temp.v17_create_run(p_tag text, p_stage text)
returns table(
  run_id uuid,
  issuer_id uuid,
  security_id uuid,
  dossier_id uuid
)
language plpgsql
as $$
declare
  v_run uuid := gen_random_uuid();
  v_issuer uuid;
  v_security uuid;
  v_dossier uuid;
  v_pins jsonb := pg_temp.v17_pins();
  v_contract_hash text := public.orotitan_contract_set_sha256(v_pins);
  v_stage_key text := case p_stage
    when 'RESEARCH' then 'research_stage'
    when 'DEEP_DIVE' then 'deep_dive_stage'
    when 'INTEGRATION' then 'integration_stage'
    else null
  end;
  v_gate text := case p_stage
    when 'RESEARCH' then 'READY_FOR_DEEP_DIVE'
    when 'DEEP_DIVE' then 'READY_FOR_INTEGRATION'
    when 'INTEGRATION' then 'READY_TO_PUBLISH'
  end;
begin
  if v_stage_key is null then raise exception 'invalid test stage'; end if;

  select m.issuer_id,m.security_id,m.dossier_id
    into v_issuer,v_security,v_dossier
  from public.legacy_company_identity_map m
  order by m.legacy_company_id
  limit 1;

  insert into public.orotitan_runs(
    run_id,creation_idempotency_key,run_scope,
    issuer_id,security_id,dossier_id,
    entry_path,canonical_mode,run_type,
    run_status,current_stage,data_cutoff,
    process_version,pilotage_contract_version,
    contract_pins,contract_set_sha256,state_version
  ) values (
    v_run,'v17:' || p_tag || ':' || v_run::text,'COMPANY_ANALYSIS',
    v_issuer,v_security,v_dossier,
    'IMPOSED_COMPANY','ANALYZE','INITIAL',
    'ACTIVE',p_stage,date '2026-09-20',
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
    v_run,p_stage,1,
    v_pins->v_stage_key->>'name',
    v_pins->v_stage_key->>'version',
    v_pins->v_stage_key->>'content_sha256',
    'IN_PROGRESS','V17_TEST_IN_PROGRESS',
    v_gate,'NOT_EVALUATED',
    1,now()
  );

  run_id := v_run;
  issuer_id := v_issuer;
  security_id := v_security;
  dossier_id := v_dossier;
  return next;
end;
$$;

create or replace function pg_temp.v17_manifest(
  p_run_id uuid,
  p_stage text,
  p_kind text,
  p_revision integer,
  p_manifest_id uuid,
  p_output_regs jsonb,
  p_parent_manifest_id uuid default null,
  p_parent_manifest_version integer default null
)
returns jsonb
language plpgsql
as $$
declare
  v_run public.orotitan_runs%rowtype;
  v_stage public.orotitan_run_stages%rowtype;
  v_refs jsonb;
  v_stage_key text := case p_stage
    when 'RESEARCH' then 'research_stage'
    when 'DEEP_DIVE' then 'deep_dive_stage'
    when 'INTEGRATION' then 'integration_stage'
  end;
  v_complete boolean := p_kind='FINAL';
begin
  select * into v_run from public.orotitan_runs where run_id=p_run_id;
  select * into v_stage from public.orotitan_run_stages where run_id=p_run_id and stage_code=p_stage;
  select jsonb_agg(pg_temp.v17_ref(value) order by ordinality)
    into v_refs
  from jsonb_array_elements(p_output_regs) with ordinality;

  return jsonb_build_object(
    'manifest_schema_version','1.0.0',
    'manifest_id',p_manifest_id,
    'manifest_kind',p_kind,
    'run_id',p_run_id,
    'stage',p_stage,
    'stage_revision',p_revision,
    'issuer_id',v_run.issuer_id,
    'security_id',v_run.security_id,
    'dossier_id',v_run.dossier_id,
    'canonical_mode',v_run.canonical_mode,
    'run_type',v_run.run_type,
    'data_cutoff',v_run.data_cutoff::text,
    'baseline_snapshot_id',v_run.baseline_snapshot_id,
    'process_version',v_run.process_version,
    'pilotage_contract_version',v_run.pilotage_contract_version,
    'contract_pins',v_run.contract_pins,
    'stage_contract',v_run.contract_pins->v_stage_key,
    'contract_set_sha256',v_run.contract_set_sha256,
    'input_artifacts','[]'::jsonb,
    'output_artifacts',coalesce(v_refs,'[]'::jsonb),
    'stage_status',case when v_complete then 'COMPLETE' else 'IN_PROGRESS' end,
    'contract_status_code',case
      when p_stage='INTEGRATION' and v_complete then 'COMPLETE_READY_TO_PUBLISH'
      when v_complete then 'COMPLETE'
      else 'V17_TEST_CHECKPOINT'
    end,
    'handoff_gate',jsonb_build_object(
      'name',v_stage.handoff_gate_name,
      'state',case when v_complete then 'YES' else 'NOT_EVALUATED' end
    ),
    'critical_blockers','[]'::jsonb,
    'open_material_limitations','[]'::jsonb,
    'parent_manifests',case
      when p_parent_manifest_id is null then '[]'::jsonb
      else jsonb_build_array(jsonb_build_object(
        'manifest_id',p_parent_manifest_id,
        'version',p_parent_manifest_version
      ))
    end,
    'started_at','2026-09-20T10:00:00Z',
    'completed_at',case when v_complete then to_jsonb('2026-09-20T11:00:00Z'::text) else 'null'::jsonb end
  );
end;
$$;

create or replace function pg_temp.v17_edges(
  p_run_id uuid,
  p_manifest_id uuid,
  p_manifest_version integer,
  p_regs jsonb,
  p_predecessor_manifest_id uuid default null,
  p_predecessor_manifest_version integer default null
)
returns jsonb
language plpgsql
as $$
declare
  v_edges jsonb;
begin
  select coalesce(jsonb_agg(jsonb_build_object(
    'child_run_id',p_run_id,
    'child_artifact_id',p_manifest_id,
    'child_version',p_manifest_version,
    'parent_run_id',p_run_id,
    'parent_artifact_id',value->>'artifact_id',
    'parent_version',(value->>'version')::integer,
    'relation_type','CONSUMES'
  ) order by ordinality),'[]'::jsonb)
    into v_edges
  from jsonb_array_elements(p_regs) with ordinality;

  if p_predecessor_manifest_id is not null then
    v_edges := v_edges || jsonb_build_array(jsonb_build_object(
      'child_run_id',p_run_id,
      'child_artifact_id',p_manifest_id,
      'child_version',p_manifest_version,
      'parent_run_id',p_run_id,
      'parent_artifact_id',p_predecessor_manifest_id,
      'parent_version',p_predecessor_manifest_version,
      'relation_type','SUPERSEDES'
    ));
  end if;
  return v_edges;
end;
$$;

create or replace function pg_temp.v17_checkpoint(
  p_run_id uuid,
  p_manifest_id uuid,
  p_manifest_version integer,
  p_regs jsonb,
  p_idempotency_key text,
  p_fingerprint text,
  p_predecessor_version integer default null
)
returns jsonb
language plpgsql
as $$
declare
  v_run public.orotitan_runs%rowtype;
  v_stage public.orotitan_run_stages%rowtype;
  v_manifest jsonb;
  v_manifest_reg jsonb;
  v_edges jsonb;
begin
  select * into v_run from public.orotitan_runs where run_id=p_run_id;
  select * into v_stage from public.orotitan_run_stages where run_id=p_run_id and stage_code='DEEP_DIVE';

  v_manifest := pg_temp.v17_manifest(
    p_run_id,'DEEP_DIVE','CHECKPOINT',v_stage.stage_revision,
    p_manifest_id,p_regs,
    case when p_predecessor_version is null then null else p_manifest_id end,
    p_predecessor_version
  );
  v_manifest_reg := pg_temp.v17_manifest_registration(
    v_manifest,p_manifest_id,p_manifest_version,'DEEP_DIVE_STAGE_MANIFEST',
    'v17/' || p_run_id || '/checkpoint-' || p_manifest_version || '.json',
    'CHECKPOINT_STAGE_OUTPUT','CHECKPOINT'
  );
  v_edges := pg_temp.v17_edges(
    p_run_id,p_manifest_id,p_manifest_version,p_regs,
    case when p_predecessor_version is null then null else p_manifest_id end,
    p_predecessor_version
  );

  return public.checkpoint_orotitan_stage(
    p_run_id,'DEEP_DIVE',v_run.state_version,v_stage.state_version,
    v_manifest,v_manifest_reg,p_regs,v_edges,
    'IN_PROGRESS',p_idempotency_key,p_fingerprint,'SYSTEM'
  );
end;
$$;

-- TEST 1 / 2 / 3 / 18: all-new, all-reused, partial-reuse, CHECKPOINT successor.
do $$
declare
  v_run uuid; v_manifest uuid := gen_random_uuid();
  a jsonb; b jsonb; c jsonb; d jsonb; e jsonb;
  regs1 jsonb; regs2 jsonb; regs3 jsonb;
begin
  select run_id into v_run from pg_temp.v17_create_run('T1-T3','DEEP_DIVE');

  a := pg_temp.v17_artifact(gen_random_uuid(),1,'A','1','v17/a.json');
  b := pg_temp.v17_artifact(gen_random_uuid(),1,'B','2','v17/b.json');
  c := pg_temp.v17_artifact(gen_random_uuid(),1,'C','3','v17/c.json');
  regs1 := jsonb_build_array(a,b,c);
  perform pg_temp.v17_checkpoint(v_run,v_manifest,1,regs1,'v17:t1:m1',repeat('1',64),null);

  d := pg_temp.v17_artifact(gen_random_uuid(),1,'D','4','v17/d.json');
  e := pg_temp.v17_artifact(gen_random_uuid(),1,'E','5','v17/e.json');
  regs2 := jsonb_build_array(d,e,pg_temp.v17_artifact(gen_random_uuid(),1,'F','6','v17/f.json'));
  perform pg_temp.v17_checkpoint(v_run,v_manifest,2,regs2,'v17:t1:m2',repeat('2',64),1);
  if (select count(*) from public.orotitan_artifacts where artifact_id in ((a->>'artifact_id')::uuid,(b->>'artifact_id')::uuid,(c->>'artifact_id')::uuid) and authority_state='SUPERSEDED') <> 3
     or (select count(*) from public.orotitan_artifacts where manifest_artifact_id=v_manifest and manifest_version=2 and authority_state='CHECKPOINT' and artifact_type in ('D','E','F')) <> 3 then
    raise exception 'TEST 1 failed';
  end if;

  -- all reused from M2 -> M3
  perform pg_temp.v17_checkpoint(v_run,v_manifest,3,regs2,'v17:t2:m3',repeat('3',64),2);
  if (select count(*) from public.orotitan_artifacts where artifact_id in ((d->>'artifact_id')::uuid,(e->>'artifact_id')::uuid) and authority_state='CHECKPOINT' and manifest_artifact_id=v_manifest and manifest_version=3) <> 2
     or not exists (select 1 from public.orotitan_artifacts where artifact_type='F' and run_id=v_run and authority_state='CHECKPOINT' and manifest_version=3) then
    raise exception 'TEST 2 / 18 failed';
  end if;

  -- partial reuse from M3 -> M4: keep D/E, drop F, add G.
  regs3 := jsonb_build_array(d,e,pg_temp.v17_artifact(gen_random_uuid(),1,'G','7','v17/g.json'));
  perform pg_temp.v17_checkpoint(v_run,v_manifest,4,regs3,'v17:t3:m4',repeat('4',64),3);
  if not exists (select 1 from public.orotitan_artifacts where artifact_id=(d->>'artifact_id')::uuid and authority_state='CHECKPOINT' and manifest_version=4)
     or not exists (select 1 from public.orotitan_artifacts where artifact_id=(e->>'artifact_id')::uuid and authority_state='CHECKPOINT' and manifest_version=4)
     or not exists (select 1 from public.orotitan_artifacts where artifact_type='F' and run_id=v_run and authority_state='SUPERSEDED' and manifest_version=3)
     or not exists (select 1 from public.orotitan_artifacts where artifact_type='G' and run_id=v_run and authority_state='CHECKPOINT' and manifest_version=4) then
    raise exception 'TEST 3 failed';
  end if;
end $$;

-- TEST 5 / 6 / 7 / 8 / 9 / 10: immutable conflict, wrong run/stage, absent successor.
do $$
declare
  r1 uuid; r2 uuid; m1 uuid := gen_random_uuid(); m2 uuid := gen_random_uuid();
  a jsonb; regs jsonb; bad jsonb; rejected boolean;
  manifest jsonb; manifest_reg jsonb; edges jsonb;
begin
  select run_id into r1 from pg_temp.v17_create_run('T5-T10-A','DEEP_DIVE');
  select run_id into r2 from pg_temp.v17_create_run('T5-T10-B','DEEP_DIVE');
  a := pg_temp.v17_artifact(gen_random_uuid(),1,'IMMUTABLE_A','8','v17/immutable-a.json');
  regs := jsonb_build_array(a);
  perform pg_temp.v17_checkpoint(r1,m1,1,regs,'v17:t5:base',repeat('5',64),null);

  -- TEST 5 hash conflict.
  bad := jsonb_set(a,'{content_sha256}',to_jsonb(repeat('0',64)));
  rejected := false;
  begin
    perform pg_temp.v17_checkpoint(r1,m1,2,jsonb_build_array(bad),'v17:t5:hash',repeat('6',64),1);
  exception when others then rejected := true; end;
  if not rejected then raise exception 'TEST 5 failed'; end if;

  -- TEST 6 size conflict.
  bad := jsonb_set(a,'{size_bytes}','3'::jsonb);
  rejected := false;
  begin
    perform pg_temp.v17_checkpoint(r1,m1,2,jsonb_build_array(bad),'v17:t6:size',repeat('7',64),1);
  exception when others then rejected := true; end;
  if not rejected then raise exception 'TEST 6 failed'; end if;

  -- TEST 7 storage provenance conflict.
  bad := jsonb_set(a,'{storage_uri}',to_jsonb('supabase://orotitan-private/v17/changed.json'::text));
  bad := jsonb_set(bad,'{supabase_object_path}',to_jsonb('v17/changed.json'::text));
  rejected := false;
  begin
    perform pg_temp.v17_checkpoint(r1,m1,2,jsonb_build_array(bad),'v17:t7:storage',repeat('8',64),1);
  exception when others then rejected := true; end;
  if not rejected then raise exception 'TEST 7 failed'; end if;

  -- TEST 8 wrong run: exact artifact from r1 cannot bind into r2.
  manifest := pg_temp.v17_manifest(r2,'DEEP_DIVE','CHECKPOINT',1,m2,jsonb_build_array(a),null,null);
  manifest_reg := pg_temp.v17_manifest_registration(manifest,m2,1,'DEEP_DIVE_STAGE_MANIFEST','v17/wrong-run-manifest.json','CHECKPOINT_STAGE_OUTPUT','CHECKPOINT');
  rejected := false;
  begin
    perform public.orotitan_register_manifest_bundle(r2,'DEEP_DIVE',manifest,manifest_reg,jsonb_build_array(a),'[]'::jsonb);
  exception when others then rejected := true; end;
  if not rejected then raise exception 'TEST 8 failed'; end if;

  -- TEST 9 wrong stage.
  rejected := false;
  begin
    perform public.orotitan_register_manifest_bundle(r1,'RESEARCH',manifest,manifest_reg,jsonb_build_array(a),'[]'::jsonb);
  exception when others then rejected := true; end;
  if not rejected then raise exception 'TEST 9 failed'; end if;

  -- TEST 10 artifact not explicitly in successor manifest.
  manifest := pg_temp.v17_manifest(r1,'DEEP_DIVE','CHECKPOINT',1,m2,jsonb_build_array(a),null,null);
  manifest_reg := pg_temp.v17_manifest_registration(manifest,m2,1,'DEEP_DIVE_STAGE_MANIFEST','v17/absent-manifest.json','CHECKPOINT_STAGE_OUTPUT','CHECKPOINT');
  bad := pg_temp.v17_artifact(gen_random_uuid(),1,'EXTRA','a','v17/extra.json');
  rejected := false;
  begin
    perform public.orotitan_register_manifest_bundle(r1,'DEEP_DIVE',manifest,manifest_reg,jsonb_build_array(a,bad),'[]'::jsonb);
  exception when others then rejected := true; end;
  if not rejected then raise exception 'TEST 10 failed'; end if;
end $$;

-- TEST 11 / 12: INVALIDATED, WITHDRAWN and MISSING reject.
do $$
declare
  r uuid; m uuid := gen_random_uuid(); a jsonb; regs jsonb; rejected boolean;
begin
  select run_id into r from pg_temp.v17_create_run('T11','DEEP_DIVE');
  a := pg_temp.v17_artifact(gen_random_uuid(),1,'INVALIDATED_A','b','v17/invalidated.json');
  regs := jsonb_build_array(a);
  perform pg_temp.v17_checkpoint(r,m,1,regs,'v17:t11:base',repeat('9',64),null);
  update public.orotitan_artifacts set artifact_status='INVALIDATED'
    where artifact_id=(a->>'artifact_id')::uuid and version=1;
  rejected := false;
  begin
    perform pg_temp.v17_checkpoint(r,m,2,regs,'v17:t11:retry',repeat('a',64),1);
  exception when others then rejected := true; end;
  if not rejected then raise exception 'TEST 11 failed'; end if;

  m := gen_random_uuid();
  select run_id into r from pg_temp.v17_create_run('T12W','DEEP_DIVE');
  a := pg_temp.v17_artifact(gen_random_uuid(),1,'WITHDRAWN_A','c','v17/withdrawn.json');
  regs := jsonb_build_array(a);
  perform pg_temp.v17_checkpoint(r,m,1,regs,'v17:t12w:base',repeat('b',64),null);
  update public.orotitan_artifacts set availability_state='WITHDRAWN'
    where artifact_id=(a->>'artifact_id')::uuid and version=1;
  rejected := false;
  begin
    perform pg_temp.v17_checkpoint(r,m,2,regs,'v17:t12w:retry',repeat('c',64),1);
  exception when others then rejected := true; end;
  if not rejected then raise exception 'TEST 12 WITHDRAWN failed'; end if;

  m := gen_random_uuid();
  select run_id into r from pg_temp.v17_create_run('T12M','DEEP_DIVE');
  a := pg_temp.v17_artifact(gen_random_uuid(),1,'MISSING_A','d','v17/missing.json');
  regs := jsonb_build_array(a);
  perform pg_temp.v17_checkpoint(r,m,1,regs,'v17:t12m:base',repeat('d',64),null);
  update public.orotitan_artifacts set availability_state='MISSING'
    where artifact_id=(a->>'artifact_id')::uuid and version=1;
  rejected := false;
  begin
    perform pg_temp.v17_checkpoint(r,m,2,regs,'v17:t12m:retry',repeat('e',64),1);
  exception when others then rejected := true; end;
  if not rejected then raise exception 'TEST 12 MISSING failed'; end if;
end $$;

-- TEST 15 / 16 / 17: idempotent replay, fingerprint collision, history preservation.
do $$
declare
  r uuid; m uuid := gen_random_uuid(); a jsonb; regs jsonb; res1 jsonb; res2 jsonb;
  event_count bigint; rejected boolean := false;
begin
  select run_id into r from pg_temp.v17_create_run('T15-T17','DEEP_DIVE');
  a := pg_temp.v17_artifact(gen_random_uuid(),1,'IDEMP_A','e','v17/idemp.json');
  regs := jsonb_build_array(a);
  res1 := pg_temp.v17_checkpoint(r,m,1,regs,'v17:t15',repeat('f',64),null);
  select count(*) into event_count from public.orotitan_run_events where run_id=r and idempotency_key='v17:t15';
  res2 := public.checkpoint_orotitan_stage(
    r,'DEEP_DIVE',
    (select state_version-1 from public.orotitan_runs where run_id=r),
    (select state_version-1 from public.orotitan_run_stages where run_id=r and stage_code='DEEP_DIVE'),
    pg_temp.v17_manifest(r,'DEEP_DIVE','CHECKPOINT',1,m,regs,null,null),
    pg_temp.v17_manifest_registration(
      pg_temp.v17_manifest(r,'DEEP_DIVE','CHECKPOINT',1,m,regs,null,null),
      m,1,'DEEP_DIVE_STAGE_MANIFEST','v17/' || r || '/checkpoint-1.json',
      'CHECKPOINT_STAGE_OUTPUT','CHECKPOINT'
    ),
    regs,pg_temp.v17_edges(r,m,1,regs,null,null),
    'IN_PROGRESS','v17:t15',repeat('f',64),'SYSTEM'
  );
  if not (res2->>'idempotent_replay')::boolean
     or (select count(*) from public.orotitan_run_events where run_id=r and idempotency_key='v17:t15') <> event_count then
    raise exception 'TEST 15 failed';
  end if;

  begin
    perform public.checkpoint_orotitan_stage(
      r,'DEEP_DIVE',
      (select state_version from public.orotitan_runs where run_id=r),
      (select state_version from public.orotitan_run_stages where run_id=r and stage_code='DEEP_DIVE'),
      pg_temp.v17_manifest(r,'DEEP_DIVE','CHECKPOINT',1,m,regs,null,null),
      pg_temp.v17_manifest_registration(
      pg_temp.v17_manifest(r,'DEEP_DIVE','CHECKPOINT',1,m,regs,null,null),
      m,1,'DEEP_DIVE_STAGE_MANIFEST','v17/' || r || '/checkpoint-1.json',
      'CHECKPOINT_STAGE_OUTPUT','CHECKPOINT'
    ),
      regs,pg_temp.v17_edges(r,m,1,regs,null,null),
      'IN_PROGRESS','v17:t15',repeat('0',64),'SYSTEM'
    );
  exception when others then rejected := true; end;
  if not rejected then raise exception 'TEST 16 failed'; end if;

  perform pg_temp.v17_checkpoint(r,m,2,regs,'v17:t17:m2',repeat('1',64),1);
  if not exists(select 1 from public.orotitan_artifacts where artifact_id=m and version=1 and authority_state='SUPERSEDED')
     or not exists(select 1 from public.orotitan_run_events where run_id=r and event_type='STAGE_CHECKPOINTED' and payload->>'manifest_version'='1')
     or not exists(select 1 from public.orotitan_artifacts where artifact_id=(a->>'artifact_id')::uuid and version=1 and authority_state='CHECKPOINT' and manifest_version=2) then
    raise exception 'TEST 17 failed';
  end if;
end $$;

-- TEST 4 / 13 / 14 / 19 / 20: COMPLETE reopen -> FINAL successor, stale CAS,
-- FINAL exact-output reactivation and Brookfield-shaped 6 reused + 1 changed.
do $$
declare
  r uuid; issuer uuid; security uuid; dossier uuid;
  m uuid := gen_random_uuid();
  snap jsonb; map1 jsonb; map2 jsonb; schema_r jsonb; i2 jsonb; hist jsonb; i3b jsonb; prepub jsonb;
  regs1 jsonb; regs2 jsonb; manifest1 jsonb; manifest2 jsonb; mreg1 jsonb; mreg2 jsonb; edges1 jsonb; edges2 jsonb;
  run_v bigint; stage_v bigint; reopen_result jsonb; finalize_result jsonb;
  rejected boolean; old_event uuid; reused_before jsonb; reused_after jsonb;
begin
  select run_id,issuer_id,security_id,dossier_id into r,issuer,security,dossier
  from pg_temp.v17_create_run('T4-T20','INTEGRATION');

  snap := pg_temp.v17_artifact(gen_random_uuid(),1,'CANONICAL_SNAPSHOT_CANDIDATE','1','v17/final/snapshot.json','AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE');
  map1 := pg_temp.v17_artifact(gen_random_uuid(),1,'INTEGRATION_MAPPING_RECORD','2','v17/final/mapping-v1.json','AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE');
  schema_r := pg_temp.v17_artifact(gen_random_uuid(),1,'SCHEMA_VALIDATION_REPORT','3','v17/final/schema.json','AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE');
  i2 := pg_temp.v17_artifact(gen_random_uuid(),1,'I2_RECONCILIATION_REPORT','4','v17/final/i2.json','AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE');
  hist := pg_temp.v17_artifact(gen_random_uuid(),1,'HISTORY_TRANSITION_VALIDATION_REPORT','5','v17/final/history.json','AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE');
  i3b := pg_temp.v17_artifact(gen_random_uuid(),1,'I3B_ADMISSION_REPORT','6','v17/final/i3b.json','AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE');
  prepub := pg_temp.v17_artifact(gen_random_uuid(),1,'PRE_PUBLICATION_CONTROL_CARD','7','v17/final/prepub.json','AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE');
  regs1 := jsonb_build_array(snap,map1,schema_r,i2,hist,i3b,prepub);

  manifest1 := pg_temp.v17_manifest(r,'INTEGRATION','FINAL',1,m,regs1,null,null);
  mreg1 := pg_temp.v17_manifest_registration(manifest1,m,1,'INTEGRATION_STAGE_MANIFEST','v17/final/manifest-v1.json','AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE');
  edges1 := pg_temp.v17_edges(r,m,1,regs1,null,null);
  select state_version into run_v from public.orotitan_runs where run_id=r;
  select state_version into stage_v from public.orotitan_run_stages where run_id=r and stage_code='INTEGRATION';
  perform public.finalize_orotitan_stage(r,'INTEGRATION',run_v,stage_v,manifest1,mreg1,regs1,edges1,'v17:final:m1',repeat('2',64),'SYSTEM');

  select event_id into old_event from public.orotitan_run_events
    where run_id=r and event_type='STAGE_FINALIZED' and payload->>'manifest_version'='1';

  -- TEST 13 stale run CAS.
  select state_version into run_v from public.orotitan_runs where run_id=r;
  select state_version into stage_v from public.orotitan_run_stages where run_id=r and stage_code='INTEGRATION';
  rejected := false;
  begin
    perform public.reopen_orotitan_stage(r,'INTEGRATION',run_v-1,stage_v,'IN_PROGRESS',
      jsonb_build_object('code','V17_TEST','summary','stale run'),
      'v17:t13',repeat('3',64));
  exception when others then rejected := true; end;
  if not rejected then raise exception 'TEST 13 failed'; end if;

  -- TEST 14 stale stage CAS.
  rejected := false;
  begin
    perform public.reopen_orotitan_stage(r,'INTEGRATION',run_v,stage_v-1,'IN_PROGRESS',
      jsonb_build_object('code','V17_TEST','summary','stale stage'),
      'v17:t14',repeat('4',64));
  exception when others then rejected := true; end;
  if not rejected then raise exception 'TEST 14 failed'; end if;

  reopen_result := public.reopen_orotitan_stage(r,'INTEGRATION',run_v,stage_v,'IN_PROGRESS',
    jsonb_build_object('code','REGISTRY_REUSED_OUTPUT_REBIND_TEST','summary','controlled final rebind regression'),
    'v17:t4:reopen',repeat('5',64));

  if (select stage_revision from public.orotitan_run_stages where run_id=r and stage_code='INTEGRATION') <> 2
     or exists(select 1 from public.orotitan_artifacts where run_id=r and (artifact_id=m or manifest_artifact_id=m) and authority_state <> 'SUPERSEDED') then
    raise exception 'TEST 4 reopen predecessor supersession failed';
  end if;

  select jsonb_agg(jsonb_build_object(
    'artifact_id',artifact_id,'version',version,'content_sha256',content_sha256,
    'size_bytes',size_bytes,'storage_uri',storage_uri
  ) order by artifact_id)
  into reused_before
  from public.orotitan_artifacts
  where run_id=r
    and artifact_id in (
      (snap->>'artifact_id')::uuid,(schema_r->>'artifact_id')::uuid,(i2->>'artifact_id')::uuid,
      (hist->>'artifact_id')::uuid,(i3b->>'artifact_id')::uuid,(prepub->>'artifact_id')::uuid
    );

  -- Changed logical Mapping gets version 2; all six other exact outputs are reused.
  map2 := pg_temp.v17_artifact((map1->>'artifact_id')::uuid,2,'INTEGRATION_MAPPING_RECORD','9','v17/final/mapping-v2.json','AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE');
  regs2 := jsonb_build_array(snap,map2,schema_r,i2,hist,i3b,prepub);
  manifest2 := pg_temp.v17_manifest(r,'INTEGRATION','FINAL',2,m,regs2,m,1);
  mreg2 := pg_temp.v17_manifest_registration(manifest2,m,2,'INTEGRATION_STAGE_MANIFEST','v17/final/manifest-v2.json','AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE');
  edges2 := pg_temp.v17_edges(r,m,2,regs2,m,1);

  select state_version into run_v from public.orotitan_runs where run_id=r;
  select state_version into stage_v from public.orotitan_run_stages where run_id=r and stage_code='INTEGRATION';
  finalize_result := public.finalize_orotitan_stage(
    r,'INTEGRATION',run_v,stage_v,manifest2,mreg2,regs2,edges2,
    'v17:t4:finalize2',repeat('6',64),'SYSTEM'
  );

  select jsonb_agg(jsonb_build_object(
    'artifact_id',artifact_id,'version',version,'content_sha256',content_sha256,
    'size_bytes',size_bytes,'storage_uri',storage_uri
  ) order by artifact_id)
  into reused_after
  from public.orotitan_artifacts
  where run_id=r
    and artifact_id in (
      (snap->>'artifact_id')::uuid,(schema_r->>'artifact_id')::uuid,(i2->>'artifact_id')::uuid,
      (hist->>'artifact_id')::uuid,(i3b->>'artifact_id')::uuid,(prepub->>'artifact_id')::uuid
    );

  if reused_before is distinct from reused_after then
    raise exception 'TEST 4/19/20 immutable reuse changed';
  end if;
  if (select count(*) from public.orotitan_artifacts where run_id=r
      and artifact_id in (
        (snap->>'artifact_id')::uuid,(schema_r->>'artifact_id')::uuid,(i2->>'artifact_id')::uuid,
        (hist->>'artifact_id')::uuid,(i3b->>'artifact_id')::uuid,(prepub->>'artifact_id')::uuid
      )
      and authority_state='AUTHORITATIVE' and manifest_artifact_id=m and manifest_version=2) <> 6 then
    raise exception 'TEST 4/19/20 reused FINAL outputs not rebound 6/6';
  end if;
  if not exists(select 1 from public.orotitan_artifacts where artifact_id=(map1->>'artifact_id')::uuid and version=1 and authority_state='SUPERSEDED' and manifest_version=1)
     or not exists(select 1 from public.orotitan_artifacts where artifact_id=(map2->>'artifact_id')::uuid and version=2 and authority_state='AUTHORITATIVE' and manifest_artifact_id=m and manifest_version=2)
     or not exists(select 1 from public.orotitan_artifacts where artifact_id=m and version=1 and authority_state='SUPERSEDED')
     or not exists(select 1 from public.orotitan_artifacts where artifact_id=m and version=2 and authority_state='AUTHORITATIVE')
     or not exists(select 1 from public.orotitan_run_events where event_id=old_event and event_type='STAGE_FINALIZED') then
    raise exception 'TEST 4/17/19/20 history/change-set closure failed';
  end if;
  if not exists(select 1 from public.orotitan_artifact_edges where child_run_id=r and child_artifact_id=m and child_version=2 and parent_artifact_id=m and parent_version=1 and relation_type='SUPERSEDES') then
    raise exception 'TEST 17/20 successor lineage missing';
  end if;
  if not exists(select 1 from public.orotitan_runs where run_id=r and run_status='READY_TO_PUBLISH' and published_at is null)
     or not exists(select 1 from public.orotitan_run_stages where run_id=r and stage_code='INTEGRATION' and lifecycle_status='COMPLETE' and active_manifest_artifact_id=m and active_manifest_version=2 and active_manifest_kind='FINAL' and handoff_gate_state='YES')
     or exists(select 1 from public.orotitan_run_events where run_id=r and event_type in ('PUBLISH_AUTHORIZED','PUBLISH_SUCCEEDED','PUBLISH_FAILED')) then
    raise exception 'TEST 20 READY_TO_PUBLISH/publication firewall failed';
  end if;
end $$;

-- Structural security guard for the internal rebinding primitives.
do $$
declare
  fn record;
begin
  for fn in
    select p.oid::regprocedure as signature,p.prosecdef,p.proconfig
    from pg_proc p
    join pg_namespace n on n.oid=p.pronamespace
    where n.nspname='public'
      and p.proname in ('orotitan_insert_artifact_registration','orotitan_register_manifest_bundle','supersede_orotitan_manifest_bundle')
  loop
    if not (fn.proconfig @> array['search_path=pg_catalog, public']::text[]) then
      raise exception 'V1.7 internal function search_path invalid: %',fn.signature;
    end if;
    if has_function_privilege('public',fn.signature::text,'EXECUTE')
       or has_function_privilege('anon',fn.signature::text,'EXECUTE')
       or has_function_privilege('authenticated',fn.signature::text,'EXECUTE')
       or has_function_privilege('service_role',fn.signature::text,'EXECUTE') then
      raise exception 'V1.7 internal function privilege boundary invalid: %',fn.signature;
    end if;
  end loop;
end $$;

select jsonb_build_object(
  'TEST_01_ALL_NEW','PASS',
  'TEST_02_ALL_REUSED','PASS',
  'TEST_03_PARTIAL_REUSE','PASS',
  'TEST_04_COMPLETE_STAGE_REOPEN_REFINALIZE','PASS',
  'TEST_05_HASH_CONFLICT','PASS',
  'TEST_06_SIZE_CONFLICT','PASS',
  'TEST_07_STORAGE_PROVENANCE_CONFLICT','PASS',
  'TEST_08_WRONG_RUN','PASS',
  'TEST_09_WRONG_STAGE','PASS',
  'TEST_10_NOT_IN_SUCCESSOR_MANIFEST','PASS',
  'TEST_11_INVALIDATED','PASS',
  'TEST_12_WITHDRAWN_MISSING','PASS',
  'TEST_13_STALE_RUN_CAS','PASS',
  'TEST_14_STALE_STAGE_CAS','PASS',
  'TEST_15_IDEMPOTENT_REPLAY','PASS',
  'TEST_16_FINGERPRINT_COLLISION','PASS',
  'TEST_17_HISTORICAL_PRESERVATION','PASS',
  'TEST_18_CHECKPOINT_SUCCESSOR','PASS',
  'TEST_19_FINAL_SUCCESSOR','PASS',
  'TEST_20_BROOKFIELD_SHAPE','PASS',
  'REGRESSION_TEST_RESULT','PASS ALL'
) as orotitan_registry_v17_successor_rebinding_regression;
