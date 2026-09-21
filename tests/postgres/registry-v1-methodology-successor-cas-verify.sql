\set ON_ERROR_STOP on

do $verify$
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
$verify$;

do $verify$
declare
  v_parent_id uuid := '31000000-0000-4000-8000-000000000001';
  v_identity record;
  v_old_pins jsonb;
  v_new_pins jsonb;
  v_new_hash text;
  v_result jsonb;
  v_child public.orotitan_runs%rowtype;
  v_failed boolean;
  v_alt_pins jsonb;
  v_alt_hash text;
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
    'process', jsonb_build_object(
      'name','OLD_PROCESS',
      'version','2.0',
      'content_sha256',repeat('1',64)
    )
  );

  v_new_pins := '{"process":{"name":"OROTITAN_EXECUTION_PROCESS_V3_FREEZE_V3.0","version":"3.0","content_sha256":"8bf1817d4a3d3655b386e452b54eb50be1025ce56d4933bd1cbb5f0ef8c04dd7","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_EXECUTION_PROCESS_V3_FREEZE_V3.0.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"11e3ae0acd6fa0645572e778aef6e51fcf70ddc6"}},"pilotage":{"name":"OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_FREEZE_V3.0","version":"3.0","content_sha256":"724d74ec82c5908f4d5d3c8a8062e64bfa19ced9abadee634ffd353e2ad21fa9","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_FREEZE_V3.0.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"9aaa2a2f87af1e00d7880ee97926db4662afcbcb"}},"research_stage":{"name":"OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0","version":"2.0","content_sha256":"cdd62f087ff8f8dc97f0c16586d634001f2cef1c44f9e4540e087778cdaa7a01","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v2/OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0.md","commit_sha":"86b227a75275cf4aaec6ef61ea2a27e87e2bfec7","blob_sha":"4b49d4ab670717d844ff0ed3c925a2004c043dfa"}},"deep_dive_stage":{"name":"OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_FREEZE_V3.0","version":"3.0","content_sha256":"f52932630702d4249d47da370604899c07d7aad9046fbd616945cdfccf2dbd16","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_FREEZE_V3.0.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"0cbf3c8ecc08b6b947152345f7a1532163d993c1"}},"integration_stage":{"name":"OROTITAN_INTEGRATION_STAGE_CONTRACT_V3_FREEZE_V3.0","version":"3.0","content_sha256":"09534d88e3ac4d37cfa0f55b7e762692cd198b7baff424ed169396bdd6851759","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_INTEGRATION_STAGE_CONTRACT_V3_FREEZE_V3.0.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"4ca5214238b1ff08a4da12dec8a89e942ac905a0"}},"analysis_standard":{"name":"OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0","version":"1.0","content_sha256":"538b70a975d00b8bfc9c3abf705d5efd425852516680d499db13d8c2ac17174f","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/01_ANALYSIS_STANDARD_V1_1_ECONOMIC_SHARE_COUNT_PATCH.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"bfd034c6a650bb2bd0db1ad3cb3d4476aa9ca66e"}},"master_prompt":{"name":"02_OROTITAN_MASTER_PROMPT_V1_1_ECONOMIC_SHARE_COUNT_PATCH","version":"1.1","content_sha256":"2447ad836d9507eb82db884b1bdb1e0062d1512526d12afa4593339ce51dd0df","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/02_OROTITAN_MASTER_PROMPT_V1_1_ECONOMIC_SHARE_COUNT_PATCH.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"b04ad374b79ed958a29dcb5b3d9943d394ece190"}},"investment_policy":{"name":"OROTITAN_INVESTMENT_POLICY_V1.0.0","version":"1.0.0","content_sha256":"66cc29ccaccf4ccf65f58c2959bf95bcf676e67567aa1dd57863b2cb9a2936c8","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/OROTITAN_INVESTMENT_POLICY_V1.0.0.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"fd0121bbb14e35370ddd700d9e9845eaf1ec9f75","locator_format":"RAW_CANONICAL_V1"}},"execution_patch":{"name":"OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1","version":"1.0.1","content_sha256":"6e45f39c03e79c34911850c84904b36f3d95266288a5528127a59d0b8d675973","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"449bb43b664eae9d8c9fd98440e5511c6fbf4be3","locator_format":"RAW_CANONICAL_V1"}},"integration_spec":{"name":"04_INTEGRATION_SPEC_V3","version":"3.0","content_sha256":"2ab3923f18973cad469510c13906a0641275f2d112c43cbdf7a9047975395e39","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/04_INTEGRATION_SPEC_V3.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"45bb894b3081d4cdbe464fb668d8176752085931"}},"screener_schema":{"name":"04_SCREENER_SCHEMA_V3","version":"3.0.0","content_sha256":"372a4e162b8f5d510661ae6ffadb6347c28fedce62779cf2ac08da3425ae4890","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/04_SCREENER_SCHEMA_V3.json","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"a53e8506bfd4dd348c8ffd46282e46da341aa0ca"}},"i2":{"name":"I2_CANONICAL_COMPUTATION_V1.1","version":"1.1","content_sha256":"ebf4d0e23784aaf67fb48f1bd372ebe1bd8efdbef9dafbeb8ecee6b9f4082b00","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"docs/orotitan-equity/I2_CANONICAL_COMPUTATION_V1.1.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"e145448c1f1d901527a6fe9b62309a0f4159a25a"}},"i3b":{"name":"I3B_VALIDATED_SNAPSHOT_WRITER_V1.1","version":"1.1","content_sha256":"fbb803f5ee26d056bcaf9b1e67ee2c2d4b59bbc5ff637ceceee667e053eacd60","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"docs/orotitan-equity/I3B_VALIDATED_SNAPSHOT_WRITER_V1.1.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"6fe5b719d83e21b50568d007a4fba9df4e5fd0e0"}},"dcf_timing":{"name":"OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0","version":"1.0","content_sha256":"9bec7dcb3af85019806f255a0d504cd1770f50fe9343fb92d8f367cb13b5c4ee","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0.md","commit_sha":"1f63baf3c5ae03e5e0a0f90a2794a0f358398c5c","blob_sha":"4316d8e574151882ae25f83c3f7b97f8bad48cd3"}}}'::jsonb;
  v_new_hash := public.orotitan_contract_set_sha256(v_new_pins);
  if v_new_hash <> '257c287357c19a5d47a42f140a1eb0377d48701b04b07e1e9e740646797c172c' then
    raise exception 'active DCF timing Contract Set fixture mismatch: %', v_new_hash;
  end if;

  insert into public.orotitan_runs(
    run_id,
    creation_idempotency_key,
    issuer_id,
    security_id,
    dossier_id,
    parent_run_id,
    baseline_snapshot_id,
    entry_path,
    canonical_mode,
    run_type,
    run_status,
    current_stage,
    data_cutoff,
    process_version,
    pilotage_contract_version,
    contract_pins,
    contract_set_sha256,
    state_version
  ) values (
    v_parent_id,
    'test:parent:dcf-cas',
    v_identity.issuer_id,
    v_identity.security_id,
    v_identity.dossier_id,
    null,
    null,
    'IMPOSED_COMPANY',
    'ANALYZE',
    'INITIAL',
    'ACTIVE',
    'DEEP_DIVE',
    '2026-09-19',
    '2.0',
    '2.0',
    v_old_pins,
    repeat('1',64),
    7
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

  select *
    into v_child
  from public.orotitan_runs
  where run_id=(v_result->>'run_id')::uuid;

  if v_child.parent_run_id <> v_parent_id
     or v_child.run_type <> 'INITIAL'
     or v_child.baseline_snapshot_id is not null
     or v_child.data_cutoff <> date '2026-09-19'
     or v_child.security_id <> v_identity.security_id
     or v_child.dossier_id <> v_identity.dossier_id
     or v_child.contract_set_sha256 <> v_new_hash then
    raise exception 'successor child persistence mismatch';
  end if;

  if (public.create_orotitan_methodology_successor_run(
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
      repeat('b',64),
      7,
      'ACTIVE',
      'DEEP_DIVE',
      repeat('1',64),
      v_identity.security_id,
      v_identity.dossier_id
    );
  exception when serialization_failure then
    if position('SUCCESSOR_PARENT_STATE_VERSION_MISMATCH' in sqlerrm)>0 then
      v_failed := true;
    else
      raise;
    end if;
  end;
  if not v_failed then
    raise exception 'stale parent CAS did not fail closed';
  end if;

  update public.orotitan_runs
  set run_status='BLOCKED', state_version=9
  where run_id=v_parent_id;

  v_failed := false;
  begin
    perform public.create_orotitan_methodology_successor_run(
      'test:successor:dcf-cas:status',
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
      repeat('c',64),
      9,
      'ACTIVE',
      'DEEP_DIVE',
      repeat('1',64),
      v_identity.security_id,
      v_identity.dossier_id
    );
  exception when check_violation then
    if position('SUCCESSOR_PARENT_STATUS_MISMATCH' in sqlerrm)>0 then
      v_failed := true;
    else
      raise;
    end if;
  end;
  if not v_failed then
    raise exception 'parent status mismatch did not fail closed';
  end if;

  v_failed := false;
  begin
    perform public.create_orotitan_methodology_successor_run(
      'test:successor:dcf-cas:run-type',
      v_identity.issuer_id,
      'IMPOSED_COMPANY',
      'ANALYZE',
      'REFRESH',
      '2026-09-19',
      v_parent_id,
      null,
      '3.0',
      '3.0',
      v_new_pins,
      v_new_hash,
      repeat('d',64),
      9,
      'ACTIVE',
      'DEEP_DIVE',
      repeat('1',64),
      v_identity.security_id,
      v_identity.dossier_id
    );
  exception when check_violation then
    if position('SUCCESSOR_RUN_TYPE_INVALID' in sqlerrm)>0 then
      v_failed := true;
    else
      raise;
    end if;
  end;
  if not v_failed then
    raise exception 'illegal successor run_type did not fail closed';
  end if;

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
  if not v_failed then
    raise exception 'generic parent-link successor bypass did not fail closed';
  end if;

  v_alt_pins := jsonb_build_object(
    'process', jsonb_build_object(
      'name','ALTERNATE_PROCESS',
      'version','3.0',
      'content_sha256',repeat('f',64)
    )
  );
  v_alt_hash := public.orotitan_contract_set_sha256(v_alt_pins);

  v_failed := false;
  begin
    perform public.create_orotitan_methodology_successor_run(
      'test:successor:alternate-contract-set',
      v_identity.issuer_id,
      'IMPOSED_COMPANY',
      'ANALYZE',
      'INITIAL',
      '2026-09-19',
      v_parent_id,
      null,
      '3.0',
      '3.0',
      v_alt_pins,
      v_alt_hash,
      repeat('f',64),
      9,
      'ACTIVE',
      'DEEP_DIVE',
      repeat('1',64),
      v_identity.security_id,
      v_identity.dossier_id
    );
  exception when check_violation then
    if position('SUCCESSOR_ACTIVE_CONTRACT_SET_MISMATCH' in sqlerrm)>0 then
      v_failed := true;
    else
      raise;
    end if;
  end;
  if not v_failed then
    raise exception 'alternate self-consistent Contract Set was admitted';
  end if;
end;
$verify$;

select 'methodology successor transactional CAS regression: PASS' as result;
