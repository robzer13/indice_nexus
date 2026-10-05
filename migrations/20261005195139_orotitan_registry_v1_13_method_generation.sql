-- Registry V1.13: explicit analytical generation, orthogonal to execution pins.
-- CODE ONLY. No deployment, activation, snapshot mutation or runtime grants.
begin;

-- Constant defaults grandfather existing rows without an UPDATE or table rewrite.
alter table public.orotitan_runs
  add column methodology_generation text not null default 'METHOD_V1',
  add column methodology_authority_sha256 text,
  add constraint orotitan_runs_methodology_identity_check check (
    (methodology_generation = 'METHOD_V1' and methodology_authority_sha256 is null)
    or (methodology_generation = 'METHOD_V2'
        and methodology_authority_sha256 is not null
        and methodology_authority_sha256 = '1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2')
  );

comment on column public.orotitan_runs.methodology_generation is
  'Analytical generation only; never inferred from process, schema, pins, snapshot method_version or runtime labels.';
comment on column public.orotitan_runs.methodology_authority_sha256 is
  'Exact frozen Method-V2 V1.1 authority-set identity; NULL for METHOD_V1.';

-- Only V2 children are restricted; unrelated historical lineages remain legal.
create unique index orotitan_runs_one_method_v2_successor_idx
  on public.orotitan_runs(parent_run_id)
  where parent_run_id is not null and methodology_generation = 'METHOD_V2';

create function public.enforce_orotitan_methodology_identity_update()
returns trigger language plpgsql set search_path = pg_catalog, public as $$
begin
  if new.methodology_generation is distinct from old.methodology_generation
     or new.methodology_authority_sha256 is distinct from old.methodology_authority_sha256 then
    raise exception 'immutable OroTitan methodology identity; create a controlled successor'
      using errcode = '23514';
  end if;
  return new;
end;
$$;
create trigger orotitan_runs_methodology_identity_immutable
before update on public.orotitan_runs for each row
execute function public.enforce_orotitan_methodology_identity_update();
revoke all on function public.enforce_orotitan_methodology_identity_update()
  from public, anon, authenticated, service_role;

-- Legacy signature and V1 admission retained; prevent cross-generation replay.
create or replace function public.create_orotitan_run(
  p_creation_idempotency_key text,
  p_issuer_id uuid,
  p_entry_path text,
  p_canonical_mode text,
  p_run_type text,
  p_data_cutoff date,
  p_parent_run_id uuid,
  p_baseline_snapshot_id uuid,
  p_process_version text,
  p_pilotage_contract_version text,
  p_contract_pins jsonb,
  p_contract_set_sha256 text,
  p_request_fingerprint_sha256 text
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $function$
declare
  v_existing public.orotitan_runs%rowtype;
  v_run_id uuid := gen_random_uuid();
  v_security_id uuid;
  v_dossier_id uuid;
  v_baseline_issuer uuid;
  v_event_id uuid;
begin
  if p_issuer_id is null then
    raise exception 'STAGE_NOT_ADMITTED: issuer_id is required for company-analysis run'
      using errcode = '23514';
  end if;
  if p_request_fingerprint_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid request fingerprint' using errcode = '22023';
  end if;
  if public.orotitan_contract_set_sha256(p_contract_pins) <> p_contract_set_sha256 then
    raise exception 'RUN_CONTRACT_SET_MISMATCH: contract_set_sha256 does not reconcile'
      using errcode = '23514';
  end if;

  select *
    into v_existing
  from public.orotitan_runs
  where creation_idempotency_key = p_creation_idempotency_key;

  if found then
    if v_existing.methodology_generation is distinct from 'METHOD_V1' then
      raise exception 'IDEMPOTENCY_CONFLICT: legacy route requires METHOD_V1' using errcode = '23514';
    end if;
    if not exists (
      select 1
      from public.orotitan_run_events e
      where e.run_id = v_existing.run_id
        and e.event_type = 'RUN_CREATED'
        and e.idempotency_key = p_creation_idempotency_key
        and e.request_fingerprint_sha256 = p_request_fingerprint_sha256
    ) then
      raise exception 'IDEMPOTENCY_CONFLICT: run creation key reused with different request'
        using errcode = '23514';
    end if;

    return jsonb_build_object(
      'run_id', v_existing.run_id,
      'state_version', v_existing.state_version,
      'run_status', v_existing.run_status,
      'idempotent_replay', true
    );
  end if;

  if p_parent_run_id is not null then
    raise exception 'PARENT_LINK_REQUIRES_CONTROLLED_SUCCESSOR_RPC'
      using errcode = '23514';
  end if;

  if p_run_type = 'REFRESH' then
    if p_baseline_snapshot_id is null then
      raise exception 'REFRESH requires baseline_snapshot_id' using errcode = '23514';
    end if;

    select rs.security_id, rs.dossier_id, rs.issuer_id
      into v_security_id, v_dossier_id, v_baseline_issuer
    from public.research_snapshots rs
    where rs.snapshot_id = p_baseline_snapshot_id;

    if not found or v_baseline_issuer <> p_issuer_id then
      raise exception 'REFRESH baseline snapshot does not match issuer'
        using errcode = '23514';
    end if;
  elsif p_baseline_snapshot_id is not null then
    raise exception 'INITIAL/non-REFRESH run must not silently inherit baseline snapshot'
      using errcode = '23514';
  end if;

  insert into public.orotitan_runs (
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
    data_cutoff,
    process_version,
    pilotage_contract_version,
    contract_pins,
    contract_set_sha256
  ) values (
    v_run_id,
    p_creation_idempotency_key,
    p_issuer_id,
    v_security_id,
    v_dossier_id,
    null,
    p_baseline_snapshot_id,
    p_entry_path,
    p_canonical_mode,
    p_run_type,
    'CREATED',
    p_data_cutoff,
    p_process_version,
    p_pilotage_contract_version,
    p_contract_pins,
    p_contract_set_sha256
  );

  v_event_id := public.orotitan_insert_event(
    v_run_id,
    null,
    'RUN_CREATED',
    p_creation_idempotency_key,
    p_request_fingerprint_sha256,
    'PILOTAGE',
    jsonb_build_object(
      'run_id', v_run_id,
      'canonical_mode', p_canonical_mode,
      'run_type', p_run_type
    )
  );

  return jsonb_build_object(
    'run_id', v_run_id,
    'state_version', 1,
    'run_status', 'CREATED',
    'event_id', v_event_id,
    'idempotent_replay', false
  );
end;
$function$;

-- Owner-only materialization path; no runtime/API role execution.
create function public.create_orotitan_method_v2_run(
  p_creation_idempotency_key text,
  p_issuer_id uuid,
  p_entry_path text,
  p_canonical_mode text,
  p_run_type text,
  p_data_cutoff date,
  p_parent_run_id uuid,
  p_baseline_snapshot_id uuid,
  p_process_version text,
  p_pilotage_contract_version text,
  p_contract_pins jsonb,
  p_contract_set_sha256 text,
  p_request_fingerprint_sha256 text,
  p_methodology_authority_sha256 text
)
returns jsonb
language plpgsql
security invoker
set search_path = pg_catalog, public
as $function$
declare
  v_existing public.orotitan_runs%rowtype;
  v_run_id uuid := gen_random_uuid();
  v_security_id uuid;
  v_dossier_id uuid;
  v_baseline_issuer uuid;
  v_event_id uuid;
  v_request jsonb := jsonb_build_object(
    'p_creation_idempotency_key', p_creation_idempotency_key,
    'p_issuer_id', p_issuer_id,
    'p_entry_path', p_entry_path,
    'p_canonical_mode', p_canonical_mode,
    'p_run_type', p_run_type,
    'p_data_cutoff', p_data_cutoff,
    'p_parent_run_id', p_parent_run_id,
    'p_baseline_snapshot_id', p_baseline_snapshot_id,
    'p_process_version', p_process_version,
    'p_pilotage_contract_version', p_pilotage_contract_version,
    'p_contract_pins', p_contract_pins,
    'p_contract_set_sha256', p_contract_set_sha256,
    'p_request_fingerprint_sha256', p_request_fingerprint_sha256,
    'p_methodology_authority_sha256', p_methodology_authority_sha256
  );
begin
  if p_methodology_authority_sha256 is distinct from '1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2' then
    raise exception 'METHOD_V2_AUTHORITY_MISMATCH' using errcode = '23514';
  end if;
  if p_issuer_id is null then
    raise exception 'STAGE_NOT_ADMITTED: issuer_id is required for company-analysis run'
      using errcode = '23514';
  end if;
  if p_request_fingerprint_sha256 is null or p_request_fingerprint_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid request fingerprint' using errcode = '22023';
  end if;
  if public.orotitan_contract_set_sha256(p_contract_pins) is distinct from p_contract_set_sha256 then
    raise exception 'RUN_CONTRACT_SET_MISMATCH: contract_set_sha256 does not reconcile'
      using errcode = '23514';
  end if;

  select *
    into v_existing
  from public.orotitan_runs
  where creation_idempotency_key = p_creation_idempotency_key;

  if found then
    if v_existing.methodology_generation is distinct from 'METHOD_V2'
       or v_existing.methodology_authority_sha256 is distinct from p_methodology_authority_sha256
       or not exists (select 1 from public.orotitan_run_events e
         where e.run_id = v_existing.run_id and e.event_type = 'RUN_CREATED'
           and e.payload->'creation_request' = v_request) then
      raise exception 'IDEMPOTENCY_CONFLICT: Method-V2 request differs' using errcode = '23514';
    end if;
    if not exists (
      select 1
      from public.orotitan_run_events e
      where e.run_id = v_existing.run_id
        and e.event_type = 'RUN_CREATED'
        and e.idempotency_key = p_creation_idempotency_key
        and e.request_fingerprint_sha256 = p_request_fingerprint_sha256
    ) then
      raise exception 'IDEMPOTENCY_CONFLICT: run creation key reused with different request'
        using errcode = '23514';
    end if;

    return jsonb_build_object(
      'run_id', v_existing.run_id,
      'state_version', v_existing.state_version,
      'run_status', v_existing.run_status,
      'idempotent_replay', true
    );
  end if;

  if p_parent_run_id is not null then
    raise exception 'PARENT_LINK_REQUIRES_CONTROLLED_SUCCESSOR_RPC'
      using errcode = '23514';
  end if;

  if p_run_type = 'REFRESH' then
    if p_baseline_snapshot_id is null then
      raise exception 'REFRESH requires baseline_snapshot_id' using errcode = '23514';
    end if;

    select rs.security_id, rs.dossier_id, rs.issuer_id
      into v_security_id, v_dossier_id, v_baseline_issuer
    from public.research_snapshots rs
    where rs.snapshot_id = p_baseline_snapshot_id;

    if not found or v_baseline_issuer <> p_issuer_id then
      raise exception 'REFRESH baseline snapshot does not match issuer'
        using errcode = '23514';
    end if;
  elsif p_baseline_snapshot_id is not null then
    raise exception 'INITIAL/non-REFRESH run must not silently inherit baseline snapshot'
      using errcode = '23514';
  end if;

  insert into public.orotitan_runs (
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
    data_cutoff,
    process_version,
    pilotage_contract_version,
    contract_pins,
    contract_set_sha256,
    methodology_generation,
    methodology_authority_sha256
  ) values (
    v_run_id,
    p_creation_idempotency_key,
    p_issuer_id,
    v_security_id,
    v_dossier_id,
    null,
    p_baseline_snapshot_id,
    p_entry_path,
    p_canonical_mode,
    p_run_type,
    'CREATED',
    p_data_cutoff,
    p_process_version,
    p_pilotage_contract_version,
    p_contract_pins,
    p_contract_set_sha256,
    'METHOD_V2',
    p_methodology_authority_sha256
  );

  v_event_id := public.orotitan_insert_event(
    v_run_id,
    null,
    'RUN_CREATED',
    p_creation_idempotency_key,
    p_request_fingerprint_sha256,
    'PILOTAGE',
    jsonb_build_object(
      'methodology_generation', 'METHOD_V2',
      'methodology_authority_sha256', p_methodology_authority_sha256,
      'creation_request', v_request,
      'run_id', v_run_id,
      'canonical_mode', p_canonical_mode,
      'run_type', p_run_type
    )
  );

  return jsonb_build_object(
    'run_id', v_run_id,
    'state_version', 1,
    'run_status', 'CREATED',
    'event_id', v_event_id,
    'idempotent_replay', false
  );
end;
$function$;

revoke all on function public.create_orotitan_method_v2_run(
  text, uuid, text, text, text, date, uuid, uuid, text, text, jsonb, text, text, text
) from public, anon, authenticated, service_role;

-- Legacy signature and V1 admission retained; prevent cross-generation replay.
create or replace function public.create_orotitan_methodology_successor_run(
  p_creation_idempotency_key text,
  p_issuer_id uuid,
  p_entry_path text,
  p_canonical_mode text,
  p_run_type text,
  p_data_cutoff date,
  p_parent_run_id uuid,
  p_baseline_snapshot_id uuid,
  p_process_version text,
  p_pilotage_contract_version text,
  p_contract_pins jsonb,
  p_contract_set_sha256 text,
  p_request_fingerprint_sha256 text,
  p_expected_parent_state_version bigint,
  p_expected_parent_run_status text,
  p_expected_parent_current_stage text,
  p_expected_parent_contract_set_sha256 text,
  p_expected_parent_security_id uuid,
  p_expected_parent_dossier_id uuid
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $function$
declare
  v_existing public.orotitan_runs%rowtype;
  v_parent public.orotitan_runs%rowtype;
  v_parent_stage public.orotitan_run_stages%rowtype;
  v_current_snapshot_id uuid;
  v_run_id uuid := gen_random_uuid();
  v_event_id uuid;
  v_is_blocked_repair boolean := false;
begin
  if p_request_fingerprint_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid request fingerprint' using errcode = '22023';
  end if;

  select *
    into v_existing
  from public.orotitan_runs
  where creation_idempotency_key = p_creation_idempotency_key;

  if found then
    if v_existing.methodology_generation is distinct from 'METHOD_V1' then
      raise exception 'IDEMPOTENCY_CONFLICT: legacy route requires METHOD_V1' using errcode = '23514';
    end if;
    if v_existing.parent_run_id is distinct from p_parent_run_id
       or v_existing.issuer_id is distinct from p_issuer_id
       or v_existing.data_cutoff is distinct from p_data_cutoff
       or v_existing.run_type is distinct from 'INITIAL'
       or v_existing.baseline_snapshot_id is not null
       or v_existing.contract_set_sha256 is distinct from p_contract_set_sha256
       or not exists (
         select 1
         from public.orotitan_run_events e
         where e.run_id = v_existing.run_id
           and e.event_type = 'RUN_CREATED'
           and e.idempotency_key = p_creation_idempotency_key
           and e.request_fingerprint_sha256 = p_request_fingerprint_sha256
       ) then
      raise exception 'IDEMPOTENCY_CONFLICT: methodology successor creation key reused with different request'
        using errcode = '23514';
    end if;

    return jsonb_build_object(
      'run_id', v_existing.run_id,
      'state_version', v_existing.state_version,
      'run_status', v_existing.run_status,
      'idempotent_replay', true
    );
  end if;

  if p_contract_set_sha256 <> '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2'
     or public.orotitan_contract_set_sha256(p_contract_pins) <> '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2' then
    raise exception 'SUCCESSOR_ACTIVE_CONTRACT_SET_MISMATCH'
      using errcode = '23514';
  end if;

  if (select count(*) from jsonb_object_keys(p_contract_pins)) <> 15
     or p_contract_pins #>> '{process,name}' <> 'OROTITAN_EXECUTION_PROCESS_V3_1_FREEZE_V3.1'
     or p_contract_pins #>> '{process,version}' <> '3.1'
     or p_contract_pins #>> '{pilotage,name}' <> 'OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_1_FREEZE_V3.1'
     or p_contract_pins #>> '{pilotage,version}' <> '3.1'
     or p_contract_pins #>> '{deep_dive_stage,version}' <> '3.1'
     or p_contract_pins #>> '{integration_stage,version}' <> '3.1'
     or p_contract_pins #>> '{dcf_timing,name}' <> 'OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0'
     or p_contract_pins #>> '{dcf_timing,version}' <> '1.0'
     or p_contract_pins #>> '{dcf_timing,content_sha256}' <> '9bec7dcb3af85019806f255a0d504cd1770f50fe9343fb92d8f367cb13b5c4ee'
     or p_contract_pins #>> '{valuation_date_alignment,name}' <> 'OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0'
     or p_contract_pins #>> '{valuation_date_alignment,version}' <> '1.0'
     or p_contract_pins #>> '{valuation_date_alignment,content_sha256}' <> '3c4e315a5b0759d13b99d77b8af5eb0298e81e46fe9ee3706ed08f3947b68ee5'
     or p_process_version <> '3.1'
     or p_pilotage_contract_version <> '3.1' then
    raise exception 'SUCCESSOR_ACTIVE_CONTRACT_SET_MISMATCH: active authority composition is not exact'
      using errcode = '23514';
  end if;

  if p_parent_run_id is null then
    raise exception 'SUCCESSOR_PARENT_NOT_FOUND: parent_run_id is required'
      using errcode = '23514';
  end if;
  if p_run_type is distinct from 'INITIAL' then
    raise exception 'SUCCESSOR_RUN_TYPE_INVALID: legal RUN_TYPE is INITIAL'
      using errcode = '23514';
  end if;
  if p_baseline_snapshot_id is not null then
    raise exception 'SUCCESSOR_BASELINE_MISMATCH: methodology replay must not inherit a baseline snapshot'
      using errcode = '23514';
  end if;
  if p_expected_parent_run_status not in ('ACTIVE','BLOCKED') then
    raise exception 'SUCCESSOR_PARENT_STATUS_MISMATCH: expected status must be ACTIVE or exact governed BLOCKED repair'
      using errcode = '23514';
  end if;
  if p_expected_parent_current_stage is distinct from 'DEEP_DIVE' then
    raise exception 'SUCCESSOR_PARENT_STAGE_MISMATCH: expected parent stage must be DEEP_DIVE'
      using errcode = '23514';
  end if;

  select *
    into v_parent
  from public.orotitan_runs
  where run_id = p_parent_run_id
  for update;

  if not found then
    raise exception 'SUCCESSOR_PARENT_NOT_FOUND' using errcode = 'P0002';
  end if;
  if v_parent.state_version <> p_expected_parent_state_version then
    raise exception 'SUCCESSOR_PARENT_STATE_VERSION_MISMATCH'
      using errcode = '40001';
  end if;
  if v_parent.run_status <> p_expected_parent_run_status then
    raise exception 'SUCCESSOR_PARENT_STATUS_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.current_stage is distinct from p_expected_parent_current_stage then
    raise exception 'SUCCESSOR_PARENT_STAGE_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.published_at is not null
     or v_parent.cancelled_at is not null
     or v_parent.run_status in ('PUBLISHED','CANCELLED','READY_TO_PUBLISH') then
    raise exception 'SUCCESSOR_PARENT_TERMINAL' using errcode = '23514';
  end if;
  if v_parent.data_cutoff <> p_data_cutoff then
    raise exception 'SUCCESSOR_PARENT_CUTOFF_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.contract_set_sha256 <> p_expected_parent_contract_set_sha256 then
    raise exception 'SUCCESSOR_PARENT_CONTRACT_SET_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.contract_set_sha256 = p_contract_set_sha256 then
    raise exception 'SUCCESSOR_PARENT_ALREADY_ON_ACTIVE_CONTRACT_SET' using errcode = '23514';
  end if;
  if v_parent.issuer_id is distinct from p_issuer_id
     or v_parent.security_id is distinct from p_expected_parent_security_id
     or v_parent.dossier_id is distinct from p_expected_parent_dossier_id then
    raise exception 'SUCCESSOR_PARENT_IDENTITY_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.entry_path is distinct from p_entry_path
     or v_parent.canonical_mode is distinct from p_canonical_mode then
    raise exception 'SUCCESSOR_PARENT_ROUTING_MISMATCH' using errcode = '23514';
  end if;

  if v_parent.run_status = 'BLOCKED' then
    if v_parent.contract_set_sha256 <> '257c287357c19a5d47a42f140a1eb0377d48701b04b07e1e9e740646797c172c' then
      raise exception 'SUCCESSOR_BLOCKED_PARENT_NOT_REPAIRABLE: wrong historical Contract Set'
        using errcode = '23514';
    end if;

    select *
      into v_parent_stage
    from public.orotitan_run_stages
    where run_id = p_parent_run_id
      and stage_code = 'DEEP_DIVE'
    for update;

    if not found
       or v_parent_stage.lifecycle_status <> 'BLOCKED'
       or v_parent_stage.contract_status_code <> 'VALUATION_BLOCKED_PINNED_TIMING_DENOMINATOR_DATE_CONFLICT'
       or v_parent_stage.stage_contract_name <> 'OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_FREEZE_V3.0'
       or v_parent_stage.stage_contract_version <> '3.0'
       or v_parent_stage.stage_contract_sha256 <> 'f52932630702d4249d47da370604899c07d7aad9046fbd616945cdfccf2dbd16'
       or v_parent_stage.handoff_gate_state <> 'NOT_EVALUATED'
       or jsonb_typeof(v_parent_stage.blocker_summary) <> 'array'
       or jsonb_array_length(v_parent_stage.blocker_summary) <> 1
       or v_parent_stage.blocker_summary->0->>'code' <> 'V3_VALUATION_TIMING_DENOMINATOR_DATE_CONFLICT'
       or v_parent_stage.blocker_summary->0->>'classification' <> 'PINNED_AUTHORITY_CONFLICT'
       or v_parent_stage.blocker_summary->0->>'scope' <> 'VALUATION_DCF_AND_DEPENDENT_OUTPUTS' then
      raise exception 'SUCCESSOR_BLOCKED_PARENT_NOT_REPAIRABLE: exact timing-conflict state not proven'
        using errcode = '23514';
    end if;

    v_is_blocked_repair := true;
  end if;

  select d.current_snapshot_id
    into v_current_snapshot_id
  from public.research_dossiers d
  where d.dossier_id = p_expected_parent_dossier_id
    and d.issuer_id = p_issuer_id
  for update;

  if not found then
    raise exception 'SUCCESSOR_PARENT_IDENTITY_MISMATCH: dossier not found'
      using errcode = '23514';
  end if;
  if v_current_snapshot_id is not null then
    raise exception 'SUCCESSOR_BASELINE_MISMATCH: current canonical snapshot now exists'
      using errcode = '23514';
  end if;

  insert into public.orotitan_runs (
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
    data_cutoff,
    process_version,
    pilotage_contract_version,
    contract_pins,
    contract_set_sha256
  ) values (
    v_run_id,
    p_creation_idempotency_key,
    p_issuer_id,
    p_expected_parent_security_id,
    p_expected_parent_dossier_id,
    p_parent_run_id,
    null,
    p_entry_path,
    p_canonical_mode,
    'INITIAL',
    'CREATED',
    p_data_cutoff,
    p_process_version,
    p_pilotage_contract_version,
    p_contract_pins,
    p_contract_set_sha256
  );

  v_event_id := public.orotitan_insert_event(
    v_run_id,
    null,
    'RUN_CREATED',
    p_creation_idempotency_key,
    p_request_fingerprint_sha256,
    'PILOTAGE',
    jsonb_build_object(
      'run_id', v_run_id,
      'canonical_mode', p_canonical_mode,
      'run_type', 'INITIAL',
      'parent_run_id', p_parent_run_id,
      'creation_reason', 'METHODOLOGY_REPLAY_SUCCESSOR',
      'parent_state_version_cas', p_expected_parent_state_version,
      'parent_status_cas', p_expected_parent_run_status,
      'blocked_methodology_defect_repair', v_is_blocked_repair
    )
  );

  return jsonb_build_object(
    'run_id', v_run_id,
    'state_version', 1,
    'run_status', 'CREATED',
    'event_id', v_event_id,
    'idempotent_replay', false,
    'blocked_methodology_defect_repair', v_is_blocked_repair
  );
end;
$function$;

-- Owner-only materialization path; no runtime/API role execution.
create function public.create_orotitan_method_v2_successor_run(
  p_creation_idempotency_key text,
  p_issuer_id uuid,
  p_entry_path text,
  p_canonical_mode text,
  p_run_type text,
  p_data_cutoff date,
  p_parent_run_id uuid,
  p_baseline_snapshot_id uuid,
  p_process_version text,
  p_pilotage_contract_version text,
  p_contract_pins jsonb,
  p_contract_set_sha256 text,
  p_request_fingerprint_sha256 text,
  p_expected_parent_state_version bigint,
  p_expected_parent_run_status text,
  p_expected_parent_current_stage text,
  p_expected_parent_contract_set_sha256 text,
  p_expected_parent_security_id uuid,
  p_expected_parent_dossier_id uuid,
  p_methodology_authority_sha256 text
)
returns jsonb
language plpgsql
security invoker
set search_path = pg_catalog, public
as $function$
declare
  v_existing public.orotitan_runs%rowtype;
  v_parent public.orotitan_runs%rowtype;
  v_current_snapshot_id uuid;
  v_run_id uuid := gen_random_uuid();
  v_event_id uuid;
  v_request jsonb := jsonb_build_object(
    'p_creation_idempotency_key', p_creation_idempotency_key,
    'p_issuer_id', p_issuer_id,
    'p_entry_path', p_entry_path,
    'p_canonical_mode', p_canonical_mode,
    'p_run_type', p_run_type,
    'p_data_cutoff', p_data_cutoff,
    'p_parent_run_id', p_parent_run_id,
    'p_baseline_snapshot_id', p_baseline_snapshot_id,
    'p_process_version', p_process_version,
    'p_pilotage_contract_version', p_pilotage_contract_version,
    'p_contract_pins', p_contract_pins,
    'p_contract_set_sha256', p_contract_set_sha256,
    'p_request_fingerprint_sha256', p_request_fingerprint_sha256,
    'p_expected_parent_state_version', p_expected_parent_state_version,
    'p_expected_parent_run_status', p_expected_parent_run_status,
    'p_expected_parent_current_stage', p_expected_parent_current_stage,
    'p_expected_parent_contract_set_sha256', p_expected_parent_contract_set_sha256,
    'p_expected_parent_security_id', p_expected_parent_security_id,
    'p_expected_parent_dossier_id', p_expected_parent_dossier_id,
    'p_methodology_authority_sha256', p_methodology_authority_sha256
  );
begin
  if p_methodology_authority_sha256 is distinct from '1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2' then
    raise exception 'METHOD_V2_AUTHORITY_MISMATCH' using errcode = '23514';
  end if;
  if p_request_fingerprint_sha256 is null or p_request_fingerprint_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid request fingerprint' using errcode = '22023';
  end if;

  -- Serialize by parent before checking replay, including simultaneous identical requests.
  select *
    into v_parent
  from public.orotitan_runs
  where run_id = p_parent_run_id
  for update;

  if not found then
    raise exception 'SUCCESSOR_PARENT_NOT_FOUND' using errcode = 'P0002';
  end if;

  -- Preserve exact idempotent replay of a child already created by this route.
  select *
    into v_existing
  from public.orotitan_runs
  where creation_idempotency_key = p_creation_idempotency_key;

  if found then
    if v_existing.methodology_generation is distinct from 'METHOD_V2'
       or v_existing.methodology_authority_sha256 is distinct from p_methodology_authority_sha256
       or not exists (select 1 from public.orotitan_run_events e
         where e.run_id = v_existing.run_id and e.event_type = 'RUN_CREATED'
           and e.payload->'creation_request' = v_request) then
      raise exception 'IDEMPOTENCY_CONFLICT: Method-V2 request differs' using errcode = '23514';
    end if;
    if v_existing.parent_run_id is distinct from p_parent_run_id
       or v_existing.issuer_id is distinct from p_issuer_id
       or v_existing.data_cutoff is distinct from p_data_cutoff
       or v_existing.run_type is distinct from 'INITIAL'
       or v_existing.baseline_snapshot_id is not null
       or v_existing.contract_set_sha256 is distinct from p_contract_set_sha256
       or not exists (
         select 1
         from public.orotitan_run_events e
         where e.run_id = v_existing.run_id
           and e.event_type = 'RUN_CREATED'
           and e.idempotency_key = p_creation_idempotency_key
           and e.request_fingerprint_sha256 = p_request_fingerprint_sha256
       ) then
      raise exception 'IDEMPOTENCY_CONFLICT: methodology successor creation key reused with different request'
        using errcode = '23514';
    end if;

    return jsonb_build_object(
      'run_id', v_existing.run_id,
      'state_version', v_existing.state_version,
      'run_status', v_existing.run_status,
      'idempotent_replay', true
    );
  end if;

  -- Identity-only transition: execution pins are compared to the locked parent below.
  if p_parent_run_id is null then
    raise exception 'SUCCESSOR_PARENT_NOT_FOUND: parent_run_id is required'
      using errcode = '23514';
  end if;
  if p_run_type is distinct from 'INITIAL' then
    raise exception 'SUCCESSOR_RUN_TYPE_INVALID: legal RUN_TYPE is INITIAL'
      using errcode = '23514';
  end if;
  if p_baseline_snapshot_id is not null then
    raise exception 'SUCCESSOR_BASELINE_MISMATCH: methodology replay must not inherit a baseline snapshot'
      using errcode = '23514';
  end if;
  if p_expected_parent_run_status is distinct from 'ACTIVE' then
    raise exception 'SUCCESSOR_PARENT_STATUS_MISMATCH: expected parent status must be ACTIVE'
      using errcode = '23514';
  end if;
  if p_expected_parent_current_stage is distinct from 'DEEP_DIVE' then
    raise exception 'SUCCESSOR_PARENT_STAGE_MISMATCH: expected parent stage must be DEEP_DIVE'
      using errcode = '23514';
  end if;

  if v_parent.state_version is distinct from p_expected_parent_state_version then
    raise exception 'SUCCESSOR_PARENT_STATE_VERSION_MISMATCH'
      using errcode = '40001';
  end if;
  if v_parent.run_status <> p_expected_parent_run_status then
    raise exception 'SUCCESSOR_PARENT_STATUS_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.current_stage is distinct from p_expected_parent_current_stage then
    raise exception 'SUCCESSOR_PARENT_STAGE_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.published_at is not null
     or v_parent.cancelled_at is not null
     or v_parent.run_status in ('PUBLISHED', 'CANCELLED', 'READY_TO_PUBLISH') then
    raise exception 'SUCCESSOR_PARENT_TERMINAL' using errcode = '23514';
  end if;
  if v_parent.data_cutoff is distinct from p_data_cutoff then
    raise exception 'SUCCESSOR_PARENT_CUTOFF_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.contract_set_sha256 is distinct from p_expected_parent_contract_set_sha256 then
    raise exception 'SUCCESSOR_PARENT_CONTRACT_SET_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.methodology_generation is distinct from 'METHOD_V1'
     or v_parent.methodology_authority_sha256 is not null then
    raise exception 'METHOD_V2_SUCCESSOR_REQUIRES_METHOD_V1_PARENT' using errcode = '23514';
  end if;
  if v_parent.contract_set_sha256 is distinct from p_contract_set_sha256
     or v_parent.contract_pins is distinct from p_contract_pins
     or v_parent.process_version is distinct from p_process_version
     or v_parent.pilotage_contract_version is distinct from p_pilotage_contract_version then
    raise exception 'METHOD_V2_SUCCESSOR_EXECUTION_IDENTITY_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.issuer_id is distinct from p_issuer_id
     or v_parent.security_id is distinct from p_expected_parent_security_id
     or v_parent.dossier_id is distinct from p_expected_parent_dossier_id then
    raise exception 'SUCCESSOR_PARENT_IDENTITY_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.entry_path is distinct from p_entry_path
     or v_parent.canonical_mode is distinct from p_canonical_mode then
    raise exception 'SUCCESSOR_PARENT_ROUTING_MISMATCH' using errcode = '23514';
  end if;

  -- Lock the dossier in the same transaction so a no-baseline methodology
  -- replay cannot race canonical publication.
  select d.current_snapshot_id
    into v_current_snapshot_id
  from public.research_dossiers d
  where d.dossier_id = p_expected_parent_dossier_id
    and d.issuer_id = p_issuer_id
  for update;

  if not found then
    raise exception 'SUCCESSOR_PARENT_IDENTITY_MISMATCH: dossier not found'
      using errcode = '23514';
  end if;
  if v_current_snapshot_id is not null then
    raise exception 'SUCCESSOR_BASELINE_MISMATCH: current canonical snapshot now exists'
      using errcode = '23514';
  end if;

  if exists (select 1 from public.orotitan_runs
             where parent_run_id = p_parent_run_id and methodology_generation = 'METHOD_V2') then
    raise exception 'METHOD_V2_SUCCESSOR_ALREADY_EXISTS' using errcode = '23514';
  end if;

  insert into public.orotitan_runs (
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
    data_cutoff,
    process_version,
    pilotage_contract_version,
    contract_pins,
    contract_set_sha256,
    methodology_generation,
    methodology_authority_sha256
  ) values (
    v_run_id,
    p_creation_idempotency_key,
    p_issuer_id,
    p_expected_parent_security_id,
    p_expected_parent_dossier_id,
    p_parent_run_id,
    null,
    p_entry_path,
    p_canonical_mode,
    'INITIAL',
    'CREATED',
    p_data_cutoff,
    p_process_version,
    p_pilotage_contract_version,
    p_contract_pins,
    p_contract_set_sha256,
    'METHOD_V2',
    p_methodology_authority_sha256
  );

  v_event_id := public.orotitan_insert_event(
    v_run_id,
    null,
    'RUN_CREATED',
    p_creation_idempotency_key,
    p_request_fingerprint_sha256,
    'PILOTAGE',
    jsonb_build_object(
      'methodology_generation', 'METHOD_V2',
      'methodology_authority_sha256', p_methodology_authority_sha256,
      'creation_request', v_request,
      'run_id', v_run_id,
      'canonical_mode', p_canonical_mode,
      'run_type', 'INITIAL',
      'parent_run_id', p_parent_run_id,
      'creation_reason', 'METHODOLOGY_REPLAY_SUCCESSOR',
      'parent_state_version_cas', p_expected_parent_state_version
    )
  );

  return jsonb_build_object(
    'run_id', v_run_id,
    'state_version', 1,
    'run_status', 'CREATED',
    'event_id', v_event_id,
    'idempotent_replay', false
  );
end;
$function$;

revoke all on function public.create_orotitan_method_v2_successor_run(
  text, uuid, text, text, text, date, uuid, uuid, text, text, jsonb, text, text, bigint, text, text, text, uuid, uuid, text
) from public, anon, authenticated, service_role;

commit;
