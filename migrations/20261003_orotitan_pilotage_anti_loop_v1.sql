-- OroTitan Pilotage Anti-Loop / Lossless Resume V1
-- Additive operational control state only. No analytical truth, artifact, stage or publication semantics are changed.

begin;

do $$
begin
  if to_regclass('public.orotitan_runs') is null
     or to_regclass('public.orotitan_run_stages') is null then
    raise exception 'OroTitan Registry must exist before Pilotage continuation ledger';
  end if;

  if to_regclass('public.orotitan_pilotage_attempts') is not null then
    raise exception 'OroTitan Pilotage continuation ledger already exists; refusing non-reviewed reapplication';
  end if;
end;
$$;

create table public.orotitan_pilotage_attempts (
  attempt_id uuid primary key default gen_random_uuid(),
  run_id uuid not null references public.orotitan_runs(run_id) on delete restrict,
  stage_code text not null check (stage_code in ('RESEARCH','DEEP_DIVE','INTEGRATION')),
  run_state_version bigint not null check (run_state_version >= 1),
  stage_state_version bigint not null check (stage_state_version >= 1),
  state_fingerprint_sha256 text not null check (state_fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
  requested_operation text not null check (
    requested_operation in (
      'RESOLVE_BLOCKER',
      'RESUME_STAGE',
      'CONTINUE_STAGE',
      'HANDOFF_NEXT_STAGE',
      'SAVE_DURABLE_CHECKPOINT',
      'GO_PUBLISH'
    )
  ),
  exact_next_action text not null check (
    exact_next_action in (
      'RESOLVE_BLOCKER',
      'RESUME_STAGE',
      'CONTINUE_STAGE',
      'HANDOFF_NEXT_STAGE',
      'SAVE_DURABLE_CHECKPOINT',
      'AWAIT_EXPLICIT_GO_PUBLISH',
      'FAIL_CLOSED'
    )
  ),
  attempt_count bigint not null default 1 check (attempt_count >= 1),
  first_seen_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now(),
  unique (
    run_id,
    stage_code,
    run_state_version,
    stage_state_version,
    requested_operation
  ),
  foreign key (run_id, stage_code)
    references public.orotitan_run_stages(run_id, stage_code)
    on delete restrict
);

comment on table public.orotitan_pilotage_attempts is
  'Operational Pilotage anti-loop ledger. Records continuation attempts by exact durable state fingerprint; never analytical truth.';

create index orotitan_pilotage_attempts_run_stage_idx
  on public.orotitan_pilotage_attempts (run_id, stage_code, last_seen_at desc);

alter table public.orotitan_pilotage_attempts enable row level security;

revoke all on table public.orotitan_pilotage_attempts
  from public, anon, authenticated, service_role;
grant select on table public.orotitan_pilotage_attempts to service_role;

create or replace function public.register_orotitan_pilotage_attempt(
  p_run_id uuid,
  p_stage_code text,
  p_expected_run_state_version bigint,
  p_expected_stage_state_version bigint,
  p_state_fingerprint_sha256 text,
  p_requested_operation text,
  p_exact_next_action text
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_run public.orotitan_runs%rowtype;
  v_stage public.orotitan_run_stages%rowtype;
  v_attempt public.orotitan_pilotage_attempts%rowtype;
  v_has_process_state boolean;
  v_expected_next_action text;
begin
  if p_stage_code not in ('RESEARCH','DEEP_DIVE','INTEGRATION') then
    raise exception 'PILOTAGE_STAGE_INVALID' using errcode = '22023';
  end if;
  if p_expected_run_state_version is null or p_expected_run_state_version < 1
     or p_expected_stage_state_version is null or p_expected_stage_state_version < 1 then
    raise exception 'PILOTAGE_STATE_VERSION_INVALID' using errcode = '22023';
  end if;
  if p_state_fingerprint_sha256 is null
     or p_state_fingerprint_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'PILOTAGE_STATE_FINGERPRINT_INVALID' using errcode = '22023';
  end if;
  if p_requested_operation not in (
    'RESOLVE_BLOCKER',
    'RESUME_STAGE',
    'CONTINUE_STAGE',
    'HANDOFF_NEXT_STAGE',
    'SAVE_DURABLE_CHECKPOINT',
    'GO_PUBLISH'
  ) then
    raise exception 'PILOTAGE_REQUESTED_OPERATION_INVALID' using errcode = '22023';
  end if;
  if p_exact_next_action not in (
    'RESOLVE_BLOCKER',
    'RESUME_STAGE',
    'CONTINUE_STAGE',
    'HANDOFF_NEXT_STAGE',
    'SAVE_DURABLE_CHECKPOINT',
    'AWAIT_EXPLICIT_GO_PUBLISH',
    'FAIL_CLOSED'
  ) then
    raise exception 'PILOTAGE_EXACT_NEXT_ACTION_INVALID' using errcode = '22023';
  end if;

  select *
    into v_run
  from public.orotitan_runs
  where run_id = p_run_id
  for update;

  if not found then
    raise exception 'RUN_NOT_FOUND' using errcode = 'P0002';
  end if;
  if v_run.run_status in ('PUBLISHED','CANCELLED') then
    raise exception 'PILOTAGE_RUN_TERMINAL' using errcode = '23514';
  end if;
  if v_run.current_stage is distinct from p_stage_code then
    raise exception 'PILOTAGE_CURRENT_STAGE_MISMATCH' using errcode = '23514';
  end if;
  if v_run.state_version <> p_expected_run_state_version then
    raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001';
  end if;

  select *
    into v_stage
  from public.orotitan_run_stages
  where run_id = p_run_id
    and stage_code = p_stage_code
  for update;

  if not found then
    raise exception 'STAGE_NOT_FOUND' using errcode = 'P0002';
  end if;
  if v_stage.state_version <> p_expected_stage_state_version then
    raise exception 'STAGE_STATE_VERSION_MISMATCH' using errcode = '40001';
  end if;

  select exists (
    select 1
    from public.orotitan_artifacts a
    where a.run_id = p_run_id
      and a.stage_code = p_stage_code
      and (a.artifact_type = 'PROCESS_ENGINE_STATE' or a.logical_name = 'process_engine_state')
      and a.artifact_status = 'SEALED'
      and a.availability_state = 'AVAILABLE'
      and a.authority_state in ('AUTHORITATIVE','CHECKPOINT')
  ) into v_has_process_state;

  if v_stage.lifecycle_status = 'COMPLETE'
     and jsonb_array_length(v_stage.blocker_summary) > 0 then
    v_expected_next_action := 'FAIL_CLOSED';
  elsif v_stage.handoff_gate_state = 'YES'
        and v_stage.lifecycle_status <> 'COMPLETE' then
    v_expected_next_action := 'FAIL_CLOSED';
  elsif v_stage.lifecycle_status = 'BLOCKED'
        or jsonb_array_length(v_stage.blocker_summary) > 0 then
    v_expected_next_action := 'RESOLVE_BLOCKER';
  elsif v_stage.lifecycle_status = 'PAUSED' then
    if v_stage.active_manifest_artifact_id is null and not v_has_process_state then
      v_expected_next_action := 'SAVE_DURABLE_CHECKPOINT';
    else
      v_expected_next_action := 'RESUME_STAGE';
    end if;
  elsif v_stage.lifecycle_status = 'IN_PROGRESS' then
    if v_stage.active_manifest_artifact_id is null and not v_has_process_state then
      v_expected_next_action := 'SAVE_DURABLE_CHECKPOINT';
    else
      v_expected_next_action := 'CONTINUE_STAGE';
    end if;
  elsif v_stage.lifecycle_status = 'COMPLETE' then
    if v_stage.handoff_gate_state <> 'YES' then
      v_expected_next_action := 'FAIL_CLOSED';
    elsif p_stage_code = 'INTEGRATION' then
      v_expected_next_action := 'AWAIT_EXPLICIT_GO_PUBLISH';
    else
      v_expected_next_action := 'HANDOFF_NEXT_STAGE';
    end if;
  else
    v_expected_next_action := 'FAIL_CLOSED';
  end if;

  if p_exact_next_action is distinct from v_expected_next_action then
    raise exception 'PILOTAGE_EXACT_NEXT_ACTION_MISMATCH' using errcode = '23514';
  end if;
  if v_expected_next_action in ('AWAIT_EXPLICIT_GO_PUBLISH','FAIL_CLOSED') then
    raise exception 'PILOTAGE_ROUTE_NOT_DISPATCHABLE' using errcode = '23514';
  end if;
  if p_requested_operation is distinct from v_expected_next_action then
    raise exception 'PILOTAGE_ROUTE_MISMATCH' using errcode = '23514';
  end if;

  insert into public.orotitan_pilotage_attempts (
    run_id,
    stage_code,
    run_state_version,
    stage_state_version,
    state_fingerprint_sha256,
    requested_operation,
    exact_next_action
  ) values (
    p_run_id,
    p_stage_code,
    p_expected_run_state_version,
    p_expected_stage_state_version,
    p_state_fingerprint_sha256,
    p_requested_operation,
    p_exact_next_action
  )
  on conflict (
    run_id,
    stage_code,
    run_state_version,
    stage_state_version,
    requested_operation
  )
  do nothing
  returning * into v_attempt;

  if found then
    return jsonb_build_object(
      'decision', 'FIRST_ATTEMPT',
      'run_id', v_attempt.run_id,
      'stage_code', v_attempt.stage_code,
      'state_fingerprint_sha256', v_attempt.state_fingerprint_sha256,
      'requested_operation', v_attempt.requested_operation,
      'exact_next_action', v_attempt.exact_next_action,
      'attempt_count', v_attempt.attempt_count,
      'retry_without_reload_allowed', false
    );
  end if;

  select *
    into v_attempt
  from public.orotitan_pilotage_attempts
  where run_id = p_run_id
    and stage_code = p_stage_code
    and run_state_version = p_expected_run_state_version
    and stage_state_version = p_expected_stage_state_version
    and requested_operation = p_requested_operation
  for update;

  if not found then
    raise exception 'PILOTAGE_ATTEMPT_CONFLICT_NOT_FOUND' using errcode = '23514';
  end if;
  if v_attempt.run_state_version <> p_expected_run_state_version
     or v_attempt.stage_state_version <> p_expected_stage_state_version
     or v_attempt.exact_next_action is distinct from p_exact_next_action then
    raise exception 'PILOTAGE_ATTEMPT_IDENTITY_MISMATCH' using errcode = '23514';
  end if;
  if v_attempt.state_fingerprint_sha256 is distinct from p_state_fingerprint_sha256 then
    raise exception 'PILOTAGE_STATE_FINGERPRINT_DRIFT' using errcode = '23514';
  end if;

  update public.orotitan_pilotage_attempts
  set attempt_count = attempt_count + 1,
      last_seen_at = now()
  where attempt_id = v_attempt.attempt_id
  returning * into v_attempt;

  return jsonb_build_object(
    'decision', 'NO_PROGRESS_REPLAY',
    'run_id', v_attempt.run_id,
    'stage_code', v_attempt.stage_code,
    'state_fingerprint_sha256', v_attempt.state_fingerprint_sha256,
    'requested_operation', v_attempt.requested_operation,
    'exact_next_action', v_attempt.exact_next_action,
    'attempt_count', v_attempt.attempt_count,
    'retry_without_reload_allowed', false
  );
end;
$$;

revoke all on function public.register_orotitan_pilotage_attempt(
  uuid,text,bigint,bigint,text,text,text
) from public, anon, authenticated, service_role;
grant execute on function public.register_orotitan_pilotage_attempt(
  uuid,text,bigint,bigint,text,text,text
) to service_role;

commit;
