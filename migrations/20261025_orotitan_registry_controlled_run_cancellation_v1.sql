-- Issue #427: controlled administrative cancellation of two proven redundant runs.
-- Scope deliberately closed: this is NOT a general-purpose deletion or archival API.
-- Code review + non-production tests required before deployment. No production DML here.
begin;


-- PRODUCTION PREREQUISITE: Pilotage attempts are provisioned outside the
-- historical V1.12 Registry migration chain. Refuse deployment if missing.
-- No replacement table is fabricated in production.
do $dependency$
declare
  v_columns text[];
begin
  if to_regclass('public.orotitan_pilotage_attempts') is null then
    raise exception 'CLEANUP_PILOTAGE_ATTEMPTS_DEPENDENCY_MISSING'
      using errcode='42P01';
  end if;

  select array_agg(attname || ':' ||
        pg_catalog.format_type(atttypid,atttypmod) || ':' ||
        attnotnull::text order by attnum)
  into v_columns
  from pg_catalog.pg_attribute
  where attrelid='public.orotitan_pilotage_attempts'::regclass
    and attnum>0 and not attisdropped;

  if v_columns is distinct from array[
    'attempt_id:uuid:true',
    'run_id:uuid:true',
    'stage_code:text:true',
    'run_state_version:bigint:true',
    'stage_state_version:bigint:true',
    'state_fingerprint_sha256:text:true',
    'requested_operation:text:true',
    'exact_next_action:text:true',
    'first_seen_at:timestamp with time zone:true'
  ]::text[] then
    raise exception 'CLEANUP_PILOTAGE_ATTEMPTS_SCHEMA_MISMATCH'
      using errcode='23514';
  end if;

  if not exists (
    select 1
    from pg_catalog.pg_constraint c
    where c.conrelid='public.orotitan_pilotage_attempts'::regclass
      and c.contype='f'
      and pg_catalog.pg_get_constraintdef(c.oid) =
        'FOREIGN KEY (run_id) REFERENCES orotitan_runs(run_id) ON DELETE RESTRICT'
  ) then
    raise exception 'CLEANUP_PILOTAGE_ATTEMPTS_RUN_FK_MISSING'
      using errcode='23514';
  end if;
end;
$dependency$;

create or replace function public.cancel_orotitan_redundant_run_v1(
  p_run_id uuid,
  p_expected_state_version bigint,
  p_replacement_run_id uuid,
  p_reason_code text,
  p_idempotency_key text,
  p_request_fingerprint_sha256 text
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $function$
declare
  v_target public.orotitan_runs%rowtype;
  v_replacement public.orotitan_runs%rowtype;
  v_stage public.orotitan_run_stages%rowtype;
  v_existing jsonb;
  v_event_id uuid;
  v_current_snapshot_id uuid;
  v_artifact_count integer;
  v_expected_key text;
  v_expected_reason text;
begin
  if p_run_id is null or p_replacement_run_id is null
     or p_expected_state_version is null
     or p_idempotency_key is null or length(btrim(p_idempotency_key))=0
     or p_request_fingerprint_sha256 is null
     or p_request_fingerprint_sha256 !~ '^[0-9a-f]{64}$'
  then
    raise exception 'CLEANUP_REQUEST_INVALID' using errcode='22023';
  end if;

  -- Exact allowlist: no other run can be cancelled by this one-off procedure.
  if p_run_id = 'd5cd803b-f959-4e41-ba55-5fbfc2a7d245'::uuid then
    v_expected_reason := 'REDUNDANT_HISTORICAL_REFRESH_ALREADY_PUBLISHED_V2';
    if p_replacement_run_id <> 'e41a8fe0-29cb-45e3-8bca-b1f19ac67d12'::uuid
       or p_expected_state_version <> 3 then
      raise exception 'CLEANUP_PAIR_OR_VERSION_NOT_AUTHORIZED' using errcode='23514';
    end if;
  elsif p_run_id = 'd56a0bc9-4e80-48b3-b326-74d1fea15e63'::uuid then
    v_expected_reason := 'EMPTY_PARALLEL_METHOD_V3_1_SUCCESSOR_REDUNDANT';
    if p_replacement_run_id <> 'a4cf9002-b52d-4440-8902-adc70bd777dd'::uuid
       or p_expected_state_version <> 2 then
      raise exception 'CLEANUP_PAIR_OR_VERSION_NOT_AUTHORIZED' using errcode='23514';
    end if;
  else
    raise exception 'CLEANUP_RUN_NOT_ALLOWLISTED' using errcode='23514';
  end if;

  if p_reason_code is distinct from v_expected_reason then
    raise exception 'CLEANUP_REASON_NOT_AUTHORIZED' using errcode='23514';
  end if;

  -- Idempotency is checked before the CAS, but an old different-key request
  -- may never reopen a terminal run. Each event remains append-only.
  v_existing := public.orotitan_existing_event(
    p_run_id, p_idempotency_key, p_request_fingerprint_sha256
  );
  if v_existing is not null then
    select * into v_target from public.orotitan_runs
      where run_id=p_run_id;
    if not found
       or v_target.run_status <> 'CANCELLED'
       or v_target.cancelled_at is null
       or v_existing->>'event_type' <> 'RUN_CANCELLED'
       or v_existing->'payload'->>'reason_code' is distinct from p_reason_code
       or v_existing->'payload'->>'replacement_run_id' is distinct from p_replacement_run_id::text
    then
      raise exception 'CLEANUP_IDEMPOTENCY_STATE_CONFLICT' using errcode='23514';
    end if;
    return jsonb_build_object(
      'run_id',p_run_id,'run_status','CANCELLED',
      'state_version',v_target.state_version,
      'event_id',v_existing->>'event_id','idempotent_replay',true
    );
  end if;

  select * into v_target from public.orotitan_runs
    where run_id=p_run_id for update;
  if not found then raise exception 'CLEANUP_TARGET_MISSING' using errcode='P0002'; end if;
  if v_target.run_status <> 'ACTIVE'
     or v_target.current_stage <> 'RESEARCH'
     or v_target.state_version <> p_expected_state_version
     or v_target.published_at is not null
     or v_target.cancelled_at is not null
     or v_target.issuer_id is null
     or v_target.security_id is null
     or v_target.dossier_id is null
  then raise exception 'CLEANUP_TARGET_CAS_OR_IDENTITY_CONFLICT' using errcode='40001'; end if;

  select * into v_replacement from public.orotitan_runs
    where run_id=p_replacement_run_id for share;
  if not found then raise exception 'CLEANUP_REPLACEMENT_MISSING' using errcode='23514'; end if;
  if v_replacement.run_id = v_target.run_id
     or v_replacement.issuer_id is distinct from v_target.issuer_id
     or v_replacement.security_id is distinct from v_target.security_id
     or v_replacement.dossier_id is distinct from v_target.dossier_id
  then raise exception 'CLEANUP_IDENTITY_MISMATCH' using errcode='23514'; end if;

  select * into v_stage from public.orotitan_run_stages
    where run_id=p_run_id and stage_code='RESEARCH' for share;
  if not found then raise exception 'CLEANUP_RESEARCH_STAGE_MISSING' using errcode='23514'; end if;
  if exists (select 1 from public.orotitan_run_stages
             where run_id=p_run_id and stage_code<>'RESEARCH')
     or exists (select 1 from public.orotitan_runs
                where parent_run_id=p_run_id)
     or exists (select 1 from public.orotitan_artifact_edges
                where parent_run_id=p_run_id or child_run_id=p_run_id)
     or exists (select 1 from public.orotitan_pilotage_attempts
                where run_id=p_run_id)
     or exists (select 1 from public.orotitan_run_events
                where run_id=p_run_id and event_type in
                ('PUBLISH_AUTHORIZED','PUBLISH_SUCCEEDED','RUN_CANCELLED'))
  then raise exception 'CLEANUP_DOWNSTREAM_OR_PUBLISH_DEPENDENCY' using errcode='23514'; end if;

  select count(*) into v_artifact_count from public.orotitan_artifacts
    where run_id=p_run_id;

  select current_snapshot_id into v_current_snapshot_id
    from public.research_dossiers
    where dossier_id=v_target.dossier_id for share;
  if not found then raise exception 'CLEANUP_DOSSIER_MISSING' using errcode='23514'; end if;

  if p_reason_code='REDUNDANT_HISTORICAL_REFRESH_ALREADY_PUBLISHED_V2' then
    if v_target.process_version <> '1.0'
       or v_target.pilotage_contract_version <> '1.0.1'
       or v_target.canonical_mode <> 'REFRESH'
       or v_target.run_type <> 'REFRESH'
       or v_target.data_cutoff <> date '2026-09-15'
       or v_target.parent_run_id is not null
       or v_target.baseline_snapshot_id is distinct from
          '960fa825-adad-4bea-bd43-c45847c4c78c'::uuid
       or v_stage.lifecycle_status <> 'COMPLETE'
       or v_stage.handoff_gate_state <> 'YES'
       or v_artifact_count <> 8
       or v_replacement.run_status <> 'PUBLISHED'
       or v_replacement.published_at is null
       or v_replacement.process_version <> '2.0'
       or v_replacement.run_type <> 'REFRESH'
       or v_replacement.data_cutoff is distinct from v_target.data_cutoff
       or v_replacement.baseline_snapshot_id is distinct from v_target.baseline_snapshot_id
       or v_current_snapshot_id is distinct from
          'e42a1396-357a-42f5-ae0c-5e750f321180'::uuid
       or not exists(
         select 1 from public.research_snapshots s
         where s.snapshot_id=v_current_snapshot_id
           and s.dossier_id=v_target.dossier_id
           and s.issuer_id=v_target.issuer_id
           and s.security_id=v_target.security_id
       )
    then raise exception 'CLEANUP_QUALYS_REPLACEMENT_NOT_PROVEN' using errcode='23514'; end if;
  elsif p_reason_code='EMPTY_PARALLEL_METHOD_V3_1_SUCCESSOR_REDUNDANT' then
    if v_target.process_version <> '3.1'
       or v_target.pilotage_contract_version <> '3.1'
       or v_target.canonical_mode <> 'ANALYZE'
       or v_target.run_type <> 'INITIAL'
       or v_target.data_cutoff <> date '2026-09-22'
       or v_target.parent_run_id is distinct from
          '76ee7eb1-8658-4038-83bf-e2c951db731a'::uuid
       or v_target.contract_set_sha256 is distinct from
          '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2'
       or v_stage.lifecycle_status <> 'IN_PROGRESS'
       or v_stage.active_manifest_artifact_id is not null
       or v_artifact_count <> 0
       or v_replacement.process_version <> '3.1'
       or v_replacement.pilotage_contract_version <> '3.1'
       or v_replacement.contract_set_sha256 is distinct from v_target.contract_set_sha256
       or v_replacement.data_cutoff is distinct from v_target.data_cutoff
       or v_replacement.parent_run_id is distinct from v_target.parent_run_id
       or v_replacement.run_status not in ('ACTIVE','BLOCKED')
       or v_replacement.current_stage <> 'DEEP_DIVE'
       or not exists(
         select 1 from public.orotitan_run_stages s
         where s.run_id=v_replacement.run_id and s.stage_code='DEEP_DIVE'
       )
       or not exists(
         select 1 from public.orotitan_artifacts a where a.run_id=v_replacement.run_id
       )
       or v_current_snapshot_id is not null
    then raise exception 'CLEANUP_VEOLIA_PARALLEL_REPLACEMENT_NOT_PROVEN' using errcode='23514'; end if;
  end if;

  update public.orotitan_runs
  set run_status='CANCELLED',
      cancelled_at=now(),
      state_version=state_version+1
  where run_id=p_run_id and state_version=p_expected_state_version
    and run_status='ACTIVE';
  if not found then raise exception 'CLEANUP_CAS_UPDATE_FAILED' using errcode='40001'; end if;

  v_event_id := public.orotitan_insert_event(
    p_run_id,'RESEARCH','RUN_CANCELLED',
    p_idempotency_key,p_request_fingerprint_sha256,'PILOTAGE',
    jsonb_build_object(
      'reason_code',p_reason_code,
      'replacement_run_id',p_replacement_run_id,
      'previous_state_version',p_expected_state_version,
      'new_state_version',p_expected_state_version+1,
      'preserve_artifact_history',true,
      'physical_delete',false,
      'canonical_pointer_unchanged',true,
      'issue','robzer13/indice_nexus#427'
    )
  );
  if v_event_id is null then
    raise exception 'CLEANUP_AUDIT_WRITE_FAILED' using errcode='23514';
  end if;
  return jsonb_build_object(
    'run_id',p_run_id,
    'run_status','CANCELLED',
    'state_version',p_expected_state_version+1,
    'event_id',v_event_id,
    'replacement_run_id',p_replacement_run_id,
    'idempotent_replay',false
  );
end;
$function$;

revoke all on function public.cancel_orotitan_redundant_run_v1(uuid,bigint,uuid,text,text,text)
  from public, anon, authenticated, service_role;
grant execute on function public.cancel_orotitan_redundant_run_v1(uuid,bigint,uuid,text,text,text)
  to service_role;

-- Fail deployment if EXECUTE accidentally reaches public users.
do $verify$
declare
  v_sig regprocedure :=
    'public.cancel_orotitan_redundant_run_v1(uuid,bigint,uuid,text,text,text)'::regprocedure;
begin
  if not (select prosecdef
          and proconfig @> array['search_path=pg_catalog, public']::text[]
          from pg_catalog.pg_proc where oid=v_sig)
     or has_function_privilege('public',v_sig,'EXECUTE')
     or has_function_privilege('anon',v_sig,'EXECUTE')
     or has_function_privilege('authenticated',v_sig,'EXECUTE')
     or not has_function_privilege('service_role',v_sig,'EXECUTE')
  then raise exception 'CLEANUP_RPC_SECURITY_BOUNDARY_INVALID'; end if;
end;
$verify$;

commit;
