-- OroTitan single operational reference per issuer, additive and lineage-preserving.
-- References are PINNED selections, not edits to historical Registry rows.
-- Historical run cancellation / publication gates remain independent.
begin;

create table public.orotitan_issuer_reference_runs_v1 (
  issuer_id uuid primary key references public.issuers(issuer_id) on delete restrict,
  run_id uuid not null unique references public.orotitan_runs(run_id) on delete restrict,
  selected_run_state_version bigint not null check (selected_run_state_version>=1),
  selection_policy text not null check
    (selection_policy in ('INITIAL_CANONICAL_BACKFILL_V1','NEW_ISSUER_REFERENCE_V1')),
  selected_at timestamptz not null default now()
);

create index orotitan_issuer_reference_runs_v1_run_idx
  on public.orotitan_issuer_reference_runs_v1(run_id);

alter table public.orotitan_issuer_reference_runs_v1 enable row level security;
revoke all on table public.orotitan_issuer_reference_runs_v1
  from public,anon,authenticated,service_role;
grant select on table public.orotitan_issuer_reference_runs_v1 to service_role;

create function public.reject_orotitan_reference_history_mutation_v1()
returns trigger
language plpgsql
security definer
set search_path = pg_catalog,public
as $function$
begin
  raise exception 'OROTITAN_REFERENCE_SELECTION_IMMUTABLE_V1'
    using errcode='23514';
end;
$function$;

create trigger orotitan_reference_prevent_update_delete_v1
before update or delete on public.orotitan_issuer_reference_runs_v1
for each row execute function public.reject_orotitan_reference_history_mutation_v1();

revoke all on function public.reject_orotitan_reference_history_mutation_v1()
  from public,anon,authenticated,service_role;

-- One-time deterministic backfill; callers supply and must verify exact cohort
-- size. The function does not embed UUIDs from the production dataset.
create function public.bootstrap_orotitan_references_v1(
  p_expected_dossiers integer,
  p_expected_references integer,
  p_expected_published integer
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog,public
as $function$
declare
  v_dossiers integer;
  v_ref_count integer;
  v_published integer;
  v_inserts integer;
begin
  if p_expected_dossiers is null or p_expected_references is null
     or p_expected_published is null
     or p_expected_dossiers < 1 or p_expected_references < 1
     or p_expected_published < 0
     or p_expected_published > p_expected_references
     or p_expected_references > p_expected_dossiers
  then raise exception 'OROTITAN_REFERENCE_INVALID_COHORT' using errcode='22023'; end if;

  -- Serializes initial provisioning and prevents silent rebootstrap.
  perform pg_catalog.pg_advisory_xact_lock(1267651,20261026);
  if exists (select 1 from public.orotitan_issuer_reference_runs_v1)
  then raise exception 'OROTITAN_REFERENCES_ALREADY_INITIALIZED' using errcode='23514'; end if;

  select count(*) into v_dossiers from public.research_dossiers where active is true;
  select count(*) into v_published
    from public.research_dossiers where active is true and current_snapshot_id is not null;
  if v_dossiers<>p_expected_dossiers or v_published<>p_expected_published
  then raise exception 'OROTITAN_REFERENCE_DOSSIER_COHORT_CHANGED' using errcode='40001'; end if;

  -- Do not silently reinterpret or mutate old runs; rank for operational
  -- navigation only. A published dossier always keeps a published run;
  -- an unpublished dossier prefers advanced stage, then methodology and cutoff.
  with candidates as (
    select r.issuer_id,r.run_id,r.state_version,
      row_number() over(partition by r.issuer_id order by
        case when d.current_snapshot_id is not null
          then case when r.run_status='PUBLISHED' and exists (
            select 1 from public.orotitan_run_events e
            where e.run_id=r.run_id and e.event_type='PUBLISH_SUCCEEDED'
              and e.payload->>'snapshot_id'=d.current_snapshot_id::text
          ) then 0 else 1 end
          else case when r.run_status in ('ACTIVE','BLOCKED','READY_TO_PUBLISH') then 0 else 1 end
        end,
        case when d.current_snapshot_id is not null then 0
          else case r.current_stage
            when 'INTEGRATION' then 0 when 'DEEP_DIVE' then 1
            when 'RESEARCH' then 2 else 3 end end,
        case when d.current_snapshot_id is not null then r.data_cutoff end desc nulls last,
        case when d.current_snapshot_id is null then
          string_to_array(r.process_version,'.')::int[] end desc nulls last,
        case when d.current_snapshot_id is null then r.data_cutoff end desc nulls last,
        r.created_at desc,r.run_id
      ) as candidate_rank
    from public.orotitan_runs r
    join public.research_dossiers d on d.dossier_id=r.dossier_id and d.issuer_id=r.issuer_id
    where d.active is true and r.run_status<>'CANCELLED'
  )
  insert into public.orotitan_issuer_reference_runs_v1(
    issuer_id,run_id,selected_run_state_version,selection_policy
  )
  select issuer_id,run_id,state_version,'INITIAL_CANONICAL_BACKFILL_V1'
  from candidates where candidate_rank=1;
  get diagnostics v_inserts = row_count;

  if v_inserts<>p_expected_references
  then raise exception 'OROTITAN_REFERENCE_RUN_COHORT_CHANGED'
      using errcode='40001'; end if;

  if exists(
    select 1 from public.orotitan_issuer_reference_runs_v1 ref
    join public.orotitan_runs r on r.run_id=ref.run_id
    join public.research_dossiers d on d.issuer_id=ref.issuer_id and d.active is true
    where r.issuer_id is distinct from ref.issuer_id
       or r.dossier_id is distinct from d.dossier_id
       or r.run_status='CANCELLED'
       or (d.current_snapshot_id is not null and (
             r.run_status<>'PUBLISHED' or not exists(
               select 1 from public.orotitan_run_events e
               where e.run_id=r.run_id and e.event_type='PUBLISH_SUCCEEDED'
                 and e.payload->>'snapshot_id'=d.current_snapshot_id::text)))
       or (d.current_snapshot_id is null and r.run_status='PUBLISHED')
  ) then raise exception 'OROTITAN_REFERENCE_SELECTION_INVALID' using errcode='23514'; end if;

  select count(*) into v_ref_count from public.orotitan_issuer_reference_runs_v1;
  return jsonb_build_object('active_dossiers',v_dossiers,
    'reference_runs',v_ref_count,'published_dossiers',v_published,
    'selection_policy','INITIAL_CANONICAL_BACKFILL_V1');
end;
$function$;

-- Enrol a future first run only; replacement of an existing pinned selection
-- needs a separate governed successor decision, not a silent overwrite.
create function public.enroll_orotitan_reference_run_v1(
  p_run_id uuid,p_expected_run_state_version bigint
)
returns uuid
language plpgsql
security definer
set search_path = pg_catalog,public
as $function$
declare
  v_run public.orotitan_runs%rowtype;
  v_dossier public.research_dossiers%rowtype;
begin
  if p_run_id is null or p_expected_run_state_version is null
  then raise exception 'OROTITAN_ENROLL_INVALID_REQUEST' using errcode='22023'; end if;
  select * into v_run from public.orotitan_runs where run_id=p_run_id for share;
  if not found or v_run.run_status in ('CANCELLED','PUBLISHED')
     or v_run.state_version<>p_expected_run_state_version
     or v_run.issuer_id is null or v_run.dossier_id is null
  then raise exception 'OROTITAN_ENROLL_INVALID_RUN' using errcode='23514'; end if;
  select * into v_dossier from public.research_dossiers
    where dossier_id=v_run.dossier_id for share;
  if not found or v_dossier.active is not true or
     v_dossier.issuer_id is distinct from v_run.issuer_id or
     v_dossier.current_snapshot_id is not null
  then raise exception 'OROTITAN_ENROLL_INVALID_DOSSIER' using errcode='23514'; end if;
  insert into public.orotitan_issuer_reference_runs_v1
    (issuer_id,run_id,selected_run_state_version,selection_policy)
  values (v_run.issuer_id,p_run_id,p_expected_run_state_version,'NEW_ISSUER_REFERENCE_V1');
  return p_run_id;
end;
$function$;

create view public.orotitan_current_reference_run_v1
with (security_invoker=true)
as
select i.issuer_id,i.display_name as issuer_name,d.dossier_id,
  d.current_snapshot_id,r.run_id,r.run_status,r.current_stage,r.canonical_mode,
  r.data_cutoff,r.state_version,r.created_at,r.process_version,
  r.pilotage_contract_version,ref.selection_policy,ref.selected_at
from public.research_dossiers d
join public.issuers i on i.issuer_id=d.issuer_id
left join public.orotitan_issuer_reference_runs_v1 ref on ref.issuer_id=i.issuer_id
left join public.orotitan_runs r on r.run_id=ref.run_id
where d.active=true;

revoke all on table public.orotitan_current_reference_run_v1
  from public,anon,authenticated,service_role;
grant select on table public.orotitan_current_reference_run_v1 to service_role;

revoke all on function public.bootstrap_orotitan_references_v1(integer,integer,integer)
  from public,anon,authenticated,service_role;
grant execute on function public.bootstrap_orotitan_references_v1(integer,integer,integer)
  to service_role;
revoke all on function public.enroll_orotitan_reference_run_v1(uuid,bigint)
  from public,anon,authenticated,service_role;
grant execute on function public.enroll_orotitan_reference_run_v1(uuid,bigint)
  to service_role;

do $verify$
begin
  if has_function_privilege('anon',
       'public.bootstrap_orotitan_references_v1(integer,integer,integer)','EXECUTE')
     or has_function_privilege('authenticated',
       'public.bootstrap_orotitan_references_v1(integer,integer,integer)','EXECUTE')
     or has_function_privilege('public',
       'public.bootstrap_orotitan_references_v1(integer,integer,integer)','EXECUTE')
     or not has_function_privilege('service_role',
       'public.bootstrap_orotitan_references_v1(integer,integer,integer)','EXECUTE')
  then raise exception 'OROTITAN_REFERENCE_ACCESS_INVALID' using errcode='23514'; end if;
end;
$verify$;
commit;
