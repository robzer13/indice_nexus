-- OroTitan J0 Control Tower — SHADOW-only, governance evidence journal.
-- Deliberately separate from orotitan_runs, stage/artifact/snapshot and
-- methodological contracts. Never deploy this migration to production.
begin;

create table if not exists public.orotitan_j0_milestones_v1 (
  milestone_code text primary key check (milestone_code ~ '^J[0-8]$'),
  ordinal smallint not null unique check (ordinal between 0 and 8),
  title text not null check (length(btrim(title)) between 5 and 180),
  objective text not null check (length(btrim(objective)) between 15 and 1000),
  gate_criteria jsonb not null check (jsonb_typeof(gate_criteria)='array' and jsonb_array_length(gate_criteria)>=3),
  initial_status text not null check (initial_status in ('NON_DEMARRE','EN_COURS','BLOQUE','TESTE','ACCEPTE')),
  scope text not null default 'SHADOW_ONLY' check (scope='SHADOW_ONLY'),
  plan_version text not null default 'OROTITAN_RUN_FACTORY_PLAN_DIRECTEUR_JALONS_V1_20261031',
  created_at timestamptz not null default now()
);
create table if not exists public.orotitan_j0_events_v1 (
  event_id uuid primary key default gen_random_uuid(),
  event_key text not null unique check (length(event_key) between 14 and 160),
  milestone_code text not null references public.orotitan_j0_milestones_v1(milestone_code) on delete restrict,
  event_type text not null check (event_type in (
   'EVIDENCE_RECORDED','BLOCKER_OPENED','BLOCKER_RESOLVED','GATE_REVIEW_REQUESTED',
   'ROADMAP_APPROVED','J0_STARTED')),
  title text not null check(length(btrim(title)) between 5 and 200),
  evidence_ref text,
  run_id uuid references public.orotitan_runs(run_id) on delete restrict,
  operator_details jsonb not null default '{}'::jsonb check(jsonb_typeof(operator_details)='object'),
  verification_state text not null default 'REPORTED'
    check (verification_state in ('REPORTED','VERIFIED_DIRECT_REGISTRY_READBACK')),
  verified_snapshot jsonb,
  actor text not null default 'AUTHORIZED_SHADOW_OPERATOR'
    check (actor='AUTHORIZED_SHADOW_OPERATOR'),
  created_at timestamptz not null default now(),
  constraint j0_verified_snapshot_consistency check (
   (verification_state='REPORTED' and verified_snapshot is null)
   or (verification_state='VERIFIED_DIRECT_REGISTRY_READBACK' and
       verified_snapshot is not null and run_id is not null))
);
create index if not exists orotitan_j0_events_milestone_created_idx
 on public.orotitan_j0_events_v1(milestone_code,created_at desc,event_id);
create index if not exists orotitan_j0_events_run_idx
 on public.orotitan_j0_events_v1(run_id) where run_id is not null;

create or replace function public.orotitan_j0_deny_history_change_v1()
returns trigger language plpgsql set search_path=pg_catalog,public as $$
begin
 raise exception 'J0_APPEND_ONLY_HISTORY';
end $$;
drop trigger if exists orotitan_j0_milestones_append_only on public.orotitan_j0_milestones_v1;
create trigger orotitan_j0_milestones_append_only before update or delete
 on public.orotitan_j0_milestones_v1
 for each row execute function public.orotitan_j0_deny_history_change_v1();
drop trigger if exists orotitan_j0_events_append_only on public.orotitan_j0_events_v1;
create trigger orotitan_j0_events_append_only before update or delete
 on public.orotitan_j0_events_v1
 for each row execute function public.orotitan_j0_deny_history_change_v1();

alter table public.orotitan_j0_milestones_v1 enable row level security;
alter table public.orotitan_j0_events_v1 enable row level security;
revoke all on public.orotitan_j0_milestones_v1 from public,anon,authenticated,service_role;
revoke all on public.orotitan_j0_events_v1 from public,anon,authenticated,service_role;
grant select on public.orotitan_j0_milestones_v1 to service_role;
grant select on public.orotitan_j0_events_v1 to service_role;

-- One restricted, idempotent, event-only control path.
-- It cannot accept a milestone, start a stage or modify an existing event.
create or replace function public.orotitan_j0_record_event_v1(
 p_event_key text,p_milestone_code text,p_event_type text,p_title text,
 p_evidence_ref text,p_run_id uuid,p_operator_details jsonb
) returns jsonb
language plpgsql security invoker set search_path=pg_catalog,public as $$
declare
 v_old public.orotitan_j0_events_v1%rowtype;
 v_created public.orotitan_j0_events_v1%rowtype;
 v_verify text:='REPORTED';
 v_verified jsonb;
 v_run public.orotitan_runs%rowtype;
 v_issue_state text;
begin
 if p_event_key is null or
    p_event_key !~ '^orotitan:j0:[a-zA-Z0-9:_-]{3,145}$' or
    p_milestone_code !~ '^J[0-8]$' or
    p_event_type not in ('EVIDENCE_RECORDED','BLOCKER_OPENED',
                         'BLOCKER_RESOLVED','GATE_REVIEW_REQUESTED') or
    p_title is null or length(btrim(p_title)) not between 5 and 200 or
    p_operator_details is null or jsonb_typeof(p_operator_details)<>'object' or
    length(p_operator_details::text)>3000 or
    p_evidence_ref is null or length(btrim(p_evidence_ref)) not between 8 and 400 or
    p_evidence_ref ~ '[[:cntrl:]]' then
  raise exception 'J0_EVENT_INVALID';
 end if;
 if not exists (select 1 from public.orotitan_j0_milestones_v1
                where milestone_code=p_milestone_code)
 then raise exception 'J0_UNKNOWN_MILESTONE';end if;
 perform pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('orotitan:j0:event:'||p_event_key,0));
 select * into v_old from public.orotitan_j0_events_v1 where event_key=p_event_key;
 if found then
  if v_old.milestone_code<>p_milestone_code or v_old.event_type<>p_event_type
     or v_old.title<>p_title
     or v_old.evidence_ref is distinct from p_evidence_ref
     or v_old.run_id is distinct from p_run_id
     or v_old.operator_details<>p_operator_details then
   raise exception 'J0_IDEMPOTENCY_CONFLICT';
  end if;
  return jsonb_build_object('event_id',v_old.event_id,'status','REPLAYED',
      'verification_state',v_old.verification_state);
 end if;
 if p_event_type='EVIDENCE_RECORDED' then
   if p_evidence_ref like 'registry:run:%' then
    if p_run_id is null or p_evidence_ref<>'registry:run:'||p_run_id::text
    then raise exception 'J0_REGISTRY_READBACK_KEY_INVALID';end if;
    select * into v_run from public.orotitan_runs where run_id=p_run_id;
    if not found then raise exception 'J0_RUN_NOT_FOUND';end if;
    v_verify:='VERIFIED_DIRECT_REGISTRY_READBACK';
    v_verified:=jsonb_build_object(
     'run_id',v_run.run_id,'issuer_id',v_run.issuer_id,
     'security_id',v_run.security_id,'dossier_id',v_run.dossier_id,
     'run_status',v_run.run_status,'current_stage',v_run.current_stage,
     'state_version',v_run.state_version,'data_cutoff',v_run.data_cutoff,
     'contract_set_sha256',v_run.contract_set_sha256,
     'event_count',(select count(*) from public.orotitan_run_events
                    where run_id=p_run_id),
     'stage_count',(select count(*) from public.orotitan_run_stages
                    where run_id=p_run_id),
     'artifact_count',(select count(*) from public.orotitan_artifacts
                       where run_id=p_run_id),
     'p0e_receipts',(select count(*) from
                      public.orotitan_p0e_run_admission_receipts_v1
                      where run_id=p_run_id),
     'observed_at',clock_timestamp());
   end if;
 elsif p_event_type in ('BLOCKER_OPENED','BLOCKER_RESOLVED') then
   if p_evidence_ref !~ '^issue:[a-z0-9][a-z0-9_-]{3,120}$' then
    raise exception 'J0_ISSUE_KEY_INVALID';
   end if;
   perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('orotitan:j0:issue:'||p_evidence_ref,0));
   select event_type into v_issue_state from public.orotitan_j0_events_v1
    where evidence_ref=p_evidence_ref and milestone_code=p_milestone_code
      and event_type in ('BLOCKER_OPENED','BLOCKER_RESOLVED')
    order by created_at desc,event_id desc limit 1;
   if (p_event_type='BLOCKER_OPENED' and v_issue_state='BLOCKER_OPENED')
    or (p_event_type='BLOCKER_RESOLVED'
        and v_issue_state is distinct from 'BLOCKER_OPENED') then
    raise exception 'J0_ISSUE_STATE_CONFLICT';
   end if;
 end if;
 insert into public.orotitan_j0_events_v1(
  event_key,milestone_code,event_type,title,evidence_ref,run_id,
  operator_details,verification_state,verified_snapshot)
 values(p_event_key,p_milestone_code,p_event_type,p_title,p_evidence_ref,
        p_run_id,p_operator_details,v_verify,v_verified)
 returning * into v_created;
 return jsonb_build_object('event_id',v_created.event_id,'status','CREATED',
  'verification_state',v_created.verification_state);
end $$;
revoke all on function public.orotitan_j0_record_event_v1(
 text,text,text,text,text,uuid,jsonb) from public,anon,authenticated;
grant execute on function public.orotitan_j0_record_event_v1(
 text,text,text,text,text,uuid,jsonb) to service_role;
-- The invoked function uses the service_role's INSERT privileges, not
-- definer privilege escalation.
grant insert on public.orotitan_j0_events_v1 to service_role;

insert into public.orotitan_j0_milestones_v1
 (milestone_code,ordinal,title,objective,gate_criteria,initial_status)
 values
 ('J0',0,'Cockpit et gouvernance',
  'Une vue du Registry et des preuves par jalon, sans acceptation automatique.',
  '["Etats relus du Registry SHADOW","Registre J0-J8 persistant","Preuves et décisions immuables","Exceptions traçables","Contrôles d’accès et négatifs","Démonstration interface authentifiée","Relecture indépendante du reçu J0","Acceptation explicite distincte"]'::jsonb,'EN_COURS'),
 ('J1',1,'Run Factory P0 de bout en bout',
  'Commande utilisateur authentifiée vers un run durable, idempotent et réutilisable.',
  '["L’Oréal HTTP REPLAYED","Second émetteur CREATED","401/403 et RLS","Concurrence et doublons","Reprise d’erreur","Aucun stage ou snapshot","Lecture Registry indépendante","Acceptation humaine"]'::jsonb,'EN_COURS'),
 ('J2',2,'Source Vault et acquisition',
  'Acquisition, stockage privé et relecture d’octets officiels horodatés.',
  '["Deux sources officielles ou BLOCKED tracé","SHA-256 et rehash","MIME taille cutoff","Provenance et reçu","RLS et URLs refusées","Reprise et idempotence","Second émetteur","Acceptation"]'::jsonb,'NON_DEMARRE'),
 ('J3',3,'Extraction réutilisable',
  'Parsing versionné une seule fois par source, avec reprise et index localisé.',
  '["PDF et XBRL","Cache déterministe","Pages et tables localisées","Reprise interruption","Hash d’entrée vérifié","ASML et L’Oréal","Référence annotée","Acceptation"]'::jsonb,'NON_DEMARRE'),
 ('J4',4,'Evidence et QA',
  'Claims financiers typés, rapprochés, traçables et admis seulement avec preuve.',
  '["Unité période norme","Evidence ledger","Conflict ledger","Gap register","Aucun chiffre inventé","Cas négatifs","Deux émetteurs","Acceptation"]'::jsonb,'NON_DEMARRE'),
 ('J5',5,'Research orchestré',
  'Recherche contractuelle pilotée par gaps et exceptions, sans score Research.',
  '["GO RESEARCH distinct","8 catégories d’artefacts","Evidence Ledger","Lock et manifest","Reprise","Stage gate","Aucun score Research","Acceptation"]'::jsonb,'NON_DEMARRE'),
 ('J6',6,'Deep Dive contrôlé',
  'Calculs, valorisation, contradiction, revue humaine et certification.',
  '["Modèles tracés","ROIC FCF DCF","Red Team","Blockers matériels","Sensibilités","Deux émetteurs","Aucune publication","Acceptation"]'::jsonb,'NON_DEMARRE'),
 ('J7',7,'Integration et prépublication',
  'Mapping I2/I3-B et candidats snapshots sans publication silencieuse.',
  '["Schéma canonique","Réconciliation I2","Contrôle I3-B","Snapshot candidat","Aucun pointer mutation sans GO","Historique intact","Cas négatifs","Acceptation"]'::jsonb,'NON_DEMARRE'),
 ('J8',8,'Service multi-issuer',
  'Répétabilité opérationnelle sans logique par entreprise, observabilité et reprise.',
  '["L’Oréal et ASML","Zéro fichier spécifique","Idempotence","Reprise et cache","Monitoring","Logs sans secrets","Tests SHADOW complets","Acceptation"]'::jsonb,'NON_DEMARRE')
 on conflict (milestone_code) do nothing;

-- Authorized human decision captured without claiming gate acceptance.
insert into public.orotitan_j0_events_v1(
 event_key,milestone_code,event_type,title,evidence_ref,operator_details)
 values
 ('orotitan:j0:roadmap-approved-20261031','J0','ROADMAP_APPROVED',
  'Plan directeur J0 à J8 validé','plan:OROTITAN_RUN_FACTORY_PLAN_DIRECTEUR_JALONS_V1_20261031',
  '{"scope":"SHADOW_ONLY","acceptance":"NOT_GRANTED","next":"J0"}'::jsonb),
 ('orotitan:j0:start-authorized-20261031','J0','J0_STARTED',
  'GO J0 autorisé, aucun stage Research','decision:GO_J0_SHADOW_20261031',
  '{"production_changes":false,"stage_research_started":false}'::jsonb)
 on conflict(event_key) do nothing;
commit;
