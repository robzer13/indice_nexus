-- SHADOW PRODUCT ONLY — NOT an OroTitan Registry migration or publication writer.
-- Append-only investigations allow safe retry without redoing prior attempts.
create table if not exists public.orotitan_product_recovery_attempts_shadow_v1 (
  run_id uuid not null,
  stage_code text not null,
  run_state_version bigint not null check(run_state_version >= 0),
  stage_state_version bigint not null check(stage_state_version >= 0),
  blocker_code text not null check(length(blocker_code) between 1 and 100),
  strategy text not null check(strategy in (
    'REUSE_PRIMARY_ARTIFACTS','RECONCILE_SHARE_COUNT',
    'RECONCILE_HISTORICAL_PRICE','REBUILD_ROIC_OWNER_EARNINGS','MANUAL_REVIEW')),
  outcome text not null check(outcome in ('CANDIDATE_RECORDED','NO_PROGRESS','REQUIRES_AUTHORITY')),
  detail text not null check(length(detail) <= 1500),
  evidence_refs jsonb not null default '[]'::jsonb
    check(jsonb_typeof(evidence_refs)='array' and jsonb_array_length(evidence_refs) <= 50),
  created_at timestamptz not null default now(),
  primary key(run_id, stage_code, run_state_version, stage_state_version, blocker_code, strategy)
);
alter table public.orotitan_product_recovery_attempts_shadow_v1 enable row level security;
revoke all on public.orotitan_product_recovery_attempts_shadow_v1 from public, anon, authenticated;
grant select, insert on public.orotitan_product_recovery_attempts_shadow_v1 to service_role;
-- Prevent mutable edits even by backend service-role on the journal API.
create or replace function public.orotitan_shadow_recovery_journal_immutable()
returns trigger language plpgsql as $$
begin raise exception 'OROTITAN_SHADOW_RECOVERY_JOURNAL_APPEND_ONLY'; end; $$;
drop trigger if exists orotitan_shadow_recovery_no_update on public.orotitan_product_recovery_attempts_shadow_v1;
create trigger orotitan_shadow_recovery_no_update before update or delete
on public.orotitan_product_recovery_attempts_shadow_v1
for each row execute function public.orotitan_shadow_recovery_journal_immutable();
comment on table public.orotitan_product_recovery_attempts_shadow_v1 is
'Non-authoritative SHADOW-only product investigation journal; not Registry admission, not certification, not publication';
