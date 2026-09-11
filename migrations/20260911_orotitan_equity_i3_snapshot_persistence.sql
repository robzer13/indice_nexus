-- OROTitan Equity Research V1 - I3-A canonical snapshot persistence.
-- This migration creates an immutable structured projection, not analytical truth.

begin;

create table if not exists public.research_snapshots (
  snapshot_id uuid primary key,
  dossier_id uuid not null,
  issuer_id uuid not null,
  security_id uuid not null,
  report_id text not null,
  execution_mode text not null,
  data_cutoff date not null,
  calculation_date date not null,
  report_version text not null,
  method_version text not null,
  calculation_version text not null,
  evidence_ledger_version text not null,
  contract_version text not null default '04_SCREENER_SCHEMA_V1',
  schema_version text not null default '1.0.0',
  canonical_payload jsonb not null,
  created_at timestamptz not null default now(),
  constraint research_snapshots_report_id_check
    check (length(report_id) between 1 and 256),
  constraint research_snapshots_version_strings_check
    check (
      length(report_version) > 0
      and length(method_version) > 0
      and length(calculation_version) > 0
      and length(evidence_ledger_version) > 0
    ),
  constraint research_snapshots_execution_mode_check
    check (execution_mode in ('DISCOVER', 'ANALYZE', 'DISCOVER + ANALYZE', 'REFRESH', 'ACTIVATION CHECK')),
  constraint research_snapshots_contract_version_check
    check (contract_version = '04_SCREENER_SCHEMA_V1'),
  constraint research_snapshots_schema_version_check
    check (schema_version = '1.0.0'),
  constraint research_snapshots_payload_object_check
    check (jsonb_typeof(canonical_payload) = 'object'),
  constraint research_snapshots_payload_lock_check
    check (
      canonical_payload ? 'snapshot_id'
      and jsonb_typeof(canonical_payload->'snapshot_id') = 'string'
      and canonical_payload->>'snapshot_id' = snapshot_id::text
      and canonical_payload ? 'report_id'
      and jsonb_typeof(canonical_payload->'report_id') = 'string'
      and canonical_payload->>'report_id' = report_id
      and canonical_payload ? 'issuer_id'
      and jsonb_typeof(canonical_payload->'issuer_id') = 'string'
      and canonical_payload->>'issuer_id' = issuer_id::text
      and canonical_payload ? 'security_id'
      and jsonb_typeof(canonical_payload->'security_id') = 'string'
      and canonical_payload->>'security_id' = security_id::text
      and canonical_payload ? 'execution_mode'
      and jsonb_typeof(canonical_payload->'execution_mode') = 'string'
      and canonical_payload->>'execution_mode' = execution_mode
      and canonical_payload ? 'data_lock'
      and jsonb_typeof(canonical_payload->'data_lock') = 'object'
      and (canonical_payload->'data_lock') ? 'data_cutoff'
      and jsonb_typeof(canonical_payload->'data_lock'->'data_cutoff') = 'string'
      and canonical_payload->'data_lock'->>'data_cutoff' = data_cutoff::text
      and (canonical_payload->'data_lock') ? 'calculation_date'
      and jsonb_typeof(canonical_payload->'data_lock'->'calculation_date') = 'string'
      and canonical_payload->'data_lock'->>'calculation_date' = calculation_date::text
      and canonical_payload ? 'versions'
      and jsonb_typeof(canonical_payload->'versions') = 'object'
      and (canonical_payload->'versions') ? 'report_version'
      and jsonb_typeof(canonical_payload->'versions'->'report_version') = 'string'
      and canonical_payload->'versions'->>'report_version' = report_version
      and (canonical_payload->'versions') ? 'method_version'
      and jsonb_typeof(canonical_payload->'versions'->'method_version') = 'string'
      and canonical_payload->'versions'->>'method_version' = method_version
      and (canonical_payload->'versions') ? 'calculation_version'
      and jsonb_typeof(canonical_payload->'versions'->'calculation_version') = 'string'
      and canonical_payload->'versions'->>'calculation_version' = calculation_version
      and (canonical_payload->'versions') ? 'evidence_ledger_version'
      and jsonb_typeof(canonical_payload->'versions'->'evidence_ledger_version') = 'string'
      and canonical_payload->'versions'->>'evidence_ledger_version' = evidence_ledger_version
    )
);

do $$
declare
  expected text[] := array[
    'snapshot_id:uuid:true', 'dossier_id:uuid:true', 'issuer_id:uuid:true',
    'security_id:uuid:true', 'report_id:text:true', 'execution_mode:text:true',
    'data_cutoff:date:true', 'calculation_date:date:true', 'report_version:text:true',
    'method_version:text:true', 'calculation_version:text:true',
    'evidence_ledger_version:text:true', 'contract_version:text:true',
    'schema_version:text:true', 'canonical_payload:jsonb:true',
    'created_at:timestamp with time zone:true'
  ];
  actual text[];
begin
  select array_agg(a.attname || ':' || format_type(a.atttypid, a.atttypmod) || ':' || a.attnotnull::text order by a.attnum)
    into actual
  from pg_catalog.pg_attribute a
  where a.attrelid = 'public.research_snapshots'::regclass and a.attnum > 0 and not a.attisdropped;
  if actual <> expected then
    raise exception 'research_snapshots has incompatible columns: %', actual;
  end if;
  if not exists (
    select 1 from pg_catalog.pg_constraint c
    where c.conrelid = 'public.research_snapshots'::regclass and c.contype = 'p'
      and c.conkey = array[(select attnum from pg_catalog.pg_attribute where attrelid = c.conrelid and attname = 'snapshot_id')]::smallint[]
  ) then raise exception 'research_snapshots primary key is incompatible'; end if;
end $$;

create unique index if not exists research_snapshots_snapshot_dossier_key
  on public.research_snapshots (snapshot_id, dossier_id);

do $$
begin
  if not exists (
    select 1 from pg_catalog.pg_constraint c
    where c.conrelid = 'public.research_dossiers'::regclass and c.contype = 'u'
      and c.conkey = array[
        (select attnum from pg_catalog.pg_attribute where attrelid = c.conrelid and attname = 'dossier_id'),
        (select attnum from pg_catalog.pg_attribute where attrelid = c.conrelid and attname = 'issuer_id')
      ]::smallint[]
  ) then
    alter table public.research_dossiers add constraint research_dossiers_dossier_issuer_key unique (dossier_id, issuer_id);
  end if;
  if not exists (
    select 1 from pg_catalog.pg_constraint c
    where c.conrelid = 'public.securities'::regclass and c.contype = 'u'
      and c.conkey = array[
        (select attnum from pg_catalog.pg_attribute where attrelid = c.conrelid and attname = 'security_id'),
        (select attnum from pg_catalog.pg_attribute where attrelid = c.conrelid and attname = 'issuer_id')
      ]::smallint[]
  ) then
    alter table public.securities add constraint securities_security_issuer_key unique (security_id, issuer_id);
  end if;
end $$;

do $$
begin
  if not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_snapshots'::regclass and conname = 'research_snapshots_dossier_issuer_fkey') then
    alter table public.research_snapshots add constraint research_snapshots_dossier_issuer_fkey
      foreign key (dossier_id, issuer_id) references public.research_dossiers (dossier_id, issuer_id) on delete restrict;
  end if;
  if not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_snapshots'::regclass and conname = 'research_snapshots_security_issuer_fkey') then
    alter table public.research_snapshots add constraint research_snapshots_security_issuer_fkey
      foreign key (security_id, issuer_id) references public.securities (security_id, issuer_id) on delete restrict;
  end if;
  if not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_dossiers'::regclass and conname = 'research_dossiers_current_snapshot_fkey') then
    alter table public.research_dossiers add constraint research_dossiers_current_snapshot_fkey
      foreign key (current_snapshot_id, dossier_id) references public.research_snapshots (snapshot_id, dossier_id) on delete restrict;
  end if;
end $$;

create index if not exists research_snapshots_dossier_created_at_idx
  on public.research_snapshots (dossier_id, created_at desc);
create index if not exists research_snapshots_issuer_data_cutoff_idx
  on public.research_snapshots (issuer_id, data_cutoff desc);
create index if not exists research_snapshots_security_data_cutoff_idx
  on public.research_snapshots (security_id, data_cutoff desc);

create or replace function public.prevent_orotitan_research_snapshot_mutation()
returns trigger
language plpgsql
security invoker
set search_path = pg_catalog, public
as $$
begin
  raise exception 'canonical research snapshots are immutable' using errcode = '55000';
end;
$$;

revoke execute on function public.prevent_orotitan_research_snapshot_mutation() from public, anon, authenticated, service_role;

drop trigger if exists research_snapshots_prevent_mutation on public.research_snapshots;
create trigger research_snapshots_prevent_mutation
before update or delete on public.research_snapshots
for each row execute function public.prevent_orotitan_research_snapshot_mutation();

alter table public.research_snapshots enable row level security;
revoke all on public.research_snapshots from public, anon, authenticated, service_role;
grant select on public.research_snapshots to service_role;

do $$
begin
  if not exists (select 1 from pg_catalog.pg_class where oid = 'public.research_snapshots'::regclass and relrowsecurity) then
    raise exception 'research_snapshots RLS is not enabled';
  end if;
  if exists (select 1 from pg_catalog.pg_policy where polrelid = 'public.research_snapshots'::regclass) then
    raise exception 'research_snapshots must not have browser policies';
  end if;
  if not has_table_privilege('service_role', 'public.research_snapshots', 'SELECT')
     or has_table_privilege('service_role', 'public.research_snapshots', 'INSERT')
     or has_table_privilege('service_role', 'public.research_snapshots', 'UPDATE')
     or has_table_privilege('service_role', 'public.research_snapshots', 'DELETE') then
    raise exception 'research_snapshots service_role privileges are incompatible';
  end if;
  if not exists (select 1 from pg_catalog.pg_trigger where tgrelid = 'public.research_snapshots'::regclass and tgname = 'research_snapshots_prevent_mutation' and not tgisinternal) then
    raise exception 'research_snapshots immutability trigger is missing';
  end if;
  if not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_snapshots'::regclass and conname = 'research_snapshots_report_id_check')
     or not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_snapshots'::regclass and conname = 'research_snapshots_version_strings_check')
     or not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_snapshots'::regclass and conname = 'research_snapshots_execution_mode_check')
     or not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_snapshots'::regclass and conname = 'research_snapshots_contract_version_check')
     or not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_snapshots'::regclass and conname = 'research_snapshots_schema_version_check')
     or not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_snapshots'::regclass and conname = 'research_snapshots_payload_object_check')
     or not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_snapshots'::regclass and conname = 'research_snapshots_payload_lock_check') then
    raise exception 'research_snapshots checks are incomplete';
  end if;
  if not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_snapshots'::regclass and conname = 'research_snapshots_dossier_issuer_fkey' and confdeltype = 'r')
     or not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_snapshots'::regclass and conname = 'research_snapshots_security_issuer_fkey' and confdeltype = 'r')
     or not exists (select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_dossiers'::regclass and conname = 'research_dossiers_current_snapshot_fkey' and confdeltype = 'r') then
    raise exception 'research_snapshots foreign keys are incomplete';
  end if;
  if not exists (select 1 from pg_catalog.pg_class where oid = 'public.research_snapshots_snapshot_dossier_key'::regclass and relkind = 'i')
     or not exists (select 1 from pg_catalog.pg_class where oid = 'public.research_snapshots_dossier_created_at_idx'::regclass and relkind = 'i')
     or not exists (select 1 from pg_catalog.pg_class where oid = 'public.research_snapshots_issuer_data_cutoff_idx'::regclass and relkind = 'i')
     or not exists (select 1 from pg_catalog.pg_class where oid = 'public.research_snapshots_security_data_cutoff_idx'::regclass and relkind = 'i') then
    raise exception 'research_snapshots indexes are incomplete';
  end if;
  if not exists (
    select 1 from pg_catalog.pg_proc p
    where p.oid = 'public.prevent_orotitan_research_snapshot_mutation()'::regprocedure
      and p.prosecdef = false
      and p.proconfig @> array['search_path=pg_catalog, public']::text[]
  ) then
    raise exception 'research_snapshots trigger function security boundary is incompatible';
  end if;
  if not exists (
    select 1 from pg_catalog.pg_constraint c
    where c.conrelid = 'public.research_snapshots'::regclass
      and c.conname = 'research_snapshots_dossier_issuer_fkey'
      and c.confrelid = 'public.research_dossiers'::regclass
      and c.conkey = array[
        (select attnum from pg_catalog.pg_attribute where attrelid = c.conrelid and attname = 'dossier_id'),
        (select attnum from pg_catalog.pg_attribute where attrelid = c.conrelid and attname = 'issuer_id')
      ]::smallint[]
      and c.confkey = array[
        (select attnum from pg_catalog.pg_attribute where attrelid = c.confrelid and attname = 'dossier_id'),
        (select attnum from pg_catalog.pg_attribute where attrelid = c.confrelid and attname = 'issuer_id')
      ]::smallint[] and c.confdeltype = 'r'
  ) or not exists (
    select 1 from pg_catalog.pg_constraint c
    where c.conrelid = 'public.research_snapshots'::regclass
      and c.conname = 'research_snapshots_security_issuer_fkey'
      and c.confrelid = 'public.securities'::regclass
      and c.conkey = array[
        (select attnum from pg_catalog.pg_attribute where attrelid = c.conrelid and attname = 'security_id'),
        (select attnum from pg_catalog.pg_attribute where attrelid = c.conrelid and attname = 'issuer_id')
      ]::smallint[]
      and c.confkey = array[
        (select attnum from pg_catalog.pg_attribute where attrelid = c.confrelid and attname = 'security_id'),
        (select attnum from pg_catalog.pg_attribute where attrelid = c.confrelid and attname = 'issuer_id')
      ]::smallint[] and c.confdeltype = 'r'
  ) or not exists (
    select 1 from pg_catalog.pg_constraint c
    where c.conrelid = 'public.research_dossiers'::regclass
      and c.conname = 'research_dossiers_current_snapshot_fkey'
      and c.confrelid = 'public.research_snapshots'::regclass
      and c.conkey = array[
        (select attnum from pg_catalog.pg_attribute where attrelid = c.conrelid and attname = 'current_snapshot_id'),
        (select attnum from pg_catalog.pg_attribute where attrelid = c.conrelid and attname = 'dossier_id')
      ]::smallint[]
      and c.confkey = array[
        (select attnum from pg_catalog.pg_attribute where attrelid = c.confrelid and attname = 'snapshot_id'),
        (select attnum from pg_catalog.pg_attribute where attrelid = c.confrelid and attname = 'dossier_id')
      ]::smallint[] and c.confdeltype = 'r'
  ) then
    raise exception 'research_snapshots foreign key material shape is incompatible';
  end if;
  if not exists (
    select 1 from pg_catalog.pg_index i
    where i.indexrelid = 'public.research_snapshots_snapshot_dossier_key'::regclass
      and i.indisunique and i.indkey = array[
        (select attnum from pg_catalog.pg_attribute where attrelid = i.indrelid and attname = 'snapshot_id'),
        (select attnum from pg_catalog.pg_attribute where attrelid = i.indrelid and attname = 'dossier_id')
      ]::int2[]
  ) then raise exception 'snapshot dossier unique index is incompatible'; end if;
  if not exists (
    select 1 from pg_catalog.pg_trigger t
    where t.tgrelid = 'public.research_snapshots'::regclass
      and t.tgname = 'research_snapshots_prevent_mutation'
      and t.tgfoid = 'public.prevent_orotitan_research_snapshot_mutation()'::regprocedure
      and (t.tgtype & 1) = 1 and (t.tgtype & 2) = 2 and (t.tgtype & 24) = 24
  ) then raise exception 'research_snapshots trigger material shape is incompatible'; end if;
  if not exists (
    select 1 from pg_catalog.pg_constraint c
    where c.conrelid = 'public.research_snapshots'::regclass
      and c.conname = 'research_snapshots_contract_version_check'
      and pg_catalog.pg_get_constraintdef(c.oid) like '%04_SCREENER_SCHEMA_V1%'
  ) or not exists (
    select 1 from pg_catalog.pg_constraint c
    where c.conrelid = 'public.research_snapshots'::regclass
      and c.conname = 'research_snapshots_schema_version_check'
      and pg_catalog.pg_get_constraintdef(c.oid) like '%1.0.0%'
  ) then raise exception 'research_snapshots frozen version checks are incompatible'; end if;
end $$;

comment on table public.research_snapshots is
  'Immutable structured projection/history of the certified research snapshot; not analytical truth.';

commit;