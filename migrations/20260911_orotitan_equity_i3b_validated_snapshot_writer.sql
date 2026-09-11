-- OROTitan Equity Research V1 - I3-B validated canonical snapshot writer.
-- The application validates the frozen contract before this RPC is called.

begin;

create or replace function public.persist_orotitan_research_snapshot(
  p_dossier_id uuid,
  p_expected_current_snapshot_id uuid,
  p_canonical_payload jsonb
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  dossier_issuer_id uuid;
  current_snapshot_id uuid;
  payload_snapshot_id uuid;
  payload_issuer_id uuid;
  payload_security_id uuid;
  payload_report_id text;
  payload_execution_mode text;
  payload_data_cutoff date;
  payload_calculation_date date;
  payload_report_version text;
  payload_method_version text;
  payload_calculation_version text;
  payload_evidence_ledger_version text;
  existing_snapshot public.research_snapshots%rowtype;
begin
  select d.issuer_id, d.current_snapshot_id
    into dossier_issuer_id, current_snapshot_id
  from public.research_dossiers d
  where d.dossier_id = p_dossier_id
  for update;

  if not found then
    raise exception 'research dossier does not exist: %', p_dossier_id using errcode = '23503';
  end if;

    if p_canonical_payload is null
      or jsonb_typeof(p_canonical_payload) is distinct from 'object' then
    raise exception 'canonical payload must be a JSON object' using errcode = '22023';
  end if;

  begin
    payload_snapshot_id := (p_canonical_payload->>'snapshot_id')::uuid;
    payload_issuer_id := (p_canonical_payload->>'issuer_id')::uuid;
    payload_security_id := (p_canonical_payload->>'security_id')::uuid;
    payload_report_id := p_canonical_payload->>'report_id';
    payload_execution_mode := p_canonical_payload->>'execution_mode';
    payload_data_cutoff := (p_canonical_payload->'data_lock'->>'data_cutoff')::date;
    payload_calculation_date := (p_canonical_payload->'data_lock'->>'calculation_date')::date;
    payload_report_version := p_canonical_payload->'versions'->>'report_version';
    payload_method_version := p_canonical_payload->'versions'->>'method_version';
    payload_calculation_version := p_canonical_payload->'versions'->>'calculation_version';
    payload_evidence_ledger_version := p_canonical_payload->'versions'->>'evidence_ledger_version';
  exception when invalid_text_representation then
    raise exception 'canonical payload contains an invalid persistence lock value' using errcode = '22023';
  end;

  if payload_issuer_id is distinct from dossier_issuer_id then
    raise exception 'canonical payload issuer does not match dossier issuer' using errcode = '23514';
  end if;

  select s.* into existing_snapshot
  from public.research_snapshots s
  where s.snapshot_id = payload_snapshot_id;

  if found then
    if existing_snapshot.dossier_id = p_dossier_id
       and existing_snapshot.issuer_id = payload_issuer_id
       and existing_snapshot.security_id = payload_security_id
       and existing_snapshot.report_id = payload_report_id
       and existing_snapshot.execution_mode = payload_execution_mode
       and existing_snapshot.data_cutoff = payload_data_cutoff
       and existing_snapshot.calculation_date = payload_calculation_date
       and existing_snapshot.report_version = payload_report_version
       and existing_snapshot.method_version = payload_method_version
       and existing_snapshot.calculation_version = payload_calculation_version
       and existing_snapshot.evidence_ledger_version = payload_evidence_ledger_version
       and existing_snapshot.contract_version = '04_SCREENER_SCHEMA_V1'
       and existing_snapshot.schema_version = '1.0.0'
       and existing_snapshot.canonical_payload = p_canonical_payload
       and current_snapshot_id = payload_snapshot_id then
      return jsonb_build_object(
        'status', 'IDEMPOTENT_SUCCESS',
        'dossier_id', p_dossier_id,
        'snapshot_id', payload_snapshot_id,
        'current_snapshot_id', current_snapshot_id
      );
    end if;
    raise exception 'snapshot_id already exists with a different identity, lock, payload, or pointer state' using errcode = '23505';
  end if;

  if current_snapshot_id is distinct from p_expected_current_snapshot_id then
    raise exception 'canonical snapshot concurrency conflict: expected %, current %',
      p_expected_current_snapshot_id, current_snapshot_id using errcode = '40001';
  end if;

  insert into public.research_snapshots (
    snapshot_id, dossier_id, issuer_id, security_id, report_id, execution_mode,
    data_cutoff, calculation_date, report_version, method_version,
    calculation_version, evidence_ledger_version, contract_version, schema_version,
    canonical_payload
  ) values (
    payload_snapshot_id, p_dossier_id, payload_issuer_id, payload_security_id,
    payload_report_id, payload_execution_mode, payload_data_cutoff, payload_calculation_date,
    payload_report_version, payload_method_version, payload_calculation_version,
    payload_evidence_ledger_version, '04_SCREENER_SCHEMA_V1', '1.0.0', p_canonical_payload
  );

  update public.research_dossiers
  set current_snapshot_id = payload_snapshot_id
  where dossier_id = p_dossier_id;

  if not found then
    raise exception 'research dossier pointer update failed' using errcode = '23503';
  end if;

  return jsonb_build_object(
    'status', 'INSERTED',
    'dossier_id', p_dossier_id,
    'snapshot_id', payload_snapshot_id,
    'current_snapshot_id', payload_snapshot_id
  );
end;
$$;

revoke execute on function public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)
  from public, anon, authenticated;
grant execute on function public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)
  to service_role;

do $$
begin
  if not exists (
    select 1 from pg_catalog.pg_proc p
    where p.oid = 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)'::regprocedure
      and p.prosecdef
      and p.proconfig @> array['search_path=pg_catalog, public']::text[]
  ) then
    raise exception 'I3-B writer security boundary is incompatible';
  end if;
  if has_function_privilege('public', 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)', 'EXECUTE')
     or has_function_privilege('anon', 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)', 'EXECUTE')
     or has_function_privilege('authenticated', 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)', 'EXECUTE')
     or not has_function_privilege('service_role', 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)', 'EXECUTE') then
    raise exception 'I3-B writer execute privileges are incompatible';
  end if;
  if has_table_privilege('service_role', 'public.research_snapshots', 'INSERT')
     or has_table_privilege('service_role', 'public.research_snapshots', 'UPDATE')
     or has_table_privilege('service_role', 'public.research_snapshots', 'DELETE') then
    raise exception 'I3-B must not grant direct snapshot table writes';
  end if;
end $$;

comment on function public.persist_orotitan_research_snapshot(uuid, uuid, jsonb) is
  'Server-only validated canonical snapshot writer with compare-and-swap pointer semantics.';

commit;