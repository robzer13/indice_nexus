\set ON_ERROR_STOP on

do $$
declare
  first_issuer uuid;
  second_issuer uuid;
  first_security uuid;
  second_security uuid;
  first_dossier uuid;
  second_dossier uuid;
  first_snapshot uuid := '90000000-0000-4000-8000-000000000001';
  second_snapshot uuid := '90000000-0000-4000-8000-000000000002';
  payload jsonb;
  base_payload jsonb;
  failed boolean;
begin
  select issuer_id, security_id, dossier_id into first_issuer, first_security, first_dossier
  from public.legacy_company_identity_map order by legacy_company_id limit 1;
  select issuer_id, security_id, dossier_id into second_issuer, second_security, second_dossier
  from public.legacy_company_identity_map order by legacy_company_id offset 1 limit 1;

  if (select count(*) from public.research_snapshots) <> 0 then raise exception 'T1 expected empty snapshot table'; end if;
  if (select count(*) from pg_attribute where attrelid = 'public.research_snapshots'::regclass and attnum > 0 and not attisdropped) <> 16 then raise exception 'T1 shape mismatch'; end if;
  if exists (select 1 from pg_attribute where attrelid = 'public.research_dossiers'::regclass and attname = 'current_snapshot_id' and attnotnull) then raise exception 'T4 pointer is not nullable'; end if;

  payload := jsonb_build_object(
    'snapshot_id', first_snapshot::text, 'report_id', 'report-alpha', 'issuer_id', first_issuer::text,
    'security_id', first_security::text, 'execution_mode', 'ANALYZE',
    'data_lock', jsonb_build_object('data_cutoff', '2026-09-10', 'calculation_date', '2026-09-11'),
    'versions', jsonb_build_object('report_version', 'r1', 'method_version', 'm1', 'calculation_version', 'c1', 'evidence_ledger_version', 'e1')
  );
  insert into public.research_snapshots (
    snapshot_id, dossier_id, issuer_id, security_id, report_id, execution_mode,
    data_cutoff, calculation_date, report_version, method_version,
    calculation_version, evidence_ledger_version, canonical_payload
  ) values (
    first_snapshot, first_dossier, first_issuer, first_security, 'report-alpha', 'ANALYZE',
    '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', payload
  );
  base_payload := payload;
  update public.research_dossiers set current_snapshot_id = first_snapshot where dossier_id = first_dossier;
  if (select current_snapshot_id from public.research_dossiers where dossier_id = first_dossier) <> first_snapshot then raise exception 'T7 pointer did not advance'; end if;

  payload := jsonb_set(payload, '{snapshot_id}', to_jsonb(second_snapshot::text));
  insert into public.research_snapshots (
    snapshot_id, dossier_id, issuer_id, security_id, report_id, execution_mode,
    data_cutoff, calculation_date, report_version, method_version,
    calculation_version, evidence_ledger_version, canonical_payload
  ) values (
    second_snapshot, first_dossier, first_issuer, first_security, 'report-beta', 'REFRESH',
    '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1',
    jsonb_set(jsonb_set(payload, '{report_id}', '"report-beta"'), '{execution_mode}', '"REFRESH"')
  );
  if (select count(*) from public.research_snapshots) <> 2 then raise exception 'T8 second snapshot was not accepted'; end if;
  update public.research_dossiers set current_snapshot_id = second_snapshot where dossier_id = first_dossier;
  if (select current_snapshot_id from public.research_dossiers where dossier_id = first_dossier) <> second_snapshot then raise exception 'T9 pointer did not advance'; end if;
  if (select report_id from public.research_snapshots where snapshot_id = first_snapshot) <> 'report-alpha' then raise exception 'T9 prior snapshot mutated'; end if;
  if not exists (select 1 from public.research_snapshots where snapshot_id = second_snapshot and report_id = 'report-beta') then raise exception 'T9 second snapshot missing'; end if;

  failed := false;
  begin update public.research_snapshots set report_id = 'mutated' where snapshot_id = first_snapshot; exception when others then failed := true; end;
  if not failed then raise exception 'T10 UPDATE was accepted'; end if;
  failed := false;
  begin delete from public.research_snapshots where snapshot_id = first_snapshot; exception when others then failed := true; end;
  if not failed then raise exception 'T11 DELETE was accepted'; end if;

  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000010', first_dossier, second_issuer, second_security, 'report-alpha', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000010"'), '{issuer_id}', to_jsonb(second_issuer::text)), '{security_id}', to_jsonb(second_security::text)), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T12 dossier/issuer mismatch was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000011', first_dossier, first_issuer, second_security, 'report-alpha', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000011"'), '{security_id}', to_jsonb(second_security::text)), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T13 security/issuer mismatch was accepted'; end if;
  failed := false;
  begin update public.research_dossiers set current_snapshot_id = second_snapshot where dossier_id = second_dossier; exception when others then failed := true; end;
  if not failed then raise exception 'T14 cross-dossier pointer was accepted'; end if;

  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000003', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"wrong"'), '{report_id}', '"report-gamma"'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T15 JSON snapshot_id mismatch was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000004', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000004"'), '{report_id}', '"wrong-report"'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T16 JSON report_id mismatch was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000013', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000013"'), '{issuer_id}', '"00000000-0000-4000-8000-000000000002"'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T16 JSON issuer_id mismatch was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000014', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000014"'), '{security_id}', '"00000000-0000-4000-8000-000000000002"'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T16 JSON security_id mismatch was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000015', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000015"'), '{execution_mode}', '"REFRESH"'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T16 JSON execution_mode mismatch was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000005', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000005"'), '{report_id}', '"report-gamma"'), '{data_lock,data_cutoff}', '"2026-01-01"'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T17 JSON date mismatch was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000016', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000016"'), '{report_id}', '"report-gamma"'), '{data_lock,calculation_date}', '"2026-01-01"'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T17 JSON calculation_date mismatch was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000006', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000006"'), '{report_id}', '"report-gamma"'), '{versions,method_version}', '"wrong"'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T18 JSON version mismatch was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000017', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000017"'), '{report_id}', '"report-gamma"'), '{versions,report_version}', '"wrong"'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T18 JSON report_version mismatch was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000018', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000018"'), '{report_id}', '"report-gamma"'), '{versions,calculation_version}', '"wrong"'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T18 JSON calculation_version mismatch was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000019', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', jsonb_set(jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000019"'), '{report_id}', '"report-gamma"'), '{versions,evidence_ledger_version}', '"wrong"'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T18 JSON evidence_ledger_version mismatch was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000007', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', (jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000007"'), '{report_id}', '"report-gamma"') - 'versions'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T19 missing versions object was accepted'; end if;
  failed := false;
  begin insert into public.research_snapshots select '90000000-0000-4000-8000-000000000012', first_dossier, first_issuer, first_security, 'report-gamma', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1', '04_SCREENER_SCHEMA_V1', '1.0.0', (jsonb_set(jsonb_set(base_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000012"'), '{report_id}', '"report-gamma"') - 'data_lock'), now(); exception when others then failed := true; end;
  if not failed then raise exception 'T19 missing data_lock object was accepted'; end if;
  if not failed then raise exception 'T19 missing JSON lock path was accepted'; end if;

  if not (select relrowsecurity from pg_class where oid = 'public.research_snapshots'::regclass) then raise exception 'T20 RLS is disabled'; end if;
  if exists (select 1 from pg_policy where polrelid = 'public.research_snapshots'::regclass) then raise exception 'T20 browser policy exists'; end if;
  if has_table_privilege('anon', 'public.research_snapshots', 'SELECT') or has_table_privilege('authenticated', 'public.research_snapshots', 'SELECT') then raise exception 'T21 browser SELECT exposed'; end if;
  if not has_table_privilege('service_role', 'public.research_snapshots', 'SELECT') or has_table_privilege('service_role', 'public.research_snapshots', 'INSERT') or has_table_privilege('service_role', 'public.research_snapshots', 'UPDATE') or has_table_privilege('service_role', 'public.research_snapshots', 'DELETE') then raise exception 'T22 service_role privileges wrong'; end if;
  if has_function_privilege('anon', 'public.prevent_orotitan_research_snapshot_mutation()', 'EXECUTE') or has_function_privilege('authenticated', 'public.prevent_orotitan_research_snapshot_mutation()', 'EXECUTE') or has_function_privilege('service_role', 'public.prevent_orotitan_research_snapshot_mutation()', 'EXECUTE') then raise exception 'T24 trigger helper execute exposed'; end if;

  failed := false;
  begin delete from public.issuers where issuer_id = first_issuer; exception when others then failed := true; end;
  if not failed then raise exception 'T25 identity delete restriction lost'; end if;
end $$;

set role service_role;
do $$
declare
  failed boolean := false;
begin
  begin
    insert into public.research_snapshots (
      snapshot_id, dossier_id, issuer_id, security_id, report_id, execution_mode,
      data_cutoff, calculation_date, report_version, method_version,
      calculation_version, evidence_ledger_version, canonical_payload
    ) select '90000000-0000-4000-8000-000000000008', dossier_id, issuer_id, security_id,
      'service-write', 'ANALYZE', '2026-09-10', '2026-09-11', 'r1', 'm1', 'c1', 'e1',
      canonical_payload from public.research_snapshots limit 1;
  exception when others then failed := true;
  end;
  if not failed then raise exception 'T23 service_role INSERT was accepted'; end if;
end $$;
reset role;

select 'I3 PostgreSQL integration matrix T1-T25 passed' as result;