\set ON_ERROR_STOP on

set role service_role;
do $$
declare
  dossier uuid;
  issuer uuid;
  security uuid;
  v1_snapshot uuid := '91000000-0000-4000-8000-000000000101';
  v2_snapshot uuid := '91000000-0000-4000-8000-000000000102';
  v1_payload jsonb;
  v2_payload jsonb;
  result jsonb;
  failed boolean;
  before_count bigint;
  before_pointer uuid;
begin
  select m.dossier_id, m.issuer_id, m.security_id into dossier, issuer, security
  from public.legacy_company_identity_map m order by m.legacy_company_id limit 1;

  if to_regprocedure('public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)') is null then raise exception 'V2-T01 V1 writer missing'; end if;
  if to_regprocedure('public.persist_orotitan_research_snapshot_v2(uuid, uuid, jsonb)') is null then raise exception 'V2-T02 V2 writer missing'; end if;
  if has_function_privilege('public', 'public.persist_orotitan_research_snapshot_v2(uuid, uuid, jsonb)', 'EXECUTE')
     or has_function_privilege('anon', 'public.persist_orotitan_research_snapshot_v2(uuid, uuid, jsonb)', 'EXECUTE')
     or has_function_privilege('authenticated', 'public.persist_orotitan_research_snapshot_v2(uuid, uuid, jsonb)', 'EXECUTE') then
    raise exception 'V2-T03 V2 writer exposed';
  end if;
  if not has_function_privilege('service_role', 'public.persist_orotitan_research_snapshot_v2(uuid, uuid, jsonb)', 'EXECUTE') then raise exception 'V2-T04 service role missing V2 writer'; end if;
  if has_table_privilege('service_role', 'public.research_snapshots', 'INSERT')
     or has_table_privilege('service_role', 'public.research_snapshots', 'UPDATE')
     or has_table_privilege('service_role', 'public.research_snapshots', 'DELETE') then raise exception 'V2-T05 direct write privilege exists'; end if;

  v1_payload := jsonb_build_object(
    'snapshot_id', v1_snapshot::text, 'report_id', 'v2-compat-v1',
    'issuer_id', issuer::text, 'security_id', security::text, 'execution_mode', 'ANALYZE',
    'data_lock', jsonb_build_object('data_cutoff', '2026-09-15', 'calculation_date', '2026-09-15'),
    'versions', jsonb_build_object('report_version', 'r1', 'method_version', 'm1', 'calculation_version', 'c1', 'evidence_ledger_version', 'e1')
  );
  result := public.persist_orotitan_research_snapshot(dossier, null, v1_payload);
  if result->>'status' <> 'INSERTED' then raise exception 'V2-T06 V1 writer no longer works'; end if;
  if not exists (select 1 from public.research_snapshots where snapshot_id=v1_snapshot and contract_version='04_SCREENER_SCHEMA_V1' and schema_version='1.0.0') then raise exception 'V2-T07 V1 pair changed'; end if;

  v2_payload := jsonb_set(v1_payload, '{snapshot_id}', to_jsonb(v2_snapshot::text));
  v2_payload := jsonb_set(v2_payload, '{report_id}', to_jsonb('v2-canonical'::text));
  v2_payload := v2_payload || jsonb_build_object('v2_product', jsonb_build_object(
    'classification', jsonb_build_object('issuer_country_code','US','primary_listing_country_code','US','sector','INFORMATION_TECHNOLOGY','industry_group','CYBERSECURITY','business_model_primary','RECURRING_SUBSCRIPTION','economic_exposure_regions',jsonb_build_array('GLOBAL'),'taxonomy_version','OROTITAN_TAXONOMY_V2.0'),
    'business_summary', jsonb_build_object('business_description_short','Cloud security subscription software.'),
    'investment_thesis', jsonb_build_object('quality_case','High recurring cash economics.','valuation_case','Return depends on entry price.','key_risk','Competitive bundling.')
  ));
  result := public.persist_orotitan_research_snapshot_v2(dossier, v1_snapshot, v2_payload);
  if result->>'status' <> 'INSERTED' then raise exception 'V2-T08 V2 insert failed'; end if;
  if not exists (select 1 from public.research_snapshots where snapshot_id=v2_snapshot and contract_version='04_SCREENER_SCHEMA_V2' and schema_version='2.0.0') then raise exception 'V2-T09 V2 pair missing'; end if;
  result := public.persist_orotitan_research_snapshot_v2(dossier, v1_snapshot, v2_payload);
  if result->>'status' <> 'IDEMPOTENT_SUCCESS' then raise exception 'V2-T10 V2 retry not idempotent'; end if;

  before_count := (select count(*) from public.research_snapshots);
  before_pointer := (select current_snapshot_id from public.research_dossiers where dossier_id=dossier);
  failed := false;
  begin
    perform public.persist_orotitan_research_snapshot(dossier, v2_snapshot,
      jsonb_set(v2_payload, '{snapshot_id}', '"91000000-0000-4000-8000-000000000103"'));
  exception when sqlstate '23514' then failed := true; end;
  if not failed then raise exception 'V2-T11 V2 payload accepted through V1 writer'; end if;
  if (select count(*) from public.research_snapshots) <> before_count or (select current_snapshot_id from public.research_dossiers where dossier_id=dossier) <> before_pointer then raise exception 'V2-T12 failed V1/V2 mix mutated state'; end if;

  failed := false;
  begin
    perform public.persist_orotitan_research_snapshot_v2(dossier, v2_snapshot,
      jsonb_set(v1_payload, '{snapshot_id}', '"91000000-0000-4000-8000-000000000104"'));
  exception when sqlstate '23514' then failed := true; end;
  if not failed then raise exception 'V2-T13 V2 writer accepted payload without V2 product'; end if;

  if not (select p.prosecdef and p.proconfig @> array['search_path=pg_catalog, public']::text[]
          from pg_proc p where p.oid='public.persist_orotitan_research_snapshot_v2(uuid, uuid, jsonb)'::regprocedure) then
    raise exception 'V2-T14 V2 writer security-definer boundary mismatch';
  end if;
end $$;
reset role;

do $$
declare
  dossier uuid;
  issuer uuid;
  security uuid;
  failed boolean := false;
  payload jsonb;
begin
  select m.dossier_id, m.issuer_id, m.security_id into dossier, issuer, security from public.legacy_company_identity_map m order by m.legacy_company_id offset 1 limit 1;
  payload := jsonb_build_object(
    'snapshot_id','91000000-0000-4000-8000-000000000105','report_id','mixed-pair',
    'issuer_id',issuer::text,'security_id',security::text,'execution_mode','ANALYZE',
    'data_lock',jsonb_build_object('data_cutoff','2026-09-15','calculation_date','2026-09-15'),
    'versions',jsonb_build_object('report_version','r1','method_version','m1','calculation_version','c1','evidence_ledger_version','e1'),
    'v2_product',jsonb_build_object('classification',jsonb_build_object('taxonomy_version','OROTITAN_TAXONOMY_V2.0'))
  );
  begin
    insert into public.research_snapshots(snapshot_id,dossier_id,issuer_id,security_id,report_id,execution_mode,data_cutoff,calculation_date,report_version,method_version,calculation_version,evidence_ledger_version,contract_version,schema_version,canonical_payload)
    values('91000000-0000-4000-8000-000000000105',dossier,issuer,security,'mixed-pair','ANALYZE','2026-09-15','2026-09-15','r1','m1','c1','e1','04_SCREENER_SCHEMA_V1','1.0.0',payload);
  exception when sqlstate '23514' then failed := true; end;
  if not failed then raise exception 'V2-T15 payload-version firewall accepted mixed V1/V2 state'; end if;
end $$;

select json_build_object(
  'result','PASS',
  'v1_writer',to_regprocedure('public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)') is not null,
  'v2_writer',to_regprocedure('public.persist_orotitan_research_snapshot_v2(uuid, uuid, jsonb)') is not null,
  'snapshot_rows',(select count(*) from public.research_snapshots)
);
