\set ON_ERROR_STOP on
INSERT INTO public.companies (
 id,slug,ticker,name,exchange,currency,quote_unit,price_decimals,
 market_data_multiplier,active
) VALUES
('00000000-0000-4000-8000-000000000091','research-alpha','RALP',
 'Synthetic Research Alpha','XPAR','EUR','MAJOR',2,1,true),
('00000000-0000-4000-8000-000000000092','research-beta','RBET',
 'Synthetic Research Beta','XAMS','EUR','MAJOR',2,1,true),
('00000000-0000-4000-8000-000000000093','research-inactive','RINA',
 'Synthetic Inactive','XPAR','EUR','MAJOR',2,1,false);

SET ROLE service_role;
-- Only company_id is required; caller-supplied metadata/ID is replaced.
INSERT INTO public.orotitan_research_intake_requests(
 request_id,company_id,company_name,ticker,exchange,status
) VALUES(
 '00000000-0000-4000-8000-000000009999',
 '00000000-0000-4000-8000-000000000091',
 'FORGED NAME','EVIL','ZZZZ','REQUESTED_NOT_EXECUTED');
RESET ROLE;

DO $verify$
DECLARE r public.orotitan_research_intake_requests%ROWTYPE;
BEGIN
 SELECT * INTO STRICT r FROM public.orotitan_research_intake_requests
 WHERE company_id='00000000-0000-4000-8000-000000000091';
 IF r.request_id='00000000-0000-4000-8000-000000009999'
 OR r.company_name<>'Synthetic Research Alpha'
 OR r.ticker<>'RALP' OR r.exchange<>'XPAR'
 OR r.status<>'REQUESTED_NOT_EXECUTED'
 OR r.registry_run_id IS NOT NULL THEN
   RAISE EXCEPTION 'PRODUCT_SHARED_FORGED_RECEIPT_ACCEPTED';
 END IF;
 IF NOT (SELECT relrowsecurity FROM pg_class
  WHERE oid='public.orotitan_research_intake_requests'::regclass)
 OR has_table_privilege('anon','public.orotitan_research_intake_requests','SELECT')
 OR has_table_privilege('authenticated','public.orotitan_research_intake_requests','INSERT')
 OR has_table_privilege('authenticated','public.orotitan_research_intake_requests','SELECT')
 OR NOT has_table_privilege('service_role','public.orotitan_research_intake_requests','INSERT')
 OR NOT has_table_privilege('service_role','public.orotitan_research_intake_requests','SELECT')
 OR has_table_privilege('service_role','public.orotitan_research_intake_requests','UPDATE')
 OR has_table_privilege('service_role','public.orotitan_research_intake_requests','DELETE') THEN
  RAISE EXCEPTION 'PRODUCT_SHARED_ACCESS_MATRIX_FAILED';
 END IF;
END;
$verify$;

SET ROLE service_role;
DO $cases$
DECLARE denied boolean; r uuid;
BEGIN
 denied:=false;
 BEGIN
  INSERT INTO public.orotitan_research_intake_requests(company_id) VALUES
   ('00000000-0000-4000-8000-000000000093');
 EXCEPTION WHEN raise_exception THEN
  denied:=SQLERRM='PRODUCT_INBOX_COMPANY_NOT_ELIGIBLE';
 END;
 IF NOT denied THEN RAISE EXCEPTION 'PRODUCT_SHARED_INACTIVE_COMPANY_INSERTED'; END IF;
 denied:=false;
 BEGIN
  INSERT INTO public.orotitan_research_intake_requests(company_id) VALUES
   ('00000000-0000-4000-8000-000000000094');
 EXCEPTION WHEN raise_exception THEN
  denied:=SQLERRM='PRODUCT_INBOX_COMPANY_NOT_ELIGIBLE';
 END;
 IF NOT denied THEN RAISE EXCEPTION 'PRODUCT_SHARED_UNKNOWN_COMPANY_INSERTED'; END IF;
 denied:=false;
 BEGIN
  INSERT INTO public.orotitan_research_intake_requests(company_id) VALUES
   ('00000000-0000-4000-8000-000000000091');
 EXCEPTION WHEN unique_violation THEN denied:=true;
 END;
 IF NOT denied THEN RAISE EXCEPTION 'PRODUCT_SHARED_DUPLICATE_INSERTED'; END IF;
 denied:=false;
 BEGIN
  UPDATE public.orotitan_research_intake_requests SET status='REQUESTED_NOT_EXECUTED'
   WHERE company_id='00000000-0000-4000-8000-000000000091';
 EXCEPTION WHEN insufficient_privilege THEN denied:=true;
 END;
 IF NOT denied THEN RAISE EXCEPTION 'PRODUCT_SHARED_SERVICE_ROLE_CAN_UPDATE'; END IF;
 denied:=false;
 BEGIN
  DELETE FROM public.orotitan_research_intake_requests
   WHERE company_id='00000000-0000-4000-8000-000000000091';
 EXCEPTION WHEN insufficient_privilege THEN denied:=true;
 END;
 IF NOT denied THEN RAISE EXCEPTION 'PRODUCT_SHARED_SERVICE_ROLE_CAN_DELETE'; END IF;
END;
$cases$;
RESET ROLE;

DO $integrity$
BEGIN
 IF (SELECT count(*) FROM public.orotitan_research_intake_requests)<>1
 OR (SELECT count(*) FROM public.companies)<>3
 OR (SELECT count(*) FROM public.snapshots)<>0
 OR to_regclass('public.orotitan_runs') IS NOT NULL THEN
   RAISE EXCEPTION 'PRODUCT_SHARED_CANONICAL_OR_REGISTRY_MODIFIED';
 END IF;
END;
$integrity$;
\echo 'Product shared inbox permissions, identity and nonexecution PASS.'
