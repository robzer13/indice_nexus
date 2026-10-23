-- Product MVP / SHARED REQUEST INBOX v1 — REVIEW-ONLY SQL PROPOSAL.
-- DO NOT APPLY TO THE LIVE SUPABASE PROJECT. No migration is authorized.
-- Tested exclusively against a disposable PostgreSQL 17 database.
-- Purpose: one durable request per EXISTING ACTIVE Screener company, shared
-- across processes. A request is NEVER a Research run or Research dossier.
BEGIN;
DO $check$
BEGIN
 IF current_setting('server_version_num')::int NOT BETWEEN 170000 AND 179999
   OR to_regclass('public.companies') IS NULL
   OR to_regclass('public.orotitan_research_intake_requests') IS NOT NULL THEN
  RAISE EXCEPTION 'PRODUCT_INBOX_DISPOSABLE_PG17_WITH_SCREENER_REQUIRED';
 END IF;
END;
$check$;

CREATE TABLE public.orotitan_research_intake_requests (
 request_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 company_id uuid NOT NULL REFERENCES public.companies(id) ON DELETE RESTRICT,
 company_name text NOT NULL,
 ticker text NOT NULL,
 exchange text NOT NULL,
 status text NOT NULL DEFAULT 'REQUESTED_NOT_EXECUTED'
  CHECK (status='REQUESTED_NOT_EXECUTED'),
 registry_run_id uuid DEFAULT NULL CHECK (registry_run_id IS NULL),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 CONSTRAINT research_intake_single_request_per_company UNIQUE(company_id)
);
CREATE INDEX research_intake_created_at_idx
 ON public.orotitan_research_intake_requests(created_at DESC,request_id DESC);

-- No actor-supplied company name/listing can enter persisted receipts.
-- The current Screener identity is verified again within the INSERT TX.
CREATE FUNCTION public.product_intake_validate_company_v1()
RETURNS trigger LANGUAGE plpgsql
SET search_path=pg_catalog,public AS $validate$
DECLARE c record;
BEGIN
 IF TG_OP<>'INSERT' THEN
   RAISE EXCEPTION 'PRODUCT_INBOX_REQUEST_IMMUTABLE';
 END IF;
 SELECT id,name,ticker,exchange,active INTO c
 FROM public.companies WHERE id=NEW.company_id FOR SHARE;
 IF NOT FOUND OR c.active IS DISTINCT FROM true OR
    c.name IS NULL OR length(trim(c.name))<2 OR
    c.ticker IS NULL OR length(trim(c.ticker))<1 OR
    c.exchange IS NULL OR length(trim(c.exchange))<2 THEN
  RAISE EXCEPTION 'PRODUCT_INBOX_COMPANY_NOT_ELIGIBLE';
 END IF;
 -- Ignore all client-supplied receipt snapshots and identifiers.
 NEW.company_name:=c.name;
 NEW.ticker:=c.ticker;
 NEW.exchange:=c.exchange;
 NEW.status:='REQUESTED_NOT_EXECUTED';
 NEW.registry_run_id:=NULL;
 NEW.created_at:=clock_timestamp();
 RETURN NEW;
END;
$validate$;
CREATE TRIGGER product_intake_before_insert_v1
 BEFORE INSERT ON public.orotitan_research_intake_requests
 FOR EACH ROW EXECUTE FUNCTION public.product_intake_validate_company_v1();

CREATE FUNCTION public.product_intake_reject_mutations_v1()
RETURNS trigger LANGUAGE plpgsql
SET search_path=pg_catalog,public AS $deny$
BEGIN RAISE EXCEPTION 'PRODUCT_INBOX_REQUEST_IMMUTABLE'; END;
$deny$;
CREATE TRIGGER product_intake_immutable_v1
 BEFORE UPDATE OR DELETE ON public.orotitan_research_intake_requests
 FOR EACH ROW EXECUTE FUNCTION public.product_intake_reject_mutations_v1();

ALTER TABLE public.orotitan_research_intake_requests ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.orotitan_research_intake_requests FROM PUBLIC;
REVOKE ALL ON public.orotitan_research_intake_requests FROM anon, authenticated;
-- Backend admin session authentication is a separate trust boundary.
-- service_role already exists in the application, and is SERVER-ONLY.
GRANT SELECT,INSERT ON public.orotitan_research_intake_requests TO service_role;
-- Avoid public function EXECUTE when no callers require it.
REVOKE ALL ON FUNCTION public.product_intake_validate_company_v1() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.product_intake_reject_mutations_v1() FROM PUBLIC;
COMMIT;
