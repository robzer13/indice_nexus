-- D30: disposable PostgreSQL 17 queue contract for mobile ChatGPT requests.
-- REVIEW ONLY. DO NOT APPLY TO A HOSTED SUPABASE PROJECT OR PRODUCTION.
-- PostgreSQL persists only CONFIRMED REQUESTS, NEVER starts Registry stages,
-- writes companies/issuers, verifies evidence or launches research workers.
BEGIN;
DO $guard$
BEGIN
 IF current_setting('server_version_num')::int NOT BETWEEN 170000 AND 179999
 OR to_regclass('public.issuers') IS NULL
 OR to_regclass('public.securities') IS NULL
 OR to_regclass('public.companies') IS NULL
 OR to_regclass('orotitan_mobile_private.previews') IS NOT NULL THEN
   RAISE EXCEPTION 'D30 LOCAL_POSTGRES_17_AND_CANONICAL_I1_REQUIRED';
 END IF;
END;
$guard$;

CREATE SCHEMA orotitan_mobile_private;
REVOKE ALL ON SCHEMA orotitan_mobile_private FROM PUBLIC;
CREATE TABLE orotitan_mobile_private.previews (
 preview_id uuid PRIMARY KEY,
 kind text NOT NULL CHECK(kind IN ('add_company','request_research')),
 actor_subject text NOT NULL CHECK(length(actor_subject) BETWEEN 4 AND 160),
 actor_client text NOT NULL CHECK(length(actor_client) BETWEEN 6 AND 160),
 actor_resource text NOT NULL
   CHECK(actor_resource='https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'),
 payload jsonb NOT NULL CHECK(jsonb_typeof(payload)='object'),
 fingerprint_sha256 text NOT NULL CHECK(fingerprint_sha256 ~ '^[a-f0-9]{64}$'),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 expires_at timestamptz NOT NULL CHECK(expires_at>created_at AND expires_at<=created_at+interval '2 minutes'),
 confirmed_request_id uuid
);
CREATE TABLE orotitan_mobile_private.command_requests (
 request_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 kind text NOT NULL CHECK(kind IN ('add_company','request_research')),
 target_key text NOT NULL,
 actor_subject text NOT NULL,
 actor_client text NOT NULL,
 actor_resource text NOT NULL,
 payload jsonb NOT NULL CHECK(jsonb_typeof(payload)='object'),
 fingerprint_sha256 text NOT NULL CHECK(fingerprint_sha256 ~ '^[a-f0-9]{64}$'),
 state text NOT NULL DEFAULT 'REQUESTED_NOT_EXECUTED'
  CHECK(state='REQUESTED_NOT_EXECUTED'),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 CONSTRAINT d30_request_target_unique UNIQUE(kind,target_key)
);
ALTER TABLE orotitan_mobile_private.previews
 ADD CONSTRAINT d30_preview_confirmed_request_fkey
 FOREIGN KEY(confirmed_request_id) REFERENCES
 orotitan_mobile_private.command_requests(request_id) ON DELETE RESTRICT;

REVOKE ALL ON ALL TABLES IN SCHEMA orotitan_mobile_private FROM PUBLIC;
ALTER TABLE orotitan_mobile_private.previews ENABLE ROW LEVEL SECURITY;
ALTER TABLE orotitan_mobile_private.command_requests ENABLE ROW LEVEL SECURITY;
-- No RLS policies and no role grants. There is NO enabled runtime path.
-- The disposable local superuser fixture is the ONLY caller of this proposal.

CREATE FUNCTION orotitan_mobile_private.confirm_request_v1(
  p_preview uuid,p_subject text,p_client text,p_resource text,p_fingerprint text
) RETURNS TABLE (request_id uuid,kind text,state text,duplicate boolean)
LANGUAGE plpgsql VOLATILE SECURITY INVOKER
SET search_path=pg_catalog,public,orotitan_mobile_private AS $confirm$
DECLARE v_preview orotitan_mobile_private.previews%ROWTYPE;
        v_request orotitan_mobile_private.command_requests%ROWTYPE;
        v_target text;v_inserted uuid;v_issuer uuid;
BEGIN
 SELECT * INTO v_preview FROM orotitan_mobile_private.previews
   WHERE preview_id=p_preview FOR UPDATE;
 IF NOT FOUND OR p_subject IS NULL OR p_client IS NULL
 OR p_resource IS NULL OR p_fingerprint IS NULL
 OR v_preview.actor_subject IS DISTINCT FROM p_subject
 OR v_preview.actor_client IS DISTINCT FROM p_client
 OR v_preview.actor_resource IS DISTINCT FROM p_resource
 OR v_preview.fingerprint_sha256 IS DISTINCT FROM p_fingerprint
 OR clock_timestamp()>=v_preview.expires_at THEN
   RAISE EXCEPTION 'D30_PREVIEW_ACTOR_FINGERPRINT_OR_EXPIRY_DENIED';
 END IF;
 IF v_preview.kind='add_company' THEN
   IF NOT (v_preview.payload ?& array['name','ticker','exchange','isin'])
    OR v_preview.payload->>'ticker' !~ '^[A-Z0-9][A-Z0-9.-]{0,19}$'
    OR v_preview.payload->>'exchange' !~ '^[A-Z0-9][A-Z0-9_-]{1,39}$'
    OR v_preview.payload->>'isin' !~ '^[A-Z]{2}[A-Z0-9]{9}[0-9]$'
    OR length(v_preview.payload->>'name') NOT BETWEEN 2 AND 160 THEN
      RAISE EXCEPTION 'D30_IDENTITY_FIELDS_INVALID';
   END IF;
   IF EXISTS(SELECT 1 FROM public.securities s
      WHERE (upper(s.exchange)=v_preview.payload->>'exchange'
        AND upper(s.ticker)=v_preview.payload->>'ticker')
        OR s.isin=v_preview.payload->>'isin')
   OR EXISTS(SELECT 1 FROM public.companies c
      WHERE upper(c.exchange)=v_preview.payload->>'exchange'
        AND upper(c.ticker)=v_preview.payload->>'ticker') THEN
      RAISE EXCEPTION 'D30_COMPANY_ALREADY_EXISTS';
   END IF;
   v_target:='company:'||(v_preview.payload->>'exchange')||':'||
     (v_preview.payload->>'ticker')||':'||(v_preview.payload->>'isin');
 ELSIF v_preview.kind='request_research' THEN
   IF NOT v_preview.payload ?& array['issuerId','name','ticker','exchange','isin']
     OR (v_preview.payload->>'issuerId') !~
      '^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$' THEN
       RAISE EXCEPTION 'D30_RESEARCH_ISSUER_INVALID';
   END IF;
   v_issuer:=(v_preview.payload->>'issuerId')::uuid;
   IF NOT EXISTS(SELECT 1 FROM public.issuers i
      JOIN public.securities s ON s.issuer_id=i.issuer_id
      WHERE i.issuer_id=v_issuer AND i.display_name=v_preview.payload->>'name'
        AND s.ticker=v_preview.payload->>'ticker'
        AND s.exchange=v_preview.payload->>'exchange'
        AND s.isin=v_preview.payload->>'isin' AND s.isin<>'UNKNOWN') THEN
     RAISE EXCEPTION 'D30_CANONICAL_ISSUER_OR_SECURITY_CHANGED';
   END IF;
   v_target:='research:'||v_issuer::text;
 ELSE RAISE EXCEPTION 'D30_COMMAND_KIND_INVALID';
 END IF;
 IF v_preview.confirmed_request_id IS NOT NULL THEN
   SELECT * INTO STRICT v_request FROM orotitan_mobile_private.command_requests q
   WHERE q.request_id=v_preview.confirmed_request_id;
   RETURN QUERY SELECT v_request.request_id,v_request.kind,v_request.state,true;
   RETURN;
 END IF;
 INSERT INTO orotitan_mobile_private.command_requests AS q(
   kind,target_key,actor_subject,actor_client,actor_resource,payload,fingerprint_sha256)
 VALUES(v_preview.kind,v_target,p_subject,p_client,p_resource,
   v_preview.payload,p_fingerprint)
 ON CONFLICT ON CONSTRAINT d30_request_target_unique DO NOTHING
 RETURNING q.request_id INTO v_inserted;
 IF v_inserted IS NULL THEN
   SELECT * INTO STRICT v_request FROM orotitan_mobile_private.command_requests q
   WHERE q.kind=v_preview.kind AND q.target_key=v_target;
   IF v_request.actor_subject<>p_subject OR v_request.actor_client<>p_client
   OR v_request.actor_resource<>p_resource
   OR v_request.fingerprint_sha256<>p_fingerprint THEN
     RAISE EXCEPTION 'D30_EXISTING_DIFFERENT_INTENT_DENIED';
   END IF;
 ELSE
   SELECT * INTO STRICT v_request FROM orotitan_mobile_private.command_requests q
   WHERE q.request_id=v_inserted;
 END IF;
 UPDATE orotitan_mobile_private.previews
 SET confirmed_request_id=v_request.request_id WHERE preview_id=p_preview;
 RETURN QUERY SELECT v_request.request_id,v_request.kind,v_request.state,
   (v_inserted IS NULL);
END;
$confirm$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.confirm_request_v1(
 uuid,text,text,text,text) FROM PUBLIC;
COMMIT;
