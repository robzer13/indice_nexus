-- D36 / REVIEW ONLY / DISPOSABLE PG17 AFTER D34. NO HOSTED MIGRATION.
-- Tenant-scoped Research request lifecycle. Distinct OAuth subjects can
-- request the same issuer independently; one subject can re-request after
-- a separately authorized lifecycle closure. No Research run is started.
BEGIN;
DO $guard$
BEGIN
 IF current_setting('server_version_num')::int NOT BETWEEN 170000 AND 179999
 OR to_regclass('orotitan_mobile_private.previews') IS NULL
 OR to_regclass('orotitan_mobile_private.encrypted_capability_vault') IS NULL
 OR to_regclass('orotitan_mobile_private.research_requests_v4') IS NOT NULL
 OR EXISTS(SELECT 1 FROM pg_roles WHERE rolname IN
   ('orotitan_d36_scope_issuer','orotitan_d36_research_submitter',
    'orotitan_d36_lifecycle_controller')) THEN
   RAISE EXCEPTION 'D36_REQUIRES_FRESH_LOCAL_PG17_WITH_D34';
 END IF;
END;
$guard$;

CREATE ROLE orotitan_d36_scope_issuer NOLOGIN NOINHERIT NOBYPASSRLS;
CREATE ROLE orotitan_d36_research_submitter NOLOGIN NOINHERIT NOBYPASSRLS;
CREATE ROLE orotitan_d36_lifecycle_controller NOLOGIN NOINHERIT NOBYPASSRLS;
GRANT USAGE ON SCHEMA orotitan_mobile_private
 TO orotitan_d36_scope_issuer,orotitan_d36_research_submitter,
 orotitan_d36_lifecycle_controller;

CREATE TABLE orotitan_mobile_private.research_requests_v4(
 request_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 actor_subject text NOT NULL,
 actor_resource text NOT NULL,
 issuer_id uuid NOT NULL,
 contract_pack_sha256 text NOT NULL CHECK(contract_pack_sha256 ~ '^[a-f0-9]{64}$'),
 cutoff_date date NOT NULL,
 state text NOT NULL DEFAULT 'REQUESTED_NOT_EXECUTED'
   CHECK(state IN ('REQUESTED_NOT_EXECUTED','CANCELLED','FULFILLED','REJECTED')),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 closed_at timestamptz,
 CHECK ((state='REQUESTED_NOT_EXECUTED')=(closed_at IS NULL))
);
-- Only ONE ACTIVE request per operator/issuer/contract/cutoff; terminal
-- history remains immutable and does not block a subsequent request.
CREATE UNIQUE INDEX d36_active_subject_research_scope
 ON orotitan_mobile_private.research_requests_v4(
 actor_subject,actor_resource,issuer_id,contract_pack_sha256,cutoff_date)
 WHERE state='REQUESTED_NOT_EXECUTED';

CREATE TABLE orotitan_mobile_private.research_preview_scope_v4(
 preview_id uuid PRIMARY KEY REFERENCES orotitan_mobile_private.previews(preview_id)
  ON DELETE RESTRICT,
 issuer_id uuid NOT NULL,
 contract_pack_sha256 text NOT NULL CHECK(contract_pack_sha256 ~ '^[a-f0-9]{64}$'),
 cutoff_date date NOT NULL,
 confirmed_request_id uuid REFERENCES orotitan_mobile_private.research_requests_v4(request_id)
  ON DELETE RESTRICT,
 created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
REVOKE ALL ON orotitan_mobile_private.research_requests_v4 FROM PUBLIC;
REVOKE ALL ON orotitan_mobile_private.research_preview_scope_v4 FROM PUBLIC;
ALTER TABLE orotitan_mobile_private.research_requests_v4 ENABLE ROW LEVEL SECURITY;
ALTER TABLE orotitan_mobile_private.research_preview_scope_v4 ENABLE ROW LEVEL SECURITY;
GRANT SELECT,INSERT,UPDATE ON orotitan_mobile_private.research_requests_v4
 TO orotitan_d32_private_executor;
GRANT SELECT,INSERT,UPDATE ON orotitan_mobile_private.research_preview_scope_v4
 TO orotitan_d32_private_executor;
CREATE POLICY d36_executor_requests ON orotitan_mobile_private.research_requests_v4
 FOR ALL TO orotitan_d32_private_executor USING(true) WITH CHECK(true);
CREATE POLICY d36_executor_scopes ON orotitan_mobile_private.research_preview_scope_v4
 FOR ALL TO orotitan_d32_private_executor USING(true) WITH CHECK(true);

-- Scope is registered by a high-trust backend AFTER actual OAuth verification.
-- Caller-supplied identity remains a HIGH-release blocker until the HTTP
-- backend binds it to a verified token/session; this role MUST NEVER be
-- granted to public, anon, authenticated or the ChatGPT runtime client.
CREATE FUNCTION orotitan_mobile_private.issue_research_scope_v4(
 p_preview uuid,p_subject text,p_client text,p_resource text,
 p_issuer uuid,p_contract_sha256 text,p_cutoff date
) RETURNS boolean
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $scope$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
BEGIN
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview FOR UPDATE;
 IF NOT FOUND OR v.kind<>'request_research'
 OR v.actor_subject IS DISTINCT FROM p_subject
 OR v.actor_client IS DISTINCT FROM p_client
 OR v.actor_resource IS DISTINCT FROM p_resource
 OR (v.payload->>'issuerId')::uuid IS DISTINCT FROM p_issuer
 OR p_contract_sha256 IS NULL OR p_contract_sha256 !~ '^[a-f0-9]{64}$'
 OR p_cutoff IS NULL OR p_cutoff>current_date
 OR p_cutoff<date '2000-01-01'
 OR v.approved_capability_sha256 IS NOT NULL
 OR v.confirmed_request_id IS NOT NULL OR v.revoked_at IS NOT NULL
 OR clock_timestamp()>=v.expires_at THEN
   RAISE EXCEPTION 'D36_SCOPE_NOT_AUTHORIZED_OR_INVALID';
 END IF;
 IF NOT EXISTS(SELECT 1 FROM public.issuers i
  JOIN public.securities s ON s.issuer_id=i.issuer_id
  WHERE i.issuer_id=p_issuer
   AND i.display_name=v.payload->>'name'
   AND s.ticker=v.payload->>'ticker'
   AND s.exchange=v.payload->>'exchange'
   AND s.isin=v.payload->>'isin' AND s.isin<>'UNKNOWN') THEN
    RAISE EXCEPTION 'D36_CANONICAL_IDENTITY_CHANGED';
 END IF;
 INSERT INTO orotitan_mobile_private.research_preview_scope_v4(
 preview_id,issuer_id,contract_pack_sha256,cutoff_date
 ) VALUES(p_preview,p_issuer,p_contract_sha256,p_cutoff);
 RETURN true;
END;
$scope$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.issue_research_scope_v4(
 uuid,text,text,text,uuid,text,date) FROM PUBLIC;

-- Uses the same D32 HMAC-capability digest + preview replay guard, but
-- deliberately DOES NOT call the legacy D30 global UNIQUE(kind,target).
-- The PREVIEW stores authenticated actor identity, not this caller's input.
CREATE FUNCTION orotitan_mobile_private.confirm_scoped_research_v4(
 p_preview uuid,p_sealed_capability text
) RETURNS TABLE(request_id uuid,state text,duplicate boolean)
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $confirm$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
 v_scope orotitan_mobile_private.research_preview_scope_v4%ROWTYPE;
 v_request orotitan_mobile_private.research_requests_v4%ROWTYPE;
 v_inserted uuid;
BEGIN
 IF p_sealed_capability IS NULL OR length(p_sealed_capability)>2200
 OR length(split_part(p_sealed_capability,'.',1))<20
 OR p_sealed_capability !~ '^[A-Za-z0-9_-]+[.][A-Za-z0-9_-]{43}$' THEN
  RAISE EXCEPTION 'D36_CAPABILITY_SHAPE_DENIED';
 END IF;
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview FOR UPDATE;
 IF NOT FOUND OR v.kind<>'request_research'
 OR v.approved_capability_sha256 IS NULL OR v.approved_at IS NULL
 OR v.revoked_at IS NOT NULL OR v.approved_at>clock_timestamp()
 OR clock_timestamp()>=v.expires_at
 OR public.digest(convert_to(p_sealed_capability,'UTF8'),'sha256')
   <> v.approved_capability_sha256 THEN
  RAISE EXCEPTION 'D36_CAPABILITY_DENIED';
 END IF;
 SELECT * INTO v_scope FROM orotitan_mobile_private.research_preview_scope_v4
 WHERE preview_id=p_preview FOR UPDATE;
 IF NOT FOUND OR (v.payload->>'issuerId')::uuid IS DISTINCT FROM v_scope.issuer_id
 OR v_scope.contract_pack_sha256 !~ '^[a-f0-9]{64}$'
 OR v_scope.cutoff_date>current_date THEN
  RAISE EXCEPTION 'D36_SCOPE_MISSING_OR_CHANGED';
 END IF;
 -- Always recheck the CURRENT canonical issuer/security at commit.
 IF NOT EXISTS(SELECT 1 FROM public.issuers i
  JOIN public.securities s ON s.issuer_id=i.issuer_id
  WHERE i.issuer_id=v_scope.issuer_id
   AND i.display_name=v.payload->>'name'
   AND s.ticker=v.payload->>'ticker'
   AND s.exchange=v.payload->>'exchange'
   AND s.isin=v.payload->>'isin' AND s.isin<>'UNKNOWN') THEN
    RAISE EXCEPTION 'D36_CANONICAL_IDENTITY_CHANGED';
 END IF;
 IF v_scope.confirmed_request_id IS NOT NULL THEN
  SELECT * INTO STRICT v_request FROM orotitan_mobile_private.research_requests_v4 r
   WHERE r.request_id=v_scope.confirmed_request_id
    AND r.actor_subject=v.actor_subject AND r.actor_resource=v.actor_resource;
  RETURN QUERY SELECT v_request.request_id,v_request.state,true;
  RETURN;
 END IF;
 INSERT INTO orotitan_mobile_private.research_requests_v4 AS r(
  actor_subject,actor_resource,issuer_id,contract_pack_sha256,cutoff_date
 ) VALUES(v.actor_subject,v.actor_resource,v_scope.issuer_id,
   v_scope.contract_pack_sha256,v_scope.cutoff_date)
 ON CONFLICT(actor_subject,actor_resource,issuer_id,contract_pack_sha256,cutoff_date)
 WHERE state='REQUESTED_NOT_EXECUTED' DO NOTHING
 RETURNING r.request_id INTO v_inserted;
 IF v_inserted IS NULL THEN
  SELECT * INTO STRICT v_request FROM orotitan_mobile_private.research_requests_v4 r
  WHERE r.actor_subject=v.actor_subject AND r.actor_resource=v.actor_resource
   AND r.issuer_id=v_scope.issuer_id
   AND r.contract_pack_sha256=v_scope.contract_pack_sha256
   AND r.cutoff_date=v_scope.cutoff_date
   AND r.state='REQUESTED_NOT_EXECUTED';
 ELSE
  SELECT * INTO STRICT v_request FROM orotitan_mobile_private.research_requests_v4 r
  WHERE r.request_id=v_inserted;
 END IF;
 UPDATE orotitan_mobile_private.research_preview_scope_v4
 SET confirmed_request_id=v_request.request_id WHERE preview_id=p_preview;
 RETURN QUERY SELECT v_request.request_id,v_request.state,
   (v_inserted IS NULL);
END;
$confirm$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.confirm_scoped_research_v4(
 uuid,text) FROM PUBLIC;

-- A distinct, independently approved WORKER/controller, not the mobile
-- requester, closes a specific request. It has no Research runner powers.
CREATE FUNCTION orotitan_mobile_private.close_research_request_v4(
 p_request uuid,p_terminal_state text
) RETURNS boolean LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $close$
DECLARE v_updated uuid;
BEGIN
 IF p_terminal_state NOT IN ('FULFILLED','CANCELLED','REJECTED') THEN
  RAISE EXCEPTION 'D36_TERMINAL_STATE_INVALID';
 END IF;
 UPDATE orotitan_mobile_private.research_requests_v4 r
 SET state=p_terminal_state,closed_at=clock_timestamp()
 WHERE r.request_id=p_request AND r.state='REQUESTED_NOT_EXECUTED'
 RETURNING r.request_id INTO v_updated;
 IF v_updated IS NULL THEN
  RAISE EXCEPTION 'D36_UNKNOWN_OR_ALREADY_CLOSED_REQUEST';
 END IF;
 RETURN true;
END;
$close$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.close_research_request_v4(
 uuid,text) FROM PUBLIC;

GRANT CREATE ON SCHEMA orotitan_mobile_private TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.issue_research_scope_v4(
 uuid,text,text,text,uuid,text,date) OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.confirm_scoped_research_v4(
 uuid,text) OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.close_research_request_v4(
 uuid,text) OWNER TO orotitan_d32_private_executor;
REVOKE CREATE ON SCHEMA orotitan_mobile_private FROM orotitan_d32_private_executor;

GRANT EXECUTE ON FUNCTION orotitan_mobile_private.issue_research_scope_v4(
 uuid,text,text,text,uuid,text,date) TO orotitan_d36_scope_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.confirm_scoped_research_v4(
 uuid,text) TO orotitan_d36_research_submitter;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.close_research_request_v4(
 uuid,text) TO orotitan_d36_lifecycle_controller;
COMMIT;
