-- D32 / REVIEW ONLY: private, capability-bound command queue endpoint.
-- DO NOT APPLY TO SUPABASE, MAIN OR ANY HOSTED DATABASE.
-- Run solely inside disposable PostgreSQL 17 fixture with D30 installed.
-- Roles are LOCAL CLUSTER TEST OBJECTS; caller must drop them after DB teardown.
BEGIN;
DO $guard$
BEGIN
 IF current_setting('server_version_num')::int NOT BETWEEN 170000 AND 179999
 OR to_regclass('orotitan_mobile_private.previews') IS NULL
 OR to_regclass('orotitan_mobile_private.command_requests') IS NULL
 OR to_regclass('public.issuers') IS NULL
 OR EXISTS (SELECT 1 FROM pg_roles WHERE rolname IN (
  'orotitan_d32_approval_issuer','orotitan_d32_command_submitter',
  'orotitan_d32_private_executor')) THEN
  RAISE EXCEPTION 'D32_REQUIRES_FRESH_LOCAL_PG17_AND_D30';
 END IF;
END;
$guard$;

CREATE ROLE orotitan_d32_approval_issuer NOLOGIN NOINHERIT NOBYPASSRLS;
CREATE ROLE orotitan_d32_command_submitter NOLOGIN NOINHERIT NOBYPASSRLS;
CREATE ROLE orotitan_d32_private_executor NOLOGIN NOINHERIT NOBYPASSRLS;
ALTER TABLE orotitan_mobile_private.previews
 ADD COLUMN approved_capability_sha256 bytea
   CHECK(approved_capability_sha256 IS NULL
    OR octet_length(approved_capability_sha256)=32),
 ADD COLUMN approved_at timestamptz,
 ADD COLUMN revoked_at timestamptz,
 ADD CONSTRAINT d32_approval_consistency
   CHECK ((approved_at IS NULL)=(approved_capability_sha256 IS NULL)
     AND (revoked_at IS NULL OR approved_at IS NOT NULL));

-- No table writes or reads are granted to the two call-only principals.
GRANT USAGE ON SCHEMA orotitan_mobile_private
 TO orotitan_d32_private_executor,orotitan_d32_approval_issuer,
    orotitan_d32_command_submitter;
GRANT SELECT, UPDATE ON orotitan_mobile_private.previews
 TO orotitan_d32_private_executor;
GRANT SELECT, INSERT ON orotitan_mobile_private.command_requests
 TO orotitan_d32_private_executor;
CREATE POLICY d32_executor_preview ON orotitan_mobile_private.previews
 FOR ALL TO orotitan_d32_private_executor USING (true) WITH CHECK (true);
CREATE POLICY d32_executor_requests_select ON orotitan_mobile_private.command_requests
 FOR SELECT TO orotitan_d32_private_executor USING (true);
CREATE POLICY d32_executor_requests_insert ON orotitan_mobile_private.command_requests
 FOR INSERT TO orotitan_d32_private_executor WITH CHECK (true);

-- Existing I1 identity tables have RLS. Executor is NOLOGIN/non-bypass;
-- only these narrow columns may be read by the private definer RPC.
GRANT SELECT (issuer_id,display_name) ON public.issuers
 TO orotitan_d32_private_executor;
GRANT SELECT (issuer_id,ticker,exchange,isin) ON public.securities
 TO orotitan_d32_private_executor;
GRANT SELECT (exchange,ticker) ON public.companies
 TO orotitan_d32_private_executor;
CREATE POLICY d32_executor_issuers ON public.issuers
 FOR SELECT TO orotitan_d32_private_executor USING (true);
CREATE POLICY d32_executor_securities ON public.securities
 FOR SELECT TO orotitan_d32_private_executor USING (true);

-- Minimal EXECUTE for executor to call D30 legacy function; neither
-- approval_issuer nor submitter may invoke the forgeable D30 function.
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.confirm_request_v1(
 uuid,text,text,text,text) TO orotitan_d32_private_executor;

-- Only a trusted backend, after verified OAuth+independent D31 policy, can
-- register the SHA-256 of a *server-only* D31 sealed capability. This is
-- NOT the ability to create new previews or actual companies/Registry runs.
CREATE FUNCTION orotitan_mobile_private.approve_preview_v2(
 p_preview uuid,p_kind text,p_subject text,p_client text,p_resource text,
 p_fingerprint text,p_capability_sha256 text
) RETURNS boolean
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $approve$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
BEGIN
 IF p_capability_sha256 IS NULL
 OR p_capability_sha256 !~ '^[0-9a-f]{64}$' THEN
   RAISE EXCEPTION 'D32_CAPABILITY_DIGEST_INVALID';
 END IF;
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview FOR UPDATE;
 IF NOT FOUND OR v.kind IS DISTINCT FROM p_kind
 OR v.actor_subject IS DISTINCT FROM p_subject
 OR v.actor_client IS DISTINCT FROM p_client
 OR v.actor_resource IS DISTINCT FROM p_resource
 OR v.fingerprint_sha256 IS DISTINCT FROM p_fingerprint
 OR v.revoked_at IS NOT NULL
 OR v.confirmed_request_id IS NOT NULL
 OR clock_timestamp()>=v.expires_at THEN
   RAISE EXCEPTION 'D32_APPROVAL_ACTOR_OR_PREVIEW_DENIED';
 END IF;
 IF v.approved_capability_sha256 IS NOT NULL THEN
   IF v.approved_capability_sha256 <> decode(p_capability_sha256,'hex') THEN
     RAISE EXCEPTION 'D32_ALREADY_APPROVED_WITH_DIFFERENT_PROOF';
   END IF;
   RETURN false; -- idempotent same private proof, no reissuance.
 END IF;
 UPDATE orotitan_mobile_private.previews
   SET approved_capability_sha256=decode(p_capability_sha256,'hex'),
       approved_at=clock_timestamp()
 WHERE preview_id=p_preview;
 RETURN true;
END;
$approve$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.approve_preview_v2(
 uuid,text,text,text,text,text,text) FROM PUBLIC;

CREATE FUNCTION orotitan_mobile_private.revoke_preview_v2(p_preview uuid)
RETURNS boolean LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $revoke$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
BEGIN
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview FOR UPDATE;
 IF NOT FOUND OR v.approved_capability_sha256 IS NULL
 OR v.confirmed_request_id IS NOT NULL THEN
  RAISE EXCEPTION 'D32_REVOKE_NOT_APPROVED_OR_ALREADY_CONFIRMED';
 END IF;
 UPDATE orotitan_mobile_private.previews
 SET revoked_at=coalesce(revoked_at,clock_timestamp())
 WHERE preview_id=p_preview;
 RETURN true;
END;
$revoke$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.revoke_preview_v2(uuid) FROM PUBLIC;

-- No actor identity parameters: this function uses only actor fields stored
-- in a previously approved private preview. The caller supplies only the
-- opaque server-held D31 capability; hash verified against private digest.
CREATE FUNCTION orotitan_mobile_private.confirm_approved_request_v2(
 p_preview uuid,p_sealed_capability text
) RETURNS TABLE(request_id uuid,kind text,state text,duplicate boolean)
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $confirm$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
BEGIN
 IF p_sealed_capability IS NULL OR length(p_sealed_capability)>2200
 OR p_sealed_capability !~ '^[A-Za-z0-9_-]{20,1700}[.][A-Za-z0-9_-]{43}$'
 THEN RAISE EXCEPTION 'D32_SERVER_CAPABILITY_SHAPE_DENIED'; END IF;
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview FOR UPDATE;
 IF NOT FOUND OR v.approved_capability_sha256 IS NULL
 OR v.approved_at IS NULL OR v.revoked_at IS NOT NULL
 OR v.approved_at>clock_timestamp() OR clock_timestamp()>=v.expires_at
 OR digest(convert_to(p_sealed_capability,'UTF8'),'sha256')
    <> v.approved_capability_sha256 THEN
  RAISE EXCEPTION 'D32_CAPABILITY_ABSENT_REVOKED_EXPIRED_OR_MISMATCH';
 END IF;
 -- D30 performs the current-company / issuer / security recheck and CAS.
 -- This older function is never callable by the untrusted submitter role.
 RETURN QUERY SELECT c.request_id,c.kind,c.state,c.duplicate
 FROM orotitan_mobile_private.confirm_request_v1(
  v.preview_id,v.actor_subject,v.actor_client,v.actor_resource,
  v.fingerprint_sha256) c;
END;
$confirm$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.confirm_approved_request_v2(
 uuid,text) FROM PUBLIC;

-- Ownership transfer to a purpose-built non-login, RLS-obeying executor.
-- Create is granted only briefly to satisfy PostgreSQL owner transfer.
GRANT CREATE ON SCHEMA orotitan_mobile_private
 TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.approve_preview_v2(
 uuid,text,text,text,text,text,text) OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.revoke_preview_v2(uuid)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.confirm_approved_request_v2(
 uuid,text) OWNER TO orotitan_d32_private_executor;
REVOKE CREATE ON SCHEMA orotitan_mobile_private
 FROM orotitan_d32_private_executor;

GRANT EXECUTE ON FUNCTION orotitan_mobile_private.approve_preview_v2(
 uuid,text,text,text,text,text,text) TO orotitan_d32_approval_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.revoke_preview_v2(uuid)
 TO orotitan_d32_approval_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.confirm_approved_request_v2(
 uuid,text) TO orotitan_d32_command_submitter;
COMMIT;
