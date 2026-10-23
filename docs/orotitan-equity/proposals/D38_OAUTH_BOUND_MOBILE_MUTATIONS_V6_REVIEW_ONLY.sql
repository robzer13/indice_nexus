-- D38 REVIEW ONLY: actor/session-bound preview create, vault store/read,
-- and final mobile request confirmations. DISPOSABLE PostgreSQL 17 ONLY.
-- The session table below is a LOCAL TEST MIRROR, NOT live Supabase Auth.
BEGIN;
DO $g$
BEGIN
 IF current_setting('server_version_num')::int NOT BETWEEN 170000 AND 179999
 OR to_regclass('orotitan_mobile_private.d37_attestation_keys') IS NULL
 OR to_regclass('orotitan_mobile_private.research_requests_v4') IS NULL
 OR EXISTS (SELECT 1 FROM pg_roles WHERE rolname='orotitan_d38_mutation_gateway')
 THEN RAISE EXCEPTION 'D38_ONLY_DISPOSABLE_D37_PG17'; END IF;
END;
$g$;
CREATE ROLE orotitan_d38_mutation_gateway NOLOGIN NOINHERIT NOBYPASSRLS;
GRANT USAGE ON SCHEMA orotitan_mobile_private TO orotitan_d38_mutation_gateway;
CREATE TABLE orotitan_mobile_private.d38_test_active_sessions(
 subject text NOT NULL, session_id uuid NOT NULL,
 active boolean NOT NULL DEFAULT true,
 revoked_at timestamptz,
 PRIMARY KEY(subject,session_id),
 CHECK(active=(revoked_at IS NULL))
);
REVOKE ALL ON orotitan_mobile_private.d38_test_active_sessions FROM PUBLIC;
ALTER TABLE orotitan_mobile_private.d38_test_active_sessions ENABLE ROW LEVEL SECURITY;
GRANT SELECT ON orotitan_mobile_private.d38_test_active_sessions
 TO orotitan_d32_private_executor;
CREATE POLICY d38_executor_session_read ON
 orotitan_mobile_private.d38_test_active_sessions
 FOR SELECT TO orotitan_d32_private_executor USING(true);

CREATE FUNCTION orotitan_mobile_private.verify_bound_session_v6(
 p_claims jsonb,p_mac text,p_action text
) RETURNS boolean LANGUAGE plpgsql VOLATILE SECURITY INVOKER
SET search_path=pg_catalog,orotitan_mobile_private AS $proof$
DECLARE v_secret bytea;v_body text;v_iat bigint;v_exp bigint;v_now bigint;
BEGIN
 IF p_claims IS NULL OR jsonb_typeof(p_claims)<>'object'
 OR (SELECT count(*) FROM jsonb_object_keys(p_claims))<>11
 OR EXISTS(SELECT 1 FROM jsonb_each(p_claims) a WHERE jsonb_typeof(a.value)<>'string')
 OR p_mac IS NULL OR p_mac !~ '^[0-9a-f]{64}$'
 OR p_action NOT IN('create_preview','approve_vault','read_vault',
                    'confirm_company','confirm_research')
 OR p_claims->>'version' IS DISTINCT FROM 'd38v1'
 OR p_claims->>'operation' IS DISTINCT FROM p_action
 OR (p_claims->>'previewId') !~
  '^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
 OR (p_claims->>'sessionId') !~
  '^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
 OR (p_claims->>'subject') !~ '^[A-Za-z0-9_.:@|-]{4,160}$'
 OR (p_claims->>'clientId') !~ '^[A-Za-z0-9_-]{6,160}$'
 OR p_claims->>'resource' IS DISTINCT FROM
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
 OR (p_claims->>'bindingSha256') !~ '^[0-9a-f]{64}$'
 OR (p_claims->>'issuedAt') !~ '^[0-9]{10}$'
 OR (p_claims->>'expiresAt') !~ '^[0-9]{10}$'
 OR (p_claims->>'nonce') !~ '^[0-9a-f]{32}$'
 THEN RAISE EXCEPTION 'D38_INVALID_PROOF_FIELDS'; END IF;
 v_iat:=(p_claims->>'issuedAt')::bigint;
 v_exp:=(p_claims->>'expiresAt')::bigint;
 v_now:=floor(extract(epoch from clock_timestamp()))::bigint;
 IF v_iat>v_now+5 OR v_iat<v_now-40 OR v_exp<=v_iat
 OR v_exp>v_iat+30 OR v_exp<=v_now THEN
  RAISE EXCEPTION 'D38_PROOF_EXPIRED'; END IF;
 SELECT secret INTO STRICT v_secret FROM orotitan_mobile_private.d37_attestation_keys
 WHERE key_id='local_v1';
 v_body:=concat_ws(E'\n','orotitan.d38.bound.v1',
 p_claims->>'version',p_claims->>'operation',
 p_claims->>'previewId',p_claims->>'subject',
 p_claims->>'clientId',p_claims->>'resource',p_claims->>'sessionId',
 p_claims->>'bindingSha256',p_claims->>'issuedAt',
 p_claims->>'expiresAt',p_claims->>'nonce');
 IF public.hmac(convert_to(v_body,'UTF8'),v_secret,'sha256')
 <> decode(p_mac,'hex') THEN RAISE EXCEPTION 'D38_BAD_SIGNATURE'; END IF;
 IF NOT EXISTS(SELECT 1 FROM orotitan_mobile_private.d38_test_active_sessions s
 WHERE s.subject=p_claims->>'subject'
  AND s.session_id=(p_claims->>'sessionId')::uuid
  AND s.active AND s.revoked_at IS NULL) THEN
  RAISE EXCEPTION 'D38_SESSION_REVOKED'; END IF;
 -- Global nonce namespace shared with D37; one-time, atomic with mutation.
 INSERT INTO orotitan_mobile_private.d37_consumed_nonces
 (nonce,actor_subject,session_id,preview_id)
 VALUES(p_claims->>'nonce',p_claims->>'subject',
   (p_claims->>'sessionId')::uuid,(p_claims->>'previewId')::uuid);
 RETURN true;
END;
$proof$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.verify_bound_session_v6(
 jsonb,text,text) FROM PUBLIC;

CREATE FUNCTION orotitan_mobile_private.create_oauth_bound_preview_v6(
 p_claims jsonb,p_mac text,p_kind text,p_payload jsonb,
 p_fingerprint text,p_expires_ms bigint
) RETURNS boolean LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $preview$
DECLARE v_digest text;
BEGIN
 PERFORM orotitan_mobile_private.verify_bound_session_v6(
 p_claims,p_mac,'create_preview');
 IF p_kind NOT IN ('add_company','request_research')
 OR p_fingerprint IS NULL OR p_fingerprint !~ '^[0-9a-f]{64}$'
 OR p_expires_ms IS NULL THEN
  RAISE EXCEPTION 'D38_PREVIEW_BINDING_MISMATCH'; END IF;
 v_digest:=encode(public.digest(convert_to(
  concat_ws(E'\n',p_kind,p_fingerprint,p_expires_ms::text),'UTF8'),'sha256'),'hex');
 IF p_claims->>'bindingSha256' IS DISTINCT FROM v_digest THEN
  RAISE EXCEPTION 'D38_PREVIEW_BINDING_MISMATCH'; END IF;
 RETURN orotitan_mobile_private.create_preview_v3(
  (p_claims->>'previewId')::uuid,p_kind,p_claims->>'subject',
  p_claims->>'clientId',p_claims->>'resource',p_payload,
  p_fingerprint,to_timestamp(p_expires_ms/1000.0));
END;
$preview$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.create_oauth_bound_preview_v6(
 jsonb,text,text,jsonb,text,bigint) FROM PUBLIC;

CREATE FUNCTION orotitan_mobile_private.approve_oauth_bound_vault_v6(
 p_claims jsonb,p_mac text,p_kind text,p_fingerprint text,
 p_capability_digest text,p_key_version text,p_nonce bytea,
 p_ciphertext bytea,p_tag bytea
) RETURNS boolean LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $vault$
DECLARE v_digest text;
BEGIN
 PERFORM orotitan_mobile_private.verify_bound_session_v6(
  p_claims,p_mac,'approve_vault');
 IF p_kind NOT IN('add_company','request_research')
 OR p_fingerprint IS NULL OR p_fingerprint !~ '^[a-f0-9]{64}$'
 OR p_capability_digest IS NULL OR p_capability_digest !~ '^[a-f0-9]{64}$'
 OR p_key_version IS NULL OR p_nonce IS NULL
 OR p_ciphertext IS NULL OR p_tag IS NULL THEN
  RAISE EXCEPTION 'D38_VAULT_BINDING_MISMATCH'; END IF;
 v_digest:=encode(public.digest(convert_to(concat_ws(E'\n',
  p_kind,p_fingerprint,p_capability_digest,p_key_version,
  encode(p_nonce,'hex'),encode(p_ciphertext,'hex'),encode(p_tag,'hex')),
  'UTF8'),'sha256'),'hex');
 IF p_claims->>'bindingSha256' IS DISTINCT FROM v_digest THEN
  RAISE EXCEPTION 'D38_VAULT_BINDING_MISMATCH'; END IF;
 RETURN orotitan_mobile_private.approve_and_store_capability_v3(
  (p_claims->>'previewId')::uuid,p_kind,p_claims->>'subject',
  p_claims->>'clientId',p_claims->>'resource',
  p_fingerprint,p_capability_digest,p_key_version,p_nonce,p_ciphertext,p_tag);
END;
$vault$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.approve_oauth_bound_vault_v6(
 jsonb,text,text,text,text,text,bytea,bytea,bytea) FROM PUBLIC;

CREATE FUNCTION orotitan_mobile_private.read_oauth_bound_vault_v6(
 p_claims jsonb,p_mac text
) RETURNS jsonb LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $vaultread$
BEGIN
 PERFORM orotitan_mobile_private.verify_bound_session_v6(
  p_claims,p_mac,'read_vault');
 IF p_claims->>'bindingSha256' IS DISTINCT FROM repeat('0',64) THEN
   RAISE EXCEPTION 'D38_VAULT_READ_BINDING_MISMATCH'; END IF;
 RETURN orotitan_mobile_private.read_encrypted_capability_v3(
  (p_claims->>'previewId')::uuid,p_claims->>'subject',
  p_claims->>'clientId',p_claims->>'resource');
END;
$vaultread$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.read_oauth_bound_vault_v6(
 jsonb,text) FROM PUBLIC;

CREATE FUNCTION orotitan_mobile_private.confirm_oauth_bound_request_v6(
 p_claims jsonb,p_mac text,p_capability text
) RETURNS jsonb LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $confirm$
DECLARE v_action text;v_preview orotitan_mobile_private.previews%ROWTYPE;
 v_result jsonb;
BEGIN
 v_action:=p_claims->>'operation';
 IF v_action NOT IN('confirm_company','confirm_research') THEN
  RAISE EXCEPTION 'D38_CONFIRM_ACTION_DENIED'; END IF;
 PERFORM orotitan_mobile_private.verify_bound_session_v6(p_claims,p_mac,v_action);
 IF p_capability IS NULL OR length(p_capability)>2200 OR
  p_capability !~ '^[A-Za-z0-9_-]+[.][A-Za-z0-9_-]{43}$'
 OR p_claims->>'bindingSha256' IS DISTINCT FROM
  encode(public.digest(convert_to(p_capability,'UTF8'),'sha256'),'hex') THEN
  RAISE EXCEPTION 'D38_CAPABILITY_BINDING_MISMATCH'; END IF;
 SELECT * INTO v_preview FROM orotitan_mobile_private.previews
 WHERE preview_id=(p_claims->>'previewId')::uuid FOR UPDATE;
 IF NOT FOUND OR v_preview.actor_subject IS DISTINCT FROM p_claims->>'subject'
 OR v_preview.actor_client IS DISTINCT FROM p_claims->>'clientId'
 OR v_preview.actor_resource IS DISTINCT FROM p_claims->>'resource'
 OR (v_action='confirm_company' AND v_preview.kind<>'add_company')
 OR (v_action='confirm_research' AND v_preview.kind<>'request_research') THEN
  RAISE EXCEPTION 'D38_CONFIRM_ACTOR_OR_KIND_DENIED'; END IF;
 IF v_action='confirm_company' THEN
  SELECT jsonb_build_object('requestId',r.request_id,'kind',r.kind,
   'state',r.state,'duplicate',r.duplicate) INTO STRICT v_result
  FROM orotitan_mobile_private.confirm_approved_request_v2(
   v_preview.preview_id,p_capability) r;
 ELSE
  SELECT jsonb_build_object('requestId',r.request_id,
   'kind','request_research','state',r.state,'duplicate',r.duplicate)
   INTO STRICT v_result
  FROM orotitan_mobile_private.confirm_scoped_research_v4(
   v_preview.preview_id,p_capability) r;
 END IF;
 RETURN v_result;
END;
$confirm$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.confirm_oauth_bound_request_v6(
 jsonb,text,text) FROM PUBLIC;

GRANT CREATE ON SCHEMA orotitan_mobile_private TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.verify_bound_session_v6(jsonb,text,text)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.create_oauth_bound_preview_v6(
 jsonb,text,text,jsonb,text,bigint) OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.approve_oauth_bound_vault_v6(
 jsonb,text,text,text,text,text,bytea,bytea,bytea)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.read_oauth_bound_vault_v6(jsonb,text)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.confirm_oauth_bound_request_v6(
 jsonb,text,text) OWNER TO orotitan_d32_private_executor;
REVOKE CREATE ON SCHEMA orotitan_mobile_private FROM orotitan_d32_private_executor;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.create_oauth_bound_preview_v6(
 jsonb,text,text,jsonb,text,bigint) TO orotitan_d38_mutation_gateway;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.approve_oauth_bound_vault_v6(
 jsonb,text,text,text,text,text,bytea,bytea,bytea)
 TO orotitan_d38_mutation_gateway;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.read_oauth_bound_vault_v6(
 jsonb,text) TO orotitan_d38_mutation_gateway;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.confirm_oauth_bound_request_v6(
 jsonb,text,text) TO orotitan_d38_mutation_gateway;

-- Simulate future cutover in this disposable DB: the older call-only
-- principals cannot sidestep the signed session operation at these paths.
REVOKE EXECUTE ON FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz)
 FROM orotitan_d34_preview_issuer;
REVOKE EXECUTE ON FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)
 FROM orotitan_d34_vault_issuer;
REVOKE EXECUTE ON FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) FROM orotitan_d34_vault_reader;
REVOKE EXECUTE ON FUNCTION orotitan_mobile_private.confirm_approved_request_v2(
 uuid,text) FROM orotitan_d32_command_submitter;
REVOKE EXECUTE ON FUNCTION orotitan_mobile_private.confirm_scoped_research_v4(
 uuid,text) FROM orotitan_d36_research_submitter;
COMMIT;
