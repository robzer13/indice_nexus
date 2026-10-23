-- D37 / REVIEW ONLY / DISPOSABLE PG17 AFTER D36. DO NOT MIGRATE.
-- Session attestation is minted by a trusted OAuth verifier + independent
-- live-session policy; SQL verifies its HMAC/TTL/nonce and derives ALL actor,
-- scope and resource values from the signed proof, not caller parameters.
BEGIN;
DO $guard$
BEGIN
 IF current_setting('server_version_num')::int NOT BETWEEN 170000 AND 179999
 OR to_regclass('orotitan_mobile_private.research_preview_scope_v4') IS NULL
 OR EXISTS(SELECT 1 FROM pg_roles WHERE rolname='orotitan_d37_session_gateway') THEN
   RAISE EXCEPTION 'D37_FRESH_DISPOSABLE_PG17_D36_REQUIRED';
 END IF;
END;
$guard$;
CREATE ROLE orotitan_d37_session_gateway NOLOGIN NOINHERIT NOBYPASSRLS;
GRANT USAGE ON SCHEMA orotitan_mobile_private TO orotitan_d37_session_gateway;

CREATE TABLE orotitan_mobile_private.d37_attestation_keys(
 key_id text PRIMARY KEY CHECK(key_id='local_v1'),
 secret bytea NOT NULL CHECK(octet_length(secret)=32)
);
CREATE TABLE orotitan_mobile_private.d37_consumed_nonces(
 nonce text PRIMARY KEY CHECK(nonce ~ '^[a-f0-9]{32}$'),
 actor_subject text NOT NULL,
 session_id uuid NOT NULL,
 preview_id uuid NOT NULL,
 consumed_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
REVOKE ALL ON orotitan_mobile_private.d37_attestation_keys FROM PUBLIC;
REVOKE ALL ON orotitan_mobile_private.d37_consumed_nonces FROM PUBLIC;
ALTER TABLE orotitan_mobile_private.d37_attestation_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE orotitan_mobile_private.d37_consumed_nonces ENABLE ROW LEVEL SECURITY;
GRANT SELECT ON orotitan_mobile_private.d37_attestation_keys
 TO orotitan_d32_private_executor;
GRANT INSERT ON orotitan_mobile_private.d37_consumed_nonces
 TO orotitan_d32_private_executor;
CREATE POLICY d37_executor_key_read ON orotitan_mobile_private.d37_attestation_keys
 FOR SELECT TO orotitan_d32_private_executor USING (true);
CREATE POLICY d37_executor_nonce_insert ON orotitan_mobile_private.d37_consumed_nonces
 FOR INSERT TO orotitan_d32_private_executor WITH CHECK (true);

-- Executes as private executor ONLY through D37 SECURITY DEFINER wrappers.
-- No grant to the gateway, authenticated, anon, PUBLIC or browser client.
CREATE FUNCTION orotitan_mobile_private.consume_verified_session_v5(
 p_claims jsonb,p_mac_sha256 text,p_action text
) RETURNS boolean LANGUAGE plpgsql VOLATILE SECURITY INVOKER
SET search_path=pg_catalog,orotitan_mobile_private AS $attest$
DECLARE v_secret bytea;v_body text;v_now bigint;v_iat bigint;v_exp bigint;
BEGIN
 IF p_claims IS NULL OR jsonb_typeof(p_claims)<>'object'
 OR (SELECT count(*) FROM jsonb_object_keys(p_claims))<>13
 OR EXISTS(SELECT 1 FROM jsonb_each(p_claims) AS v(k,val)
   WHERE jsonb_typeof(v.val)<>'string')
 OR p_mac_sha256 IS NULL OR p_mac_sha256 !~ '^[a-f0-9]{64}$'
 OR p_action NOT IN ('read_preview','issue_research_scope')
 OR p_claims->>'version' IS DISTINCT FROM 'd37v1'
 OR p_claims->>'operation' IS DISTINCT FROM p_action
 OR (p_claims->>'previewId') !~
   '^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$'
 OR (p_claims->>'sessionId') !~
   '^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$'
 OR (p_claims->>'subject') !~ '^[a-zA-Z0-9_.:@|\\-]{4,160}$'
 OR (p_claims->>'clientId') !~ '^[a-zA-Z0-9_-]{6,160}$'
 OR p_claims->>'resource' IS DISTINCT FROM
  'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
 OR (p_claims->>'issuerId') !~ '^[a-f0-9-]{36}$'
 OR (p_claims->>'contractSha256') !~ '^[a-f0-9]{64}$'
 OR (p_claims->>'cutoff') !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
 OR (p_claims->>'issuedAt') !~ '^[0-9]{10}$'
 OR (p_claims->>'expiresAt') !~ '^[0-9]{10}$'
 OR (p_claims->>'nonce') !~ '^[a-f0-9]{32}$' THEN
   RAISE EXCEPTION 'D37_UNTRUSTED_ATTESTATION_DENIED';
 END IF;
 v_iat:=(p_claims->>'issuedAt')::bigint;
 v_exp:=(p_claims->>'expiresAt')::bigint;
 v_now:=floor(extract(epoch from clock_timestamp()))::bigint;
 IF v_iat>v_now+5 OR v_iat<v_now-40 OR
   v_exp<=v_iat OR v_exp>v_iat+30 OR v_exp<=v_now THEN
   RAISE EXCEPTION 'D37_ATTESTATION_EXPIRED';
 END IF;
 IF p_action='read_preview' AND (
  p_claims->>'issuerId'<>'00000000-0000-0000-0000-000000000000'
  OR p_claims->>'contractSha256'<>repeat('0',64)
  OR p_claims->>'cutoff'<>'1970-01-01') THEN
   RAISE EXCEPTION 'D37_READ_MUST_HAVE_EMPTY_SCOPE';
 END IF;
 SELECT k.secret INTO STRICT v_secret
 FROM orotitan_mobile_private.d37_attestation_keys k
 WHERE k.key_id='local_v1';
 v_body:=concat_ws(E'\n','orotitan.d37.session.v1',
   p_claims->>'version',p_claims->>'operation',
   p_claims->>'previewId',p_claims->>'subject',
   p_claims->>'clientId',p_claims->>'resource',
   p_claims->>'sessionId',p_claims->>'issuerId',
   p_claims->>'contractSha256',p_claims->>'cutoff',
   p_claims->>'issuedAt',p_claims->>'expiresAt',
   p_claims->>'nonce');
 IF public.hmac(convert_to(v_body,'UTF8'),v_secret,'sha256')
   <> decode(p_mac_sha256,'hex') THEN
   RAISE EXCEPTION 'D37_ATTESTATION_MAC_DENIED';
 END IF;
 -- Unique nonce consumption is in the same TX as the private operation.
 -- Failed downstream operations roll back consumption for error recovery.
 INSERT INTO orotitan_mobile_private.d37_consumed_nonces
  (nonce,actor_subject,session_id,preview_id)
 VALUES(p_claims->>'nonce',p_claims->>'subject',
   (p_claims->>'sessionId')::uuid,(p_claims->>'previewId')::uuid);
 RETURN true;
END;
$attest$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.consume_verified_session_v5(
 jsonb,text,text) FROM PUBLIC;

CREATE FUNCTION orotitan_mobile_private.read_oauth_bound_preview_v5(
 p_claims jsonb,p_mac_sha256 text
) RETURNS jsonb LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $reader$
BEGIN
 PERFORM orotitan_mobile_private.consume_verified_session_v5(
   p_claims,p_mac_sha256,'read_preview');
 RETURN orotitan_mobile_private.read_preview_v3(
   (p_claims->>'previewId')::uuid,p_claims->>'subject',
   p_claims->>'clientId',p_claims->>'resource');
END;
$reader$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.read_oauth_bound_preview_v5(
 jsonb,text) FROM PUBLIC;

CREATE FUNCTION orotitan_mobile_private.issue_oauth_bound_scope_v5(
 p_claims jsonb,p_mac_sha256 text
) RETURNS boolean LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $scope$
BEGIN
 PERFORM orotitan_mobile_private.consume_verified_session_v5(
   p_claims,p_mac_sha256,'issue_research_scope');
 RETURN orotitan_mobile_private.issue_research_scope_v4(
   (p_claims->>'previewId')::uuid,p_claims->>'subject',
   p_claims->>'clientId',p_claims->>'resource',
   (p_claims->>'issuerId')::uuid,p_claims->>'contractSha256',
   (p_claims->>'cutoff')::date);
END;
$scope$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.issue_oauth_bound_scope_v5(
 jsonb,text) FROM PUBLIC;

GRANT CREATE ON SCHEMA orotitan_mobile_private
 TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.consume_verified_session_v5(
 jsonb,text,text) OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.read_oauth_bound_preview_v5(
 jsonb,text) OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.issue_oauth_bound_scope_v5(
 jsonb,text) OWNER TO orotitan_d32_private_executor;
REVOKE CREATE ON SCHEMA orotitan_mobile_private
 FROM orotitan_d32_private_executor;

-- Replace the old direct actor parameter APIs for their TEST role callers.
REVOKE EXECUTE ON FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) FROM orotitan_d34_preview_reader;
REVOKE EXECUTE ON FUNCTION orotitan_mobile_private.issue_research_scope_v4(
 uuid,text,text,text,uuid,text,date) FROM orotitan_d36_scope_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.read_oauth_bound_preview_v5(
 jsonb,text) TO orotitan_d37_session_gateway;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.issue_oauth_bound_scope_v5(
 jsonb,text) TO orotitan_d37_session_gateway;
COMMIT;
