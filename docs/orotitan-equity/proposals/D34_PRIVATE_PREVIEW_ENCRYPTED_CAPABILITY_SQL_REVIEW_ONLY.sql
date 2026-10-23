-- D34 / REVIEW ONLY / LOCAL POSTGRESQL 17 ONLY.
-- Extends the D30/D32 disposable fixture. Not a production migration.
-- Server-only preview issuer/reader and AES-256-GCM ciphertext vault:
-- plaintext capability, decryption key, and OAuth bearer never reach PostgreSQL.
BEGIN;
DO $guard$
BEGIN
 IF current_setting('server_version_num')::int NOT BETWEEN 170000 AND 179999
 OR to_regclass('orotitan_mobile_private.previews') IS NULL
 OR to_regclass('orotitan_mobile_private.command_requests') IS NULL
 OR NOT EXISTS(SELECT 1 FROM pg_roles
    WHERE rolname='orotitan_d32_private_executor')
 OR EXISTS(SELECT 1 FROM pg_roles WHERE rolname IN(
    'orotitan_d34_preview_issuer','orotitan_d34_preview_reader',
    'orotitan_d34_vault_issuer','orotitan_d34_vault_reader')) THEN
   RAISE EXCEPTION 'D34_REQUIRES_FRESH_DISPOSABLE_D32_POSTGRES_17';
 END IF;
END;
$guard$;

CREATE ROLE orotitan_d34_preview_issuer NOLOGIN NOINHERIT NOBYPASSRLS;
CREATE ROLE orotitan_d34_preview_reader NOLOGIN NOINHERIT NOBYPASSRLS;
CREATE ROLE orotitan_d34_vault_issuer NOLOGIN NOINHERIT NOBYPASSRLS;
CREATE ROLE orotitan_d34_vault_reader NOLOGIN NOINHERIT NOBYPASSRLS;

GRANT USAGE ON SCHEMA orotitan_mobile_private
 TO orotitan_d34_preview_issuer,orotitan_d34_preview_reader,
    orotitan_d34_vault_issuer,orotitan_d34_vault_reader;
GRANT INSERT ON orotitan_mobile_private.previews
 TO orotitan_d32_private_executor;

CREATE TABLE orotitan_mobile_private.encrypted_capability_vault (
 preview_id uuid PRIMARY KEY REFERENCES orotitan_mobile_private.previews(preview_id)
   ON DELETE RESTRICT,
 key_version text NOT NULL
   CHECK(key_version ~ '^[a-z0-9_-]{1,40}$'),
 nonce bytea NOT NULL CHECK(octet_length(nonce)=12),
 ciphertext bytea NOT NULL
   CHECK(octet_length(ciphertext) BETWEEN 50 AND 4000),
 auth_tag bytea NOT NULL CHECK(octet_length(auth_tag)=16),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
REVOKE ALL ON orotitan_mobile_private.encrypted_capability_vault FROM PUBLIC;
ALTER TABLE orotitan_mobile_private.encrypted_capability_vault ENABLE ROW LEVEL SECURITY;
GRANT INSERT,SELECT ON orotitan_mobile_private.encrypted_capability_vault
 TO orotitan_d32_private_executor;
CREATE POLICY d34_executor_vault_select ON
 orotitan_mobile_private.encrypted_capability_vault FOR SELECT
 TO orotitan_d32_private_executor USING(true);
CREATE POLICY d34_executor_vault_insert ON
 orotitan_mobile_private.encrypted_capability_vault FOR INSERT
 TO orotitan_d32_private_executor WITH CHECK(true);

-- Only the approved BACKEND preview issuer holds EXECUTE. Actor fields are
-- populated from previously verified JWT + independent server policy, not
-- user-supplied tool arguments. The function has no general INSERT/UPDATE.
CREATE FUNCTION orotitan_mobile_private.create_preview_v3(
 p_preview uuid,p_kind text,p_subject text,p_client text,p_resource text,
 p_payload jsonb,p_fingerprint text,p_expires timestamptz
) RETURNS boolean
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $preview$
DECLARE v_canonical text;v_now timestamptz:=clock_timestamp();
BEGIN
 IF p_preview IS NULL OR p_kind NOT IN('add_company','request_research')
 OR p_subject IS NULL OR length(p_subject) NOT BETWEEN 4 AND 160
 OR p_client IS NULL OR length(p_client) NOT BETWEEN 6 AND 160
 OR p_resource IS DISTINCT FROM
  'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
 OR p_payload IS NULL OR jsonb_typeof(p_payload)<>'object'
 OR (SELECT count(*) FROM jsonb_object_keys(p_payload)) NOT BETWEEN 1 AND 8
 OR EXISTS(SELECT 1 FROM jsonb_each(p_payload) v
    WHERE jsonb_typeof(v.value)<>'string')
 OR p_fingerprint IS NULL OR p_fingerprint !~ '^[a-f0-9]{64}$'
 OR p_expires IS NULL OR p_expires<=v_now
 OR p_expires>v_now+interval '2 minutes' THEN
  RAISE EXCEPTION 'D34_PREVIEW_INPUT_DENIED';
 END IF;
 -- D30's TypeScript canonical SHA256:
 -- JSON.stringify([kind, Object.keys(payload).sort().map(k=>[k,payload[k]])])
 -- p_payload is restricted to flat string fields; SQL never trusts a client hash.
 SELECT '['||to_json(p_kind)::text||',['||
   string_agg('['||to_json(k)::text||','||to_json(value)::text||']',',' ORDER BY k)
   ||']]' INTO v_canonical FROM jsonb_each_text(p_payload) AS x(k,value);
 IF v_canonical IS NULL OR
  encode(public.digest(convert_to(v_canonical,'UTF8'),'sha256'),'hex')
   IS DISTINCT FROM p_fingerprint THEN
  RAISE EXCEPTION 'D34_PREVIEW_FINGERPRINT_DENIED';
 END IF;

 -- Harden D34 so a database-authorized preview issuer cannot bypass the
 -- D30 exact identity shape or the I1 canonical issuer/security boundary.
 -- This does NOT replace independent OAuth or market-data provenance.
 IF (p_payload->>'name') IS NULL OR
  length(p_payload->>'name') NOT BETWEEN 2 AND 160 OR
  (p_payload->>'ticker') IS NULL OR
  (p_payload->>'ticker') !~ '^[A-Z0-9][A-Z0-9.-]{0,19}
  payload,fingerprint_sha256,expires_at
 ) VALUES(p_preview,p_kind,p_subject,p_client,p_resource,
  p_payload,p_fingerprint,p_expires);
 RETURN true;
END;
$preview$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz) FROM PUBLIC;

-- Read only this actor's ephemeral, unrevoked preview. The reader role has
-- no direct SELECT on the private tables.
CREATE FUNCTION orotitan_mobile_private.read_preview_v3(
 p_preview uuid,p_subject text,p_client text,p_resource text
) RETURNS jsonb
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $reader$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
BEGIN
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview AND actor_subject=p_subject
 AND actor_client=p_client AND actor_resource=p_resource
 AND revoked_at IS NULL AND confirmed_request_id IS NULL
 AND clock_timestamp()<expires_at;
 IF NOT FOUND THEN RETURN NULL; END IF;
 RETURN jsonb_build_object(
  'previewId',v.preview_id,'kind',v.kind,
  'actor',jsonb_build_object('subject',v.actor_subject,
   'clientId',v.actor_client,'resource',v.actor_resource),
  'payload',v.payload,'fingerprintSha256',v.fingerprint_sha256,
  'expiresAtMs',(extract(epoch from v.expires_at)*1000)::bigint);
END;
$reader$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) FROM PUBLIC;

-- Approver writes only ciphertext, nonce, tag and key identifier. PostgreSQL
-- stores ONLY a SHA-256 digest of the sealed HMAC token; the D31 token itself
-- remains encrypted using an application-held AES-256-GCM key.
-- Calling D32 approval and inserting the envelope are one transaction.
CREATE FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 p_preview uuid,p_kind text,p_subject text,p_client text,p_resource text,
 p_fingerprint text,p_digest text,p_key_version text,p_nonce bytea,
 p_ciphertext bytea,p_auth_tag bytea
) RETURNS boolean
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $approve$
DECLARE v_created boolean; v_existing orotitan_mobile_private.encrypted_capability_vault%ROWTYPE;
BEGIN
 IF p_key_version IS NULL OR p_key_version !~ '^[a-z0-9_-]{1,40}$'
 OR p_nonce IS NULL OR octet_length(p_nonce)<>12
 OR p_ciphertext IS NULL OR octet_length(p_ciphertext) NOT BETWEEN 50 AND 4000
 OR p_auth_tag IS NULL OR octet_length(p_auth_tag)<>16 THEN
  RAISE EXCEPTION 'D34_ENCRYPTED_ENVELOPE_INVALID';
 END IF;
 SELECT orotitan_mobile_private.approve_preview_v2(
  p_preview,p_kind,p_subject,p_client,p_resource,p_fingerprint,p_digest)
 INTO v_created;
 IF NOT v_created THEN
  SELECT * INTO v_existing FROM orotitan_mobile_private.encrypted_capability_vault
  WHERE preview_id=p_preview;
  IF NOT FOUND OR v_existing.key_version IS DISTINCT FROM p_key_version
  OR v_existing.nonce IS DISTINCT FROM p_nonce
  OR v_existing.ciphertext IS DISTINCT FROM p_ciphertext
  OR v_existing.auth_tag IS DISTINCT FROM p_auth_tag THEN
   RAISE EXCEPTION 'D34_ENCRYPTED_ENVELOPE_ALREADY_FIXED';
  END IF;
  RETURN false;
 END IF;
 INSERT INTO orotitan_mobile_private.encrypted_capability_vault(
  preview_id,key_version,nonce,ciphertext,auth_tag
 ) VALUES(p_preview,p_key_version,p_nonce,p_ciphertext,p_auth_tag);
 RETURN true;
END;
$approve$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea) FROM PUBLIC;

-- Server-private decryptor reads only ciphertext. No plaintext, HMAC key or
-- untrusted actor appears in this function. Exact preview actor is checked.
CREATE FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 p_preview uuid,p_subject text,p_client text,p_resource text
) RETURNS jsonb
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $vault$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
        e orotitan_mobile_private.encrypted_capability_vault%ROWTYPE;
BEGIN
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview AND actor_subject=p_subject
 AND actor_client=p_client AND actor_resource=p_resource
 AND approved_capability_sha256 IS NOT NULL AND revoked_at IS NULL
 AND confirmed_request_id IS NULL AND clock_timestamp()<expires_at;
 IF NOT FOUND THEN RETURN NULL; END IF;
 SELECT * INTO e FROM orotitan_mobile_private.encrypted_capability_vault
 WHERE preview_id=p_preview;
 IF NOT FOUND THEN RETURN NULL; END IF;
 RETURN jsonb_build_object(
  'previewId',v.preview_id,'kind',v.kind,'subject',v.actor_subject,
  'clientId',v.actor_client,'resource',v.actor_resource,
  'fingerprintSha256',v.fingerprint_sha256,
  'keyVersion',e.key_version,
  'nonce',replace(encode(e.nonce,'base64'),chr(10),''),
  'ciphertext',replace(encode(e.ciphertext,'base64'),chr(10),''),
  'authTag',replace(encode(e.auth_tag,'base64'),chr(10),''));
END;
$vault$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) FROM PUBLIC;

GRANT CREATE ON SCHEMA orotitan_mobile_private
 TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) OWNER TO orotitan_d32_private_executor;
REVOKE CREATE ON SCHEMA orotitan_mobile_private
 FROM orotitan_d32_private_executor;

GRANT EXECUTE ON FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz)
 TO orotitan_d34_preview_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) TO orotitan_d34_preview_reader;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)
 TO orotitan_d34_vault_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) TO orotitan_d34_vault_reader;
COMMIT;
 OR
  (p_payload->>'exchange') IS NULL OR
  (p_payload->>'exchange') !~ '^[A-Z0-9][A-Z0-9_-]{1,39}
  payload,fingerprint_sha256,expires_at
 ) VALUES(p_preview,p_kind,p_subject,p_client,p_resource,
  p_payload,p_fingerprint,p_expires);
 RETURN true;
END;
$preview$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz) FROM PUBLIC;

-- Read only this actor's ephemeral, unrevoked preview. The reader role has
-- no direct SELECT on the private tables.
CREATE FUNCTION orotitan_mobile_private.read_preview_v3(
 p_preview uuid,p_subject text,p_client text,p_resource text
) RETURNS jsonb
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $reader$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
BEGIN
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview AND actor_subject=p_subject
 AND actor_client=p_client AND actor_resource=p_resource
 AND revoked_at IS NULL AND clock_timestamp()<expires_at;
 IF NOT FOUND THEN RETURN NULL; END IF;
 RETURN jsonb_build_object(
  'previewId',v.preview_id,'kind',v.kind,
  'actor',jsonb_build_object('subject',v.actor_subject,
   'clientId',v.actor_client,'resource',v.actor_resource),
  'payload',v.payload,'fingerprintSha256',v.fingerprint_sha256,
  'expiresAtMs',(extract(epoch from v.expires_at)*1000)::bigint);
END;
$reader$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) FROM PUBLIC;

-- Approver writes only ciphertext, nonce, tag and key identifier. PostgreSQL
-- stores ONLY a SHA-256 digest of the sealed HMAC token; the D31 token itself
-- remains encrypted using an application-held AES-256-GCM key.
-- Calling D32 approval and inserting the envelope are one transaction.
CREATE FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 p_preview uuid,p_kind text,p_subject text,p_client text,p_resource text,
 p_fingerprint text,p_digest text,p_key_version text,p_nonce bytea,
 p_ciphertext bytea,p_auth_tag bytea
) RETURNS boolean
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $approve$
DECLARE v_created boolean; v_existing orotitan_mobile_private.encrypted_capability_vault%ROWTYPE;
BEGIN
 IF p_key_version IS NULL OR p_key_version !~ '^[a-z0-9_-]{1,40}$'
 OR p_nonce IS NULL OR octet_length(p_nonce)<>12
 OR p_ciphertext IS NULL OR octet_length(p_ciphertext) NOT BETWEEN 50 AND 4000
 OR p_auth_tag IS NULL OR octet_length(p_auth_tag)<>16 THEN
  RAISE EXCEPTION 'D34_ENCRYPTED_ENVELOPE_INVALID';
 END IF;
 SELECT orotitan_mobile_private.approve_preview_v2(
  p_preview,p_kind,p_subject,p_client,p_resource,p_fingerprint,p_digest)
 INTO v_created;
 IF NOT v_created THEN
  SELECT * INTO v_existing FROM orotitan_mobile_private.encrypted_capability_vault
  WHERE preview_id=p_preview;
  IF NOT FOUND OR v_existing.key_version IS DISTINCT FROM p_key_version
  OR v_existing.nonce IS DISTINCT FROM p_nonce
  OR v_existing.ciphertext IS DISTINCT FROM p_ciphertext
  OR v_existing.auth_tag IS DISTINCT FROM p_auth_tag THEN
   RAISE EXCEPTION 'D34_ENCRYPTED_ENVELOPE_ALREADY_FIXED';
  END IF;
  RETURN false;
 END IF;
 INSERT INTO orotitan_mobile_private.encrypted_capability_vault(
  preview_id,key_version,nonce,ciphertext,auth_tag
 ) VALUES(p_preview,p_key_version,p_nonce,p_ciphertext,p_auth_tag);
 RETURN true;
END;
$approve$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea) FROM PUBLIC;

-- Server-private decryptor reads only ciphertext. No plaintext, HMAC key or
-- untrusted actor appears in this function. Exact preview actor is checked.
CREATE FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 p_preview uuid,p_subject text,p_client text,p_resource text
) RETURNS jsonb
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $vault$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
        e orotitan_mobile_private.encrypted_capability_vault%ROWTYPE;
BEGIN
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview AND actor_subject=p_subject
 AND actor_client=p_client AND actor_resource=p_resource
 AND approved_capability_sha256 IS NOT NULL AND revoked_at IS NULL
 AND confirmed_request_id IS NULL AND clock_timestamp()<expires_at;
 IF NOT FOUND THEN RETURN NULL; END IF;
 SELECT * INTO e FROM orotitan_mobile_private.encrypted_capability_vault
 WHERE preview_id=p_preview;
 IF NOT FOUND THEN RETURN NULL; END IF;
 RETURN jsonb_build_object(
  'previewId',v.preview_id,'kind',v.kind,'subject',v.actor_subject,
  'clientId',v.actor_client,'resource',v.actor_resource,
  'fingerprintSha256',v.fingerprint_sha256,
  'keyVersion',e.key_version,
  'nonce',replace(encode(e.nonce,'base64'),chr(10),''),
  'ciphertext',replace(encode(e.ciphertext,'base64'),chr(10),''),
  'authTag',replace(encode(e.auth_tag,'base64'),chr(10),''));
END;
$vault$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) FROM PUBLIC;

GRANT CREATE ON SCHEMA orotitan_mobile_private
 TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) OWNER TO orotitan_d32_private_executor;
REVOKE CREATE ON SCHEMA orotitan_mobile_private
 FROM orotitan_d32_private_executor;

GRANT EXECUTE ON FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz)
 TO orotitan_d34_preview_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) TO orotitan_d34_preview_reader;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)
 TO orotitan_d34_vault_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) TO orotitan_d34_vault_reader;
COMMIT;
 OR
  (p_payload->>'isin') IS NULL OR
  (p_payload->>'isin') !~ '^[A-Z]{2}[A-Z0-9]{9}[0-9]
  payload,fingerprint_sha256,expires_at
 ) VALUES(p_preview,p_kind,p_subject,p_client,p_resource,
  p_payload,p_fingerprint,p_expires);
 RETURN true;
END;
$preview$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz) FROM PUBLIC;

-- Read only this actor's ephemeral, unrevoked preview. The reader role has
-- no direct SELECT on the private tables.
CREATE FUNCTION orotitan_mobile_private.read_preview_v3(
 p_preview uuid,p_subject text,p_client text,p_resource text
) RETURNS jsonb
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $reader$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
BEGIN
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview AND actor_subject=p_subject
 AND actor_client=p_client AND actor_resource=p_resource
 AND revoked_at IS NULL AND clock_timestamp()<expires_at;
 IF NOT FOUND THEN RETURN NULL; END IF;
 RETURN jsonb_build_object(
  'previewId',v.preview_id,'kind',v.kind,
  'actor',jsonb_build_object('subject',v.actor_subject,
   'clientId',v.actor_client,'resource',v.actor_resource),
  'payload',v.payload,'fingerprintSha256',v.fingerprint_sha256,
  'expiresAtMs',(extract(epoch from v.expires_at)*1000)::bigint);
END;
$reader$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) FROM PUBLIC;

-- Approver writes only ciphertext, nonce, tag and key identifier. PostgreSQL
-- stores ONLY a SHA-256 digest of the sealed HMAC token; the D31 token itself
-- remains encrypted using an application-held AES-256-GCM key.
-- Calling D32 approval and inserting the envelope are one transaction.
CREATE FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 p_preview uuid,p_kind text,p_subject text,p_client text,p_resource text,
 p_fingerprint text,p_digest text,p_key_version text,p_nonce bytea,
 p_ciphertext bytea,p_auth_tag bytea
) RETURNS boolean
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $approve$
DECLARE v_created boolean; v_existing orotitan_mobile_private.encrypted_capability_vault%ROWTYPE;
BEGIN
 IF p_key_version IS NULL OR p_key_version !~ '^[a-z0-9_-]{1,40}$'
 OR p_nonce IS NULL OR octet_length(p_nonce)<>12
 OR p_ciphertext IS NULL OR octet_length(p_ciphertext) NOT BETWEEN 50 AND 4000
 OR p_auth_tag IS NULL OR octet_length(p_auth_tag)<>16 THEN
  RAISE EXCEPTION 'D34_ENCRYPTED_ENVELOPE_INVALID';
 END IF;
 SELECT orotitan_mobile_private.approve_preview_v2(
  p_preview,p_kind,p_subject,p_client,p_resource,p_fingerprint,p_digest)
 INTO v_created;
 IF NOT v_created THEN
  SELECT * INTO v_existing FROM orotitan_mobile_private.encrypted_capability_vault
  WHERE preview_id=p_preview;
  IF NOT FOUND OR v_existing.key_version IS DISTINCT FROM p_key_version
  OR v_existing.nonce IS DISTINCT FROM p_nonce
  OR v_existing.ciphertext IS DISTINCT FROM p_ciphertext
  OR v_existing.auth_tag IS DISTINCT FROM p_auth_tag THEN
   RAISE EXCEPTION 'D34_ENCRYPTED_ENVELOPE_ALREADY_FIXED';
  END IF;
  RETURN false;
 END IF;
 INSERT INTO orotitan_mobile_private.encrypted_capability_vault(
  preview_id,key_version,nonce,ciphertext,auth_tag
 ) VALUES(p_preview,p_key_version,p_nonce,p_ciphertext,p_auth_tag);
 RETURN true;
END;
$approve$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea) FROM PUBLIC;

-- Server-private decryptor reads only ciphertext. No plaintext, HMAC key or
-- untrusted actor appears in this function. Exact preview actor is checked.
CREATE FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 p_preview uuid,p_subject text,p_client text,p_resource text
) RETURNS jsonb
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $vault$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
        e orotitan_mobile_private.encrypted_capability_vault%ROWTYPE;
BEGIN
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview AND actor_subject=p_subject
 AND actor_client=p_client AND actor_resource=p_resource
 AND approved_capability_sha256 IS NOT NULL AND revoked_at IS NULL
 AND confirmed_request_id IS NULL AND clock_timestamp()<expires_at;
 IF NOT FOUND THEN RETURN NULL; END IF;
 SELECT * INTO e FROM orotitan_mobile_private.encrypted_capability_vault
 WHERE preview_id=p_preview;
 IF NOT FOUND THEN RETURN NULL; END IF;
 RETURN jsonb_build_object(
  'previewId',v.preview_id,'kind',v.kind,'subject',v.actor_subject,
  'clientId',v.actor_client,'resource',v.actor_resource,
  'fingerprintSha256',v.fingerprint_sha256,
  'keyVersion',e.key_version,
  'nonce',replace(encode(e.nonce,'base64'),chr(10),''),
  'ciphertext',replace(encode(e.ciphertext,'base64'),chr(10),''),
  'authTag',replace(encode(e.auth_tag,'base64'),chr(10),''));
END;
$vault$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) FROM PUBLIC;

GRANT CREATE ON SCHEMA orotitan_mobile_private
 TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) OWNER TO orotitan_d32_private_executor;
REVOKE CREATE ON SCHEMA orotitan_mobile_private
 FROM orotitan_d32_private_executor;

GRANT EXECUTE ON FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz)
 TO orotitan_d34_preview_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) TO orotitan_d34_preview_reader;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)
 TO orotitan_d34_vault_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) TO orotitan_d34_vault_reader;
COMMIT;
 THEN
   RAISE EXCEPTION 'D35_PREVIEW_IDENTITY_FIELDS_DENIED';
 END IF;
 IF p_kind='add_company' THEN
   IF (SELECT count(*) FROM jsonb_object_keys(p_payload))<>4 OR
     NOT (p_payload ?& array['name','ticker','exchange','isin']) OR
     EXISTS(SELECT 1 FROM public.securities s
      WHERE (upper(s.exchange)=p_payload->>'exchange'
       AND upper(s.ticker)=p_payload->>'ticker')
       OR s.isin=p_payload->>'isin') OR
     EXISTS(SELECT 1 FROM public.companies c
      WHERE upper(c.exchange)=p_payload->>'exchange'
       AND upper(c.ticker)=p_payload->>'ticker') THEN
      RAISE EXCEPTION 'D35_COMPANY_ALREADY_TRACKED_OR_FIELDS_DENIED';
   END IF;
 ELSE
   IF (SELECT count(*) FROM jsonb_object_keys(p_payload))<>5 OR
     NOT (p_payload ?& array['issuerId','name','ticker','exchange','isin']) OR
     (p_payload->>'issuerId') IS NULL OR
     (p_payload->>'issuerId') !~
      '^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}
  payload,fingerprint_sha256,expires_at
 ) VALUES(p_preview,p_kind,p_subject,p_client,p_resource,
  p_payload,p_fingerprint,p_expires);
 RETURN true;
END;
$preview$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz) FROM PUBLIC;

-- Read only this actor's ephemeral, unrevoked preview. The reader role has
-- no direct SELECT on the private tables.
CREATE FUNCTION orotitan_mobile_private.read_preview_v3(
 p_preview uuid,p_subject text,p_client text,p_resource text
) RETURNS jsonb
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $reader$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
BEGIN
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview AND actor_subject=p_subject
 AND actor_client=p_client AND actor_resource=p_resource
 AND revoked_at IS NULL AND clock_timestamp()<expires_at;
 IF NOT FOUND THEN RETURN NULL; END IF;
 RETURN jsonb_build_object(
  'previewId',v.preview_id,'kind',v.kind,
  'actor',jsonb_build_object('subject',v.actor_subject,
   'clientId',v.actor_client,'resource',v.actor_resource),
  'payload',v.payload,'fingerprintSha256',v.fingerprint_sha256,
  'expiresAtMs',(extract(epoch from v.expires_at)*1000)::bigint);
END;
$reader$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) FROM PUBLIC;

-- Approver writes only ciphertext, nonce, tag and key identifier. PostgreSQL
-- stores ONLY a SHA-256 digest of the sealed HMAC token; the D31 token itself
-- remains encrypted using an application-held AES-256-GCM key.
-- Calling D32 approval and inserting the envelope are one transaction.
CREATE FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 p_preview uuid,p_kind text,p_subject text,p_client text,p_resource text,
 p_fingerprint text,p_digest text,p_key_version text,p_nonce bytea,
 p_ciphertext bytea,p_auth_tag bytea
) RETURNS boolean
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $approve$
DECLARE v_created boolean; v_existing orotitan_mobile_private.encrypted_capability_vault%ROWTYPE;
BEGIN
 IF p_key_version IS NULL OR p_key_version !~ '^[a-z0-9_-]{1,40}$'
 OR p_nonce IS NULL OR octet_length(p_nonce)<>12
 OR p_ciphertext IS NULL OR octet_length(p_ciphertext) NOT BETWEEN 50 AND 4000
 OR p_auth_tag IS NULL OR octet_length(p_auth_tag)<>16 THEN
  RAISE EXCEPTION 'D34_ENCRYPTED_ENVELOPE_INVALID';
 END IF;
 SELECT orotitan_mobile_private.approve_preview_v2(
  p_preview,p_kind,p_subject,p_client,p_resource,p_fingerprint,p_digest)
 INTO v_created;
 IF NOT v_created THEN
  SELECT * INTO v_existing FROM orotitan_mobile_private.encrypted_capability_vault
  WHERE preview_id=p_preview;
  IF NOT FOUND OR v_existing.key_version IS DISTINCT FROM p_key_version
  OR v_existing.nonce IS DISTINCT FROM p_nonce
  OR v_existing.ciphertext IS DISTINCT FROM p_ciphertext
  OR v_existing.auth_tag IS DISTINCT FROM p_auth_tag THEN
   RAISE EXCEPTION 'D34_ENCRYPTED_ENVELOPE_ALREADY_FIXED';
  END IF;
  RETURN false;
 END IF;
 INSERT INTO orotitan_mobile_private.encrypted_capability_vault(
  preview_id,key_version,nonce,ciphertext,auth_tag
 ) VALUES(p_preview,p_key_version,p_nonce,p_ciphertext,p_auth_tag);
 RETURN true;
END;
$approve$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea) FROM PUBLIC;

-- Server-private decryptor reads only ciphertext. No plaintext, HMAC key or
-- untrusted actor appears in this function. Exact preview actor is checked.
CREATE FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 p_preview uuid,p_subject text,p_client text,p_resource text
) RETURNS jsonb
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $vault$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
        e orotitan_mobile_private.encrypted_capability_vault%ROWTYPE;
BEGIN
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview AND actor_subject=p_subject
 AND actor_client=p_client AND actor_resource=p_resource
 AND approved_capability_sha256 IS NOT NULL AND revoked_at IS NULL
 AND confirmed_request_id IS NULL AND clock_timestamp()<expires_at;
 IF NOT FOUND THEN RETURN NULL; END IF;
 SELECT * INTO e FROM orotitan_mobile_private.encrypted_capability_vault
 WHERE preview_id=p_preview;
 IF NOT FOUND THEN RETURN NULL; END IF;
 RETURN jsonb_build_object(
  'previewId',v.preview_id,'kind',v.kind,'subject',v.actor_subject,
  'clientId',v.actor_client,'resource',v.actor_resource,
  'fingerprintSha256',v.fingerprint_sha256,
  'keyVersion',e.key_version,
  'nonce',replace(encode(e.nonce,'base64'),chr(10),''),
  'ciphertext',replace(encode(e.ciphertext,'base64'),chr(10),''),
  'authTag',replace(encode(e.auth_tag,'base64'),chr(10),''));
END;
$vault$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) FROM PUBLIC;

GRANT CREATE ON SCHEMA orotitan_mobile_private
 TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) OWNER TO orotitan_d32_private_executor;
REVOKE CREATE ON SCHEMA orotitan_mobile_private
 FROM orotitan_d32_private_executor;

GRANT EXECUTE ON FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz)
 TO orotitan_d34_preview_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) TO orotitan_d34_preview_reader;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)
 TO orotitan_d34_vault_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) TO orotitan_d34_vault_reader;
COMMIT;

   THEN
      RAISE EXCEPTION 'D35_RESEARCH_IDENTITY_FIELDS_DENIED';
   END IF;
   IF NOT EXISTS(SELECT 1 FROM public.issuers i
      JOIN public.securities s ON s.issuer_id=i.issuer_id
      WHERE i.issuer_id=(p_payload->>'issuerId')::uuid
       AND i.display_name=p_payload->>'name'
       AND s.ticker=p_payload->>'ticker'
       AND s.exchange=p_payload->>'exchange'
       AND s.isin=p_payload->>'isin' AND s.isin<>'UNKNOWN') THEN
      RAISE EXCEPTION 'D35_RESEARCH_CANONICAL_CROSSWALK_DENIED';
   END IF;
 END IF;

 INSERT INTO orotitan_mobile_private.previews(
  preview_id,kind,actor_subject,actor_client,actor_resource,
  payload,fingerprint_sha256,expires_at
 ) VALUES(p_preview,p_kind,p_subject,p_client,p_resource,
  p_payload,p_fingerprint,p_expires);
 RETURN true;
END;
$preview$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz) FROM PUBLIC;

-- Read only this actor's ephemeral, unrevoked preview. The reader role has
-- no direct SELECT on the private tables.
CREATE FUNCTION orotitan_mobile_private.read_preview_v3(
 p_preview uuid,p_subject text,p_client text,p_resource text
) RETURNS jsonb
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $reader$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
BEGIN
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview AND actor_subject=p_subject
 AND actor_client=p_client AND actor_resource=p_resource
 AND revoked_at IS NULL AND clock_timestamp()<expires_at;
 IF NOT FOUND THEN RETURN NULL; END IF;
 RETURN jsonb_build_object(
  'previewId',v.preview_id,'kind',v.kind,
  'actor',jsonb_build_object('subject',v.actor_subject,
   'clientId',v.actor_client,'resource',v.actor_resource),
  'payload',v.payload,'fingerprintSha256',v.fingerprint_sha256,
  'expiresAtMs',(extract(epoch from v.expires_at)*1000)::bigint);
END;
$reader$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) FROM PUBLIC;

-- Approver writes only ciphertext, nonce, tag and key identifier. PostgreSQL
-- stores ONLY a SHA-256 digest of the sealed HMAC token; the D31 token itself
-- remains encrypted using an application-held AES-256-GCM key.
-- Calling D32 approval and inserting the envelope are one transaction.
CREATE FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 p_preview uuid,p_kind text,p_subject text,p_client text,p_resource text,
 p_fingerprint text,p_digest text,p_key_version text,p_nonce bytea,
 p_ciphertext bytea,p_auth_tag bytea
) RETURNS boolean
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $approve$
DECLARE v_created boolean; v_existing orotitan_mobile_private.encrypted_capability_vault%ROWTYPE;
BEGIN
 IF p_key_version IS NULL OR p_key_version !~ '^[a-z0-9_-]{1,40}$'
 OR p_nonce IS NULL OR octet_length(p_nonce)<>12
 OR p_ciphertext IS NULL OR octet_length(p_ciphertext) NOT BETWEEN 50 AND 4000
 OR p_auth_tag IS NULL OR octet_length(p_auth_tag)<>16 THEN
  RAISE EXCEPTION 'D34_ENCRYPTED_ENVELOPE_INVALID';
 END IF;
 SELECT orotitan_mobile_private.approve_preview_v2(
  p_preview,p_kind,p_subject,p_client,p_resource,p_fingerprint,p_digest)
 INTO v_created;
 IF NOT v_created THEN
  SELECT * INTO v_existing FROM orotitan_mobile_private.encrypted_capability_vault
  WHERE preview_id=p_preview;
  IF NOT FOUND OR v_existing.key_version IS DISTINCT FROM p_key_version
  OR v_existing.nonce IS DISTINCT FROM p_nonce
  OR v_existing.ciphertext IS DISTINCT FROM p_ciphertext
  OR v_existing.auth_tag IS DISTINCT FROM p_auth_tag THEN
   RAISE EXCEPTION 'D34_ENCRYPTED_ENVELOPE_ALREADY_FIXED';
  END IF;
  RETURN false;
 END IF;
 INSERT INTO orotitan_mobile_private.encrypted_capability_vault(
  preview_id,key_version,nonce,ciphertext,auth_tag
 ) VALUES(p_preview,p_key_version,p_nonce,p_ciphertext,p_auth_tag);
 RETURN true;
END;
$approve$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea) FROM PUBLIC;

-- Server-private decryptor reads only ciphertext. No plaintext, HMAC key or
-- untrusted actor appears in this function. Exact preview actor is checked.
CREATE FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 p_preview uuid,p_subject text,p_client text,p_resource text
) RETURNS jsonb
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,orotitan_mobile_private AS $vault$
DECLARE v orotitan_mobile_private.previews%ROWTYPE;
        e orotitan_mobile_private.encrypted_capability_vault%ROWTYPE;
BEGIN
 SELECT * INTO v FROM orotitan_mobile_private.previews
 WHERE preview_id=p_preview AND actor_subject=p_subject
 AND actor_client=p_client AND actor_resource=p_resource
 AND approved_capability_sha256 IS NOT NULL AND revoked_at IS NULL
 AND confirmed_request_id IS NULL AND clock_timestamp()<expires_at;
 IF NOT FOUND THEN RETURN NULL; END IF;
 SELECT * INTO e FROM orotitan_mobile_private.encrypted_capability_vault
 WHERE preview_id=p_preview;
 IF NOT FOUND THEN RETURN NULL; END IF;
 RETURN jsonb_build_object(
  'previewId',v.preview_id,'kind',v.kind,'subject',v.actor_subject,
  'clientId',v.actor_client,'resource',v.actor_resource,
  'fingerprintSha256',v.fingerprint_sha256,
  'keyVersion',e.key_version,
  'nonce',replace(encode(e.nonce,'base64'),chr(10),''),
  'ciphertext',replace(encode(e.ciphertext,'base64'),chr(10),''),
  'authTag',replace(encode(e.auth_tag,'base64'),chr(10),''));
END;
$vault$;
REVOKE ALL ON FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) FROM PUBLIC;

GRANT CREATE ON SCHEMA orotitan_mobile_private
 TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)
 OWNER TO orotitan_d32_private_executor;
ALTER FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) OWNER TO orotitan_d32_private_executor;
REVOKE CREATE ON SCHEMA orotitan_mobile_private
 FROM orotitan_d32_private_executor;

GRANT EXECUTE ON FUNCTION orotitan_mobile_private.create_preview_v3(
 uuid,text,text,text,text,jsonb,text,timestamptz)
 TO orotitan_d34_preview_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.read_preview_v3(
 uuid,text,text,text) TO orotitan_d34_preview_reader;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.approve_and_store_capability_v3(
 uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)
 TO orotitan_d34_vault_issuer;
GRANT EXECUTE ON FUNCTION orotitan_mobile_private.read_encrypted_capability_v3(
 uuid,text,text,text) TO orotitan_d34_vault_reader;
COMMIT;
