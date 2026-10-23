-- D29b RESEARCH PRIVATE SQL GATES — REVIEW ONLY / NEVER APPLY IN PRODUCTION.
-- Only a disposable isolated PostgreSQL 17 fixture may execute this file.
-- No Supabase migration, storage bucket, OAuth write scope or Registry mutation.
-- Public repository: all identifiers here are generic except the known Besi run UUID.
BEGIN;
DO $guard$
BEGIN
  IF current_setting('server_version_num')::int < 170000
     OR current_setting('server_version_num')::int >= 180000
     OR to_regclass('public.orotitan_runs') IS NULL
     OR to_regclass('public.orotitan_run_stages') IS NULL THEN
     RAISE EXCEPTION 'D29b private SQL: PostgreSQL 17 / Registry prerequisite absent';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname IN
    ('orotitan_research_intake_writer','orotitan_research_intake_reader')) THEN
    RAISE EXCEPTION 'D29b private SQL: role exists; manual review required';
  END IF;
END;
$guard$;

CREATE ROLE orotitan_research_intake_writer NOLOGIN NOINHERIT NOBYPASSRLS;
CREATE ROLE orotitan_research_intake_reader NOLOGIN NOINHERIT NOBYPASSRLS;
CREATE SCHEMA orotitan_research_intake_private;
REVOKE ALL ON SCHEMA orotitan_research_intake_private FROM PUBLIC;
GRANT USAGE ON SCHEMA orotitan_research_intake_private
  TO orotitan_research_intake_writer,orotitan_research_intake_reader;
-- Intentionally no GRANT ROLE to any runtime principal, anon or authenticated.
-- Independent identity approval and membership provisioning remain UNIMPLEMENTED.

-- These are new policies, limited to a single run. They do not grant UPDATE.
GRANT SELECT (run_id,issuer_id,run_status,current_stage,state_version,data_cutoff,
  contract_set_sha256) ON public.orotitan_runs
  TO orotitan_research_intake_writer,orotitan_research_intake_reader;
GRANT SELECT (run_id,stage_code,state_version,lifecycle_status)
  ON public.orotitan_run_stages
  TO orotitan_research_intake_writer,orotitan_research_intake_reader;
CREATE POLICY d29b_research_besi_run_read
  ON public.orotitan_runs FOR SELECT
  TO orotitan_research_intake_writer,orotitan_research_intake_reader
  USING (run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid);
CREATE POLICY d29b_research_besi_stage_read
  ON public.orotitan_run_stages FOR SELECT
  TO orotitan_research_intake_writer,orotitan_research_intake_reader
  USING (run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid
    AND stage_code='RESEARCH');

CREATE TABLE orotitan_research_intake_private.submission_grants(
  grant_id text PRIMARY KEY CHECK(length(grant_id) BETWEEN 8 AND 160),
  run_id uuid NOT NULL CHECK(run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid),
  issuer_id text NOT NULL,
  subject_id text NOT NULL CHECK(length(subject_id) BETWEEN 1 AND 256),
  client_id text NOT NULL,
  mcp_resource text NOT NULL,
  contract_pins_sha256 text NOT NULL CHECK(contract_pins_sha256 ~ '^[0-9a-f]{64}$'),
  expected_run_version bigint NOT NULL CHECK(expected_run_version>=1),
  expected_stage_version bigint NOT NULL CHECK(expected_stage_version>=1),
  data_cutoff date NOT NULL,
  issued_at timestamptz NOT NULL,
  expires_at timestamptz NOT NULL,
  revoked_at timestamptz,
  permission text NOT NULL DEFAULT 'SUBMIT_DRAFT'
    CHECK(permission='SUBMIT_DRAFT'),
  CHECK(expires_at>issued_at AND expires_at<=issued_at+interval '5 minutes')
);
CREATE TABLE orotitan_research_intake_private.verifier_grants(
  verifier_grant_id text PRIMARY KEY CHECK(length(verifier_grant_id) BETWEEN 8 AND 160),
  run_id uuid NOT NULL CHECK(run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid),
  issuer_id text NOT NULL,
  reviewer_id text NOT NULL CHECK(length(reviewer_id)>0),
  expected_run_version bigint NOT NULL CHECK(expected_run_version>=1),
  expected_stage_version bigint NOT NULL CHECK(expected_stage_version>=1),
  issued_at timestamptz NOT NULL,
  expires_at timestamptz NOT NULL,
  revoked_at timestamptz,
  permission text NOT NULL DEFAULT 'READ_RECEIPT_METADATA'
    CHECK(permission='READ_RECEIPT_METADATA'),
  CHECK(expires_at>issued_at AND expires_at<=issued_at+interval '5 minutes')
);
CREATE TABLE orotitan_research_intake_private.receipts(
  receipt_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid NOT NULL CHECK(run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid),
  issuer_id text NOT NULL,
  stage text NOT NULL DEFAULT 'RESEARCH' CHECK(stage='RESEARCH'),
  run_state_version bigint NOT NULL,
  stage_state_version bigint NOT NULL,
  contract_pins_sha256 text NOT NULL CHECK(contract_pins_sha256 ~ '^[0-9a-f]{64}$'),
  grant_id text NOT NULL REFERENCES orotitan_research_intake_private.submission_grants(grant_id),
  sha256 text NOT NULL CHECK(sha256 ~ '^[0-9a-f]{64}$'),
  file_name text NOT NULL CHECK(file_name ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$'),
  media_type text NOT NULL,
  object_path text NOT NULL UNIQUE,
  size_bytes bigint NOT NULL CHECK(size_bytes BETWEEN 1 AND 1000000),
  source_date date NOT NULL,
  submitted_by_subject text NOT NULL,
  submitted_by_client text NOT NULL,
  receipt_status text NOT NULL DEFAULT 'RECEIVED_NOT_VERIFIED'
    CHECK(receipt_status='RECEIVED_NOT_VERIFIED'),
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  UNIQUE(run_id,stage,run_state_version,stage_state_version,sha256)
);
CREATE TABLE orotitan_research_intake_private.receipt_audit(
  event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  receipt_id uuid NOT NULL REFERENCES orotitan_research_intake_private.receipts(receipt_id),
  event_type text NOT NULL CHECK(event_type='DRAFT_RECEIVED'),
  run_id uuid NOT NULL,
  sha256 text NOT NULL CHECK(sha256 ~ '^[0-9a-f]{64}$'),
  actor_subject text NOT NULL,
  actor_client text NOT NULL,
  event_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

-- Private tables are invisible to anon/authenticated and use defense-in-depth RLS.
REVOKE ALL ON ALL TABLES IN SCHEMA orotitan_research_intake_private FROM PUBLIC;
ALTER TABLE orotitan_research_intake_private.submission_grants ENABLE ROW LEVEL SECURITY;
ALTER TABLE orotitan_research_intake_private.verifier_grants ENABLE ROW LEVEL SECURITY;
ALTER TABLE orotitan_research_intake_private.receipts ENABLE ROW LEVEL SECURITY;
ALTER TABLE orotitan_research_intake_private.receipt_audit ENABLE ROW LEVEL SECURITY;
GRANT SELECT ON orotitan_research_intake_private.submission_grants
  TO orotitan_research_intake_writer;
GRANT SELECT ON orotitan_research_intake_private.verifier_grants
  TO orotitan_research_intake_reader;
GRANT SELECT,INSERT ON orotitan_research_intake_private.receipts
  TO orotitan_research_intake_writer;
GRANT SELECT ON orotitan_research_intake_private.receipts
  TO orotitan_research_intake_reader;
GRANT INSERT ON orotitan_research_intake_private.receipt_audit
  TO orotitan_research_intake_writer;
CREATE POLICY d29b_research_writer_grants ON
  orotitan_research_intake_private.submission_grants FOR SELECT
  TO orotitan_research_intake_writer
  USING(run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid);
CREATE POLICY d29b_research_reader_grants ON
  orotitan_research_intake_private.verifier_grants FOR SELECT
  TO orotitan_research_intake_reader
  USING(run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid);
CREATE POLICY d29b_research_writer_receipts_select ON
  orotitan_research_intake_private.receipts FOR SELECT
  TO orotitan_research_intake_writer
  USING(run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid);
CREATE POLICY d29b_research_reader_receipts_select ON
  orotitan_research_intake_private.receipts FOR SELECT
  TO orotitan_research_intake_reader
  USING(run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid);
CREATE POLICY d29b_research_writer_receipts_insert ON
  orotitan_research_intake_private.receipts FOR INSERT
  TO orotitan_research_intake_writer
  WITH CHECK(run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid
    AND stage='RESEARCH' AND receipt_status='RECEIVED_NOT_VERIFIED');
CREATE POLICY d29b_research_writer_audit_insert ON
  orotitan_research_intake_private.receipt_audit FOR INSERT
  TO orotitan_research_intake_writer
  WITH CHECK(run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid
    AND event_type='DRAFT_RECEIVED');

-- Audit trigger runs as INVOKER; failures abort the receipt transaction.
CREATE FUNCTION orotitan_research_intake_private.receipt_audit_v1()
RETURNS trigger LANGUAGE plpgsql SECURITY INVOKER
SET search_path=pg_catalog,orotitan_research_intake_private
AS $audit$
BEGIN
  INSERT INTO orotitan_research_intake_private.receipt_audit
    (receipt_id,event_type,run_id,sha256,actor_subject,actor_client)
  VALUES(NEW.receipt_id,'DRAFT_RECEIVED',NEW.run_id,NEW.sha256,
    NEW.submitted_by_subject,NEW.submitted_by_client);
  RETURN NULL;
END;
$audit$;
CREATE TRIGGER d29b_research_receipt_append_audit
  AFTER INSERT ON orotitan_research_intake_private.receipts
  FOR EACH ROW EXECUTE FUNCTION
    orotitan_research_intake_private.receipt_audit_v1();

CREATE FUNCTION orotitan_research_intake_private.preflight_v1(
  p_run uuid,p_issuer text,p_grant text,p_subject text,p_client text,
  p_resource text,p_run_ver bigint,p_stage_ver bigint,p_cutoff date,p_pins text
) RETURNS boolean LANGUAGE plpgsql STABLE SECURITY INVOKER
SET search_path=pg_catalog,public,orotitan_research_intake_private
AS $preflight$
BEGIN
  IF current_user<>'orotitan_research_intake_writer'
     OR p_run IS DISTINCT FROM
       '48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid
     OR p_resource IS DISTINCT FROM
       'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
     OR p_issuer IS NULL OR p_subject IS NULL OR p_client IS NULL
     OR p_pins !~ '^[0-9a-f]{64}$' THEN
    RETURN false;
  END IF;
  RETURN EXISTS(
    SELECT 1 FROM orotitan_research_intake_private.submission_grants g
    JOIN public.orotitan_runs r ON r.run_id=g.run_id
    JOIN public.orotitan_run_stages s ON s.run_id=r.run_id
      AND s.stage_code='RESEARCH'
    WHERE g.grant_id=p_grant AND g.run_id=p_run
      AND g.issuer_id=p_issuer AND g.issuer_id=r.issuer_id::text
      AND g.subject_id=p_subject AND g.client_id=p_client
      AND g.mcp_resource=p_resource
      AND g.permission='SUBMIT_DRAFT'
      AND g.revoked_at IS NULL AND now()>=g.issued_at AND now()<g.expires_at
      AND g.expected_run_version=p_run_ver
      AND g.expected_stage_version=p_stage_ver
      AND g.data_cutoff=p_cutoff
      AND g.contract_pins_sha256=p_pins
      AND r.run_status='ACTIVE' AND r.current_stage='RESEARCH'
      AND r.state_version=p_run_ver AND s.state_version=p_stage_ver
      AND s.lifecycle_status IN ('IN_PROGRESS','PAUSED')
      AND r.data_cutoff=p_cutoff AND r.contract_set_sha256=p_pins
  );
END;
$preflight$;

-- Note: under SERIALIZABLE, a stage change racing with commit must still be
-- tested against real Registry writer semantics. Signed receipt admissions
-- always recheck current versions; no stage gate can use this receipt.
CREATE FUNCTION orotitan_research_intake_private.commit_receipt_v1(
  p_run uuid,p_issuer text,p_grant text,p_subject text,p_client text,
  p_resource text,p_run_ver bigint,p_stage_ver bigint,p_cutoff date,
  p_pins text,p_sha text,p_filename text,p_media text,
  p_path text,p_size bigint,p_source_date text
) RETURNS TABLE (
  receipt_id text,run_id text,issuer_id text,stage text,
  run_state_version bigint,stage_state_version bigint,sha256 text,
  file_name text,size_bytes bigint,submitted_by_subject text,
  duplicate boolean,object_path text
) LANGUAGE plpgsql VOLATILE SECURITY INVOKER
SET search_path=pg_catalog,public,orotitan_research_intake_private
AS $commit$
DECLARE v_new uuid;v_found orotitan_research_intake_private.receipts%ROWTYPE;
BEGIN
  IF current_user<>'orotitan_research_intake_writer'
     OR current_setting('transaction_isolation')<>'serializable' THEN
    RAISE EXCEPTION 'D29B_PRIVATE_COMMIT_ROLE_OR_ISOLATION_DENIED';
  END IF;
  IF p_run IS DISTINCT FROM
     '48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid
     OR NOT orotitan_research_intake_private.preflight_v1(
       p_run,p_issuer,p_grant,p_subject,p_client,p_resource,
       p_run_ver,p_stage_ver,p_cutoff,p_pins) THEN
    RAISE EXCEPTION 'D29B_PRIVATE_GRANT_CAS_DENIED';
  END IF;
  IF p_sha IS NULL OR p_sha !~ '^[0-9a-f]{64}$'
     OR p_filename IS NULL OR p_filename !~
       '^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$'
     OR position('..' in p_filename)>0
     OR p_media IS NULL OR p_media NOT IN
       ('application/json','text/markdown','text/plain','text/csv')
     OR p_size IS NULL OR p_size NOT BETWEEN 1 AND 1000000
     OR p_source_date IS NULL OR p_source_date !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
     OR p_source_date::date > p_cutoff
     OR p_path IS NULL OR p_path !~ (
       '^research-intake/v1/'||p_run::text||'/RESEARCH/'||
       p_sha||'/[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-'||
       '[89ab][0-9a-f]{3}-[0-9a-f]{12}[.](json|md|txt|csv)$'
     ) OR p_media <> CASE right(p_filename,strpos(reverse(p_filename),'.')-1)
       WHEN 'json' THEN 'application/json'
       WHEN 'md' THEN 'text/markdown'
       WHEN 'txt' THEN 'text/plain'
       WHEN 'csv' THEN 'text/csv'
       ELSE 'INVALID' END
     OR right(p_path,strpos(reverse(p_path),'.')-1) <>
       right(p_filename,strpos(reverse(p_filename),'.')-1) THEN
    RAISE EXCEPTION 'D29B_PRIVATE_METADATA_DENIED';
  END IF;
  PERFORM pg_advisory_xact_lock(hashtextextended(p_run::text,71932));
  -- Re-observe versions/grant after acquiring our private writer serialization lock.
  IF NOT orotitan_research_intake_private.preflight_v1(
      p_run,p_issuer,p_grant,p_subject,p_client,p_resource,
      p_run_ver,p_stage_ver,p_cutoff,p_pins) THEN
    RAISE EXCEPTION 'D29B_PRIVATE_CHANGED_AFTER_LOCK';
  END IF;
  INSERT INTO orotitan_research_intake_private.receipts AS new_receipt(
    run_id,issuer_id,stage,run_state_version,stage_state_version,
    contract_pins_sha256,grant_id,sha256,file_name,media_type,object_path,
    size_bytes,source_date,submitted_by_subject,submitted_by_client)
  VALUES(p_run,p_issuer,'RESEARCH',p_run_ver,p_stage_ver,p_pins,p_grant,
    p_sha,p_filename,p_media,p_path,p_size,p_source_date::date,
    p_subject,p_client)
  ON CONFLICT(run_id,stage,run_state_version,stage_state_version,sha256)
    DO NOTHING RETURNING new_receipt.receipt_id INTO v_new;
  SELECT q.* INTO v_found FROM orotitan_research_intake_private.receipts q
    WHERE q.run_id=p_run AND q.stage='RESEARCH'
      AND q.run_state_version=p_run_ver AND q.stage_state_version=p_stage_ver
      AND q.sha256=p_sha;
  IF NOT FOUND OR v_found.file_name<>p_filename OR
     v_found.size_bytes<>p_size OR v_found.media_type<>p_media
     OR v_found.issuer_id<>p_issuer OR
     v_found.submitted_by_subject<>p_subject OR
     v_found.submitted_by_client<>p_client OR
     v_found.contract_pins_sha256<>p_pins THEN
    RAISE EXCEPTION 'D29B_PRIVATE_DUPLICATE_MISMATCH';
  END IF;
  RETURN QUERY SELECT v_found.receipt_id::text,v_found.run_id::text,
    v_found.issuer_id,v_found.stage,v_found.run_state_version,
    v_found.stage_state_version,v_found.sha256,v_found.file_name,
    v_found.size_bytes,v_found.submitted_by_subject,(v_new IS NULL),
    v_found.object_path;
END;
$commit$;

CREATE FUNCTION orotitan_research_intake_private.verifier_metadata_v1(
  p_run uuid,p_issuer text,p_reviewer text,p_run_ver bigint,
  p_stage_ver bigint,p_sha text
) RETURNS TABLE (
  receipt_id text,run_id text,issuer_id text,stage text,
  run_state_version bigint,stage_state_version bigint,sha256 text,
  file_name text,size_bytes bigint,submitted_by_subject text,
  duplicate boolean,object_path text
) LANGUAGE plpgsql STABLE SECURITY INVOKER
SET search_path=pg_catalog,public,orotitan_research_intake_private
AS $reader$
BEGIN
  IF current_user<>'orotitan_research_intake_reader'
     OR p_run IS DISTINCT FROM
       '48b9426b-4d0d-4858-9d87-efcd13ac415d'::uuid
     OR p_sha IS NULL OR p_sha !~ '^[0-9a-f]{64}$' THEN
    RETURN;
  END IF;
  IF NOT EXISTS(
    SELECT 1 FROM orotitan_research_intake_private.verifier_grants g
    JOIN public.orotitan_runs r ON r.run_id=g.run_id
    JOIN public.orotitan_run_stages s ON s.run_id=r.run_id
      AND s.stage_code='RESEARCH'
    WHERE g.run_id=p_run AND g.issuer_id=p_issuer
      AND g.issuer_id=r.issuer_id::text
      AND g.reviewer_id=p_reviewer AND g.permission='READ_RECEIPT_METADATA'
      AND g.revoked_at IS NULL AND now()>=g.issued_at AND now()<g.expires_at
      AND g.expected_run_version=p_run_ver
      AND g.expected_stage_version=p_stage_ver
      AND r.current_stage='RESEARCH' AND r.state_version=p_run_ver
      AND s.state_version=p_stage_ver
  ) THEN RETURN; END IF;
  RETURN QUERY SELECT q.receipt_id::text,q.run_id::text,q.issuer_id,
    q.stage,q.run_state_version,q.stage_state_version,q.sha256,q.file_name,
    q.size_bytes,q.submitted_by_subject,false,q.object_path
    FROM orotitan_research_intake_private.receipts q
    WHERE q.run_id=p_run AND q.issuer_id=p_issuer
      AND q.stage='RESEARCH' AND q.run_state_version=p_run_ver
      AND q.stage_state_version=p_stage_ver AND q.sha256=p_sha;
END;
$reader$;

REVOKE ALL ON ALL FUNCTIONS IN SCHEMA orotitan_research_intake_private FROM PUBLIC;
GRANT EXECUTE ON FUNCTION orotitan_research_intake_private.preflight_v1(
 uuid,text,text,text,text,text,bigint,bigint,date,text
) TO orotitan_research_intake_writer;
GRANT EXECUTE ON FUNCTION orotitan_research_intake_private.commit_receipt_v1(
 uuid,text,text,text,text,text,bigint,bigint,date,text,text,text,text,text,bigint,text
) TO orotitan_research_intake_writer;
GRANT EXECUTE ON FUNCTION orotitan_research_intake_private.verifier_metadata_v1(
 uuid,text,text,bigint,bigint,text
) TO orotitan_research_intake_reader;
-- No grants to public, anon or authenticated.
COMMIT;
