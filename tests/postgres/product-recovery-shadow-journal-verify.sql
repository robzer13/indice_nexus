-- Disposable PG17 verification only. The runner refuses remote database hosts.
-- Assertions execute inside a rollback-only transaction; no test evidence persists.
BEGIN;
DO $test$
DECLARE
  v_run uuid := '55555555-5555-4555-8555-555555555555';
  v_count integer;
  v_update_denied boolean := false;
  v_delete_denied boolean := false;
BEGIN
  IF NOT (SELECT relrowsecurity FROM pg_class
          WHERE oid = 'public.orotitan_product_recovery_attempts_shadow_v1'::regclass) THEN
    RAISE EXCEPTION 'RECOVERY_RLS_DISABLED';
  END IF;
  IF has_table_privilege('anon','public.orotitan_product_recovery_attempts_shadow_v1','SELECT')
     OR has_table_privilege('authenticated','public.orotitan_product_recovery_attempts_shadow_v1','SELECT')
     OR NOT has_table_privilege('service_role','public.orotitan_product_recovery_attempts_shadow_v1','SELECT')
     OR NOT has_table_privilege('service_role','public.orotitan_product_recovery_attempts_shadow_v1','INSERT')
     OR has_table_privilege('service_role','public.orotitan_product_recovery_attempts_shadow_v1','UPDATE')
     OR has_table_privilege('service_role','public.orotitan_product_recovery_attempts_shadow_v1','DELETE') THEN
    RAISE EXCEPTION 'RECOVERY_JOURNAL_PRIVILEGES_INVALID';
  END IF;

  INSERT INTO public.orotitan_product_recovery_attempts_shadow_v1
    (run_id,stage_code,run_state_version,stage_state_version,
     blocker_code,strategy,outcome,detail,evidence_refs)
  VALUES
    (v_run,'DEEP_DIVE',6,2,'DD-PRIMARY-SOURCES','REUSE_PRIMARY_ARTIFACTS',
     'NO_PROGRESS','SYNTHETIC TEST','[]'::jsonb),
    (v_run,'DEEP_DIVE',6,2,'ASML-G001','RECONCILE_SHARE_COUNT',
     'REQUIRES_AUTHORITY','SYNTHETIC TEST','[]'::jsonb),
    (v_run,'DEEP_DIVE',6,2,'ASML-G006','RECONCILE_HISTORICAL_PRICE',
     'REQUIRES_AUTHORITY','SYNTHETIC TEST','[]'::jsonb),
    (v_run,'DEEP_DIVE',6,2,'DD-ROIC-OE','REBUILD_ROIC_OWNER_EARNINGS',
     'REQUIRES_AUTHORITY','SYNTHETIC TEST','[]'::jsonb);

  SELECT count(*) INTO v_count FROM public.orotitan_product_recovery_attempts_shadow_v1
  WHERE run_id=v_run AND stage_state_version=2;
  IF v_count <> 4 THEN RAISE EXCEPTION 'RECOVERY_EXPECTED_FOUR_ATTEMPTS'; END IF;

  INSERT INTO public.orotitan_product_recovery_attempts_shadow_v1
    (run_id,stage_code,run_state_version,stage_state_version,
     blocker_code,strategy,outcome,detail,evidence_refs)
  VALUES
    (v_run,'DEEP_DIVE',6,2,'DD-PRIMARY-SOURCES','REUSE_PRIMARY_ARTIFACTS',
     'NO_PROGRESS','SYNTHETIC TEST','[]'::jsonb),
    (v_run,'DEEP_DIVE',6,2,'ASML-G001','RECONCILE_SHARE_COUNT',
     'REQUIRES_AUTHORITY','SYNTHETIC TEST','[]'::jsonb),
    (v_run,'DEEP_DIVE',6,2,'ASML-G006','RECONCILE_HISTORICAL_PRICE',
     'REQUIRES_AUTHORITY','SYNTHETIC TEST','[]'::jsonb),
    (v_run,'DEEP_DIVE',6,2,'DD-ROIC-OE','REBUILD_ROIC_OWNER_EARNINGS',
     'REQUIRES_AUTHORITY','SYNTHETIC TEST','[]'::jsonb)
  ON CONFLICT DO NOTHING;

  SELECT count(*) INTO v_count FROM public.orotitan_product_recovery_attempts_shadow_v1
  WHERE run_id=v_run AND stage_state_version=2;
  IF v_count <> 4 THEN RAISE EXCEPTION 'RECOVERY_REPLAY_INSERTED_DUPLICATES'; END IF;

  BEGIN
    UPDATE public.orotitan_product_recovery_attempts_shadow_v1
    SET detail='FORBIDDEN' WHERE run_id=v_run;
  EXCEPTION WHEN OTHERS THEN
    IF SQLERRM LIKE '%OROTITAN_SHADOW_RECOVERY_JOURNAL_APPEND_ONLY%' THEN
      v_update_denied := true;
    ELSE
      RAISE;
    END IF;
  END;
  BEGIN
    DELETE FROM public.orotitan_product_recovery_attempts_shadow_v1 WHERE run_id=v_run;
  EXCEPTION WHEN OTHERS THEN
    IF SQLERRM LIKE '%OROTITAN_SHADOW_RECOVERY_JOURNAL_APPEND_ONLY%' THEN
      v_delete_denied := true;
    ELSE
      RAISE;
    END IF;
  END;
  IF NOT v_update_denied OR NOT v_delete_denied THEN
    RAISE EXCEPTION 'RECOVERY_APPEND_ONLY_VIOLATION';
  END IF;

  INSERT INTO public.orotitan_product_recovery_attempts_shadow_v1
    (run_id,stage_code,run_state_version,stage_state_version,
     blocker_code,strategy,outcome,detail,evidence_refs)
  VALUES
    (v_run,'DEEP_DIVE',6,3,'ASML-G001','RECONCILE_SHARE_COUNT',
     'NO_PROGRESS','SYNTHETIC NEW REVISION','[]'::jsonb);
  SELECT count(*) INTO v_count FROM public.orotitan_product_recovery_attempts_shadow_v1
  WHERE run_id=v_run;
  IF v_count <> 5 THEN RAISE EXCEPTION 'RECOVERY_STAGE_REVISION_NOT_RETRYABLE'; END IF;
END
$test$;
ROLLBACK;

DO $verify$
BEGIN
  IF EXISTS (SELECT 1 FROM public.orotitan_product_recovery_attempts_shadow_v1) THEN
    RAISE EXCEPTION 'RECOVERY_TRANSACTION_LEFT_ROWS';
  END IF;
  IF to_regclass('public.orotitan_runs') IS NOT NULL
     OR to_regclass('public.orotitan_run_stages') IS NOT NULL
     OR to_regclass('public.orotitan_artifacts') IS NOT NULL THEN
    RAISE EXCEPTION 'RECOVERY_TEST_MUST_NOT_LOAD_REGISTRY';
  END IF;
END
$verify$;
SELECT 'PRODUCT RECOVERY SHADOW PG17 PASS: 4 attempts, dedupe, write-guards, retry, rollback' AS result;
