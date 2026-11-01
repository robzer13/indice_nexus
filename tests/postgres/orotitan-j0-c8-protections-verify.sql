-- LOCAL disposable qualification only. All trial lock updates roll back.
DO $guards$
DECLARE
 code text; role_name text; relation_name text; rpc regprocedure;
 source text; rejected_constraint text;
BEGIN
 source:=pg_get_functiondef('public.orotitan_j0_c8_accept_review_v1(text,uuid,text,text,text,text)'::regprocedure);
 IF position('J0_C8_REVIEWER_APPROVER_NOT_DISTINCT' IN source)=0
 OR position('lower(btrim(v_review.reviewer_label))=lower(btrim(p_approver_label))' IN source)=0
 OR position('J0_C8_MILESTONE_ACCEPTANCE_LOCKED' IN source)=0
 OR position('J0_C8_MILESTONE_AUTHORITY_REF_INVALID' IN source)=0
 OR position('J0_C8_REVIEW_STALE' IN source)=0
 OR position('J0_C8_REVIEW_HASH_INTEGRITY_FAILURE' IN source)=0
 THEN RAISE EXCEPTION 'J0_C8_INSTALLED_RPC_PROTECTIONS_MISSING';END IF;
 source:=pg_get_functiondef('public.orotitan_j0_c8_read_gate_v1(text)'::regprocedure);
 IF position('COALESCE(v_mlock,false)' IN source)=0
 THEN RAISE EXCEPTION 'J0_C8_INSTALLED_PROJECTION_DOWNGRADED';END IF;

 FOREACH relation_name IN ARRAY ARRAY[
  'orotitan_j0_c8_acceptance_control_v1','orotitan_j0_c8_milestone_activation_v1',
  'orotitan_j0_c8_reviews_v1','orotitan_j0_c8_decisions_v1'] LOOP
  IF NOT (SELECT relrowsecurity FROM pg_class WHERE oid=('public.'||relation_name)::regclass)
  THEN RAISE EXCEPTION 'J0_C8_RLS_MISSING %',relation_name;END IF;
  FOREACH role_name IN ARRAY ARRAY['anon','authenticated','service_role'] LOOP
   IF has_table_privilege(role_name,'public.'||relation_name,'INSERT,UPDATE,DELETE,TRUNCATE')
   THEN RAISE EXCEPTION 'J0_C8_DIRECT_WRITE_PRIVILEGE % %',role_name,relation_name;END IF;
  END LOOP;
 END LOOP;
 FOR rpc IN SELECT oid::regprocedure FROM pg_proc
  WHERE pronamespace='public'::regnamespace AND proname LIKE 'orotitan_j0_c8_%'
 LOOP
  FOREACH role_name IN ARRAY ARRAY['anon','authenticated'] LOOP
   IF has_function_privilege(role_name,rpc,'EXECUTE')
   THEN RAISE EXCEPTION 'J0_C8_PUBLIC_RPC % %',role_name,rpc;END IF;
  END LOOP;
 END LOOP;
 IF NOT has_function_privilege('service_role',
  'public.orotitan_j0_c8_accept_review_v1(text,uuid,text,text,text,text)','EXECUTE')
 OR NOT has_function_privilege('service_role',
  'public.orotitan_j0_c8_prepare_review_v1(text,text,text,text,text,uuid[])','EXECUTE')
 THEN RAISE EXCEPTION 'J0_C8_SERVICE_RPC_UNAVAILABLE';END IF;
 IF (SELECT count(*) FROM pg_trigger WHERE NOT tgisinternal AND tgenabled='O'
  AND tgname IN ('orotitan_j0_c8_reviews_append_only',
   'orotitan_j0_c8_decisions_append_only','orotitan_j0_c8_serialize_event_insert'))<>3
 THEN RAISE EXCEPTION 'J0_C8_INSTALLED_TRIGGER_MISSING';END IF;

 -- Exercise the real CHECKs, with successful mutations rolled back locally.
 FOREACH code IN ARRAY ARRAY['J0','J1','J2','J3','J4','J5','J6','J7','J8'] LOOP
  BEGIN
   UPDATE public.orotitan_j0_c8_acceptance_control_v1
   SET enabled=true,activation_authority_ref='GO_ACCEPT_'||code||'_LOCAL_CONSTRAINT_ONLY'
   WHERE lock_name='GLOBAL_SHADOW';
   RAISE SQLSTATE 'ZX001';
  EXCEPTION WHEN SQLSTATE 'ZX001' THEN NULL;
  END;
 END LOOP;
 BEGIN
  UPDATE public.orotitan_j0_c8_acceptance_control_v1
  SET enabled=true,activation_authority_ref='GO_ACCEPT_J9_LOCAL_CONSTRAINT_ONLY';
  RAISE EXCEPTION 'J0_C8_J9_AUTHORITY_ALLOWED';
 EXCEPTION WHEN check_violation THEN
  GET STACKED DIAGNOSTICS rejected_constraint=CONSTRAINT_NAME;
  IF rejected_constraint<>'j0_c8_disabled_without_authority' THEN RAISE;END IF;
 END;
 BEGIN
  UPDATE public.orotitan_j0_c8_milestone_activation_v1
  SET enabled=true,authority_ref='GO_ACCEPT_J0_LOCAL_CONSTRAINT_ONLY' WHERE milestone_code='J1';
  RAISE EXCEPTION 'J0_C8_MISMATCHED_MILESTONE_AUTHORITY_ALLOWED';
 EXCEPTION WHEN check_violation THEN
  GET STACKED DIAGNOSTICS rejected_constraint=CONSTRAINT_NAME;
  IF rejected_constraint<>'j0_c8_milestone_authority_match' THEN RAISE;END IF;
 END;
END $guards$;

-- Fingerprint actual installed definitions, constraints, ACLs/RLS and triggers.
-- Include the Registry pin CHECK/helper so replay cannot weaken the fixture gate.
WITH definitions AS (
 SELECT 'function:'||oid::regprocedure::text AS key,
  pg_get_functiondef(oid)||coalesce(proacl::text,'') AS definition
 FROM pg_proc WHERE pronamespace='public'::regnamespace
 AND (proname LIKE 'orotitan_j0_c8_%' OR proname='orotitan_contract_pins_complete')
 UNION ALL
 SELECT 'table:'||relname,relrowsecurity::text||coalesce(relacl::text,'')
 FROM pg_class WHERE relnamespace='public'::regnamespace AND relname LIKE 'orotitan_j0_c8_%'
 UNION ALL
 SELECT 'constraint:'||conrelid::regclass::text||':'||conname,pg_get_constraintdef(oid)||convalidated::text
 FROM pg_constraint WHERE conrelid IN (
  SELECT oid FROM pg_class WHERE relnamespace='public'::regnamespace AND relname LIKE 'orotitan_j0_c8_%'
 ) OR conname='orotitan_runs_contract_pins_complete_check'
 UNION ALL
 SELECT 'trigger:'||tgname,pg_get_triggerdef(oid)||tgenabled::text
 FROM pg_trigger WHERE NOT tgisinternal AND tgname LIKE 'orotitan_j0_c8_%'
)
SELECT md5(string_agg(key||E'\n'||definition,E'\n' ORDER BY key COLLATE "C")) FROM definitions;
