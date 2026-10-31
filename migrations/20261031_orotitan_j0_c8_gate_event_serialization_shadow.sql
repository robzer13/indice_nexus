-- OroTitan C8 per-milestone journal serialization (SHADOW only).
-- Prevent an operator issue/evidence event from racing a future gate approval.
-- The C8 prepare and accept RPCs already take the same transaction advisory
-- lock. This trigger extends it to every write on the J0 governance journal.
-- It neither enables acceptance nor changes the historic Registry.
BEGIN;
CREATE OR REPLACE FUNCTION public.orotitan_j0_c8_serialize_event_v1()
RETURNS trigger LANGUAGE plpgsql SECURITY INVOKER
SET search_path=pg_catalog,public AS $$
BEGIN
  IF NEW.milestone_code !~ '^J[0-8]$' THEN
    RAISE EXCEPTION 'J0_C8_EVENT_MILESTONE_INVALID';
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'orotitan:j0:gate:'||NEW.milestone_code,0));
  RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS orotitan_j0_c8_serialize_event_insert
 ON public.orotitan_j0_events_v1;
CREATE TRIGGER orotitan_j0_c8_serialize_event_insert
 BEFORE INSERT ON public.orotitan_j0_events_v1
 FOR EACH ROW EXECUTE FUNCTION public.orotitan_j0_c8_serialize_event_v1();
REVOKE ALL ON FUNCTION public.orotitan_j0_c8_serialize_event_v1()
 FROM PUBLIC,anon,authenticated;
COMMIT;