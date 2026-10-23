-- D29b disposable PG17 fixture only. Never use this in an existing Supabase DB.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
DO $roles$
BEGIN
  IF NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname='anon') THEN
    CREATE ROLE anon NOLOGIN;
  END IF;
  IF NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname='authenticated') THEN
    CREATE ROLE authenticated NOLOGIN;
  END IF;
END;
$roles$;
CREATE TABLE public.orotitan_runs(
  run_id uuid PRIMARY KEY,
  issuer_id uuid NOT NULL,
  run_status text NOT NULL,
  current_stage text NOT NULL,
  state_version bigint NOT NULL,
  data_cutoff date NOT NULL,
  contract_set_sha256 text NOT NULL
);
CREATE TABLE public.orotitan_run_stages(
  run_id uuid NOT NULL,
  stage_code text NOT NULL,
  state_version bigint NOT NULL,
  lifecycle_status text NOT NULL,
  PRIMARY KEY(run_id,stage_code)
);
ALTER TABLE public.orotitan_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.orotitan_run_stages ENABLE ROW LEVEL SECURITY;
INSERT INTO public.orotitan_runs VALUES(
  '48b9426b-4d0d-4858-9d87-efcd13ac415d',
  '33333333-3333-4333-8333-333333333333','ACTIVE','RESEARCH',6,
  '2026-10-12',repeat('a',64)
),(
  'd56a0bc9-4e80-48b3-b326-74d1fea15e63',
  '44444444-4444-4444-8444-444444444444','ACTIVE','RESEARCH',6,
  '2026-10-12',repeat('a',64)
);
INSERT INTO public.orotitan_run_stages VALUES(
  '48b9426b-4d0d-4858-9d87-efcd13ac415d','RESEARCH',4,'IN_PROGRESS'
),(
  'd56a0bc9-4e80-48b3-b326-74d1fea15e63','RESEARCH',4,'IN_PROGRESS'
);
