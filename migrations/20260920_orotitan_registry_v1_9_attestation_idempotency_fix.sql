-- OroTitan Registry V1.9 — persistence attestation idempotency expression fix.
-- Implementation repair only. No analytical methodology, Contract Set,
-- snapshot semantics, scoring, valuation, I2 or I3-B change.
--
-- V1.8 introduced management-plane persistence attestations. PostgreSQL
-- operator precedence makes the unparenthesized jsonb ->> expressions in the
-- idempotency-key concatenation resolve incorrectly at runtime. This repair
-- changes only that expression and preserves every validation and event field.

begin;

create or replace function public.attest_orotitan_persistence_locator(
  p_run_id uuid,
  p_stage_code text,
  p_attestation_payload jsonb
)
returns uuid
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_event_id uuid;
  v_fingerprint text;
  v_idempotency_key text;
  v_retry jsonb;
begin
  if coalesce(jsonb_typeof(p_attestation_payload),'null') <> 'object'
     or p_attestation_payload->>'attestation_schema_version' is distinct from '1.0'
     or p_attestation_payload->>'verification_method' is distinct from 'GITHUB_CONNECTOR_PRIVATE_REREAD_V1'
     or p_attestation_payload->>'trust_boundary' is distinct from 'SUPABASE_MANAGEMENT_PLANE'
     or p_attestation_payload->>'storage_backend' is distinct from 'PRIVATE_GITHUB'
     or p_attestation_payload->>'github_repository' is distinct from 'robzer13/real-orotitan'
     or p_attestation_payload->>'github_commit_sha' !~ '^[0-9a-f]{40}$'
     or p_attestation_payload->>'github_blob_sha' !~ '^[0-9a-f]{40}$'
     or p_attestation_payload->>'content_sha256' !~ '^[0-9a-f]{64}$'
     or coalesce((p_attestation_payload->>'commit_path_resolved')::boolean,false) is not true
     or (p_attestation_payload->>'run_id')::uuid is distinct from p_run_id
     or p_attestation_payload->>'stage_code' is distinct from p_stage_code then
    raise exception 'PERSISTENCE_ATTESTATION_INVALID'
      using errcode = '23514';
  end if;

  if not exists (
    select 1 from public.orotitan_runs r
    join public.orotitan_run_stages s on s.run_id=r.run_id
    where r.run_id=p_run_id
      and r.current_stage=p_stage_code
      and r.run_status not in ('PUBLISHED','CANCELLED')
      and s.stage_code=p_stage_code
      and s.lifecycle_status <> 'COMPLETE'
  ) then
    raise exception 'PERSISTENCE_ATTESTATION_STAGE_NOT_ACTIVE'
      using errcode = '23514';
  end if;

  v_fingerprint := encode(
    extensions.digest(convert_to(p_attestation_payload::text,'UTF8'),'sha256'),
    'hex'
  );
  v_idempotency_key := 'persistence-attest:' ||
    (p_attestation_payload->>'artifact_id') || ':' ||
    (p_attestation_payload->>'version') || ':' ||
    v_fingerprint;

  v_retry := public.orotitan_existing_event(p_run_id,v_idempotency_key,v_fingerprint);
  if v_retry is not null then
    return (v_retry->>'event_id')::uuid;
  end if;

  v_event_id := public.orotitan_insert_event(
    p_run_id,p_stage_code,'PERSISTENCE_ATTESTED',
    v_idempotency_key,v_fingerprint,'SYSTEM',p_attestation_payload
  );
  return v_event_id;
end;
$$;

commit;
