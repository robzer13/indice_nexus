-- D29b — PROPOSAL ONLY. DO NOT EXECUTE IN LIVE SUPABASE.
-- Revocation is a trusted internal operation: mark revoked_at and make RLS hide
-- the receipt on the next read. No OAuth write privileges are granted.
-- A separately approved, audited signer must populate signed receipts only after
-- independently completing D2 immutable-byte/hash/contract/gate verification.
-- This is *not* the existing D28 GRANT/RLS migration and is not authorized here.
-- Substitute 3 placeholders only in a future private, reviewed rollout process.
begin;

do $preflight$
begin
  if current_setting('server_version_num')::int < 170000
     or to_regclass('public.orotitan_mcp_verified_handoffs') is not null
     or to_regclass('public.orotitan_runs') is null
     or to_regprocedure('auth.jwt()') is null
     or to_regprocedure('auth.uid()') is null
  then
    raise exception 'D29b expected schema/version absent or changed';
  end if;
end;
$preflight$;

-- In public only so PostgREST can access it through the caller's JWT under RLS.
-- This stores a *short-lived signed instruction/receipt*, never source artifacts.
create table public.orotitan_mcp_verified_handoffs (
  receipt_id uuid primary key default gen_random_uuid(),
  run_id uuid not null references public.orotitan_runs(run_id),
  subject_id uuid not null,
  oauth_client_id text not null,
  mcp_resource text not null,
  key_id text not null check (length(key_id) between 1 and 64),
  payload_base64url text not null check (length(payload_base64url) between 100 and 24000),
  signature_base64url text not null check (length(signature_base64url) between 80 and 100),
  issued_at timestamptz not null,
  expires_at timestamptz not null,
  revoked_at timestamptz null,
  created_at timestamptz not null default now(),
  constraint d29b_ttl_short
    check(expires_at > issued_at and expires_at <= issued_at + interval '5 minutes'),
  constraint d29b_unique_single_receipt
    unique (run_id,subject_id,oauth_client_id,mcp_resource)
);

alter table public.orotitan_mcp_verified_handoffs enable row level security;
revoke all on public.orotitan_mcp_verified_handoffs from public,anon,authenticated;

-- No update, insert, delete, raw artifacts, private repositories or RPC grants.
grant select (receipt_id,run_id,key_id,payload_base64url,signature_base64url,expires_at)
  on public.orotitan_mcp_verified_handoffs to authenticated;

create policy d29b_besi_oauth_receipt_read_only
on public.orotitan_mcp_verified_handoffs
for select to authenticated
using (
  run_id = '__BESI_RUN_UUID__'::uuid
  and subject_id = '__APPROVED_SUPABASE_AUTH_USER_UUID__'::uuid
  and subject_id = (select auth.uid())
  and oauth_client_id = '__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__'
  and oauth_client_id = (select auth.jwt()->>'client_id')
  and mcp_resource =
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
  and mcp_resource = (select auth.jwt()->>'resource')
  and expires_at > now()
  and revoked_at is null
);

-- Independently approved writer/signing automation and key custody are OUT OF SCOPE.
-- The JWT-verifying MCP endpoint must still recheck exact current CAS versions,
-- signed contract hash, signed D2 provenance and expiry AFTER RLS retrieval.
commit;
