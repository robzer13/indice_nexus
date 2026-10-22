-- D27 REVIEW ONLY — NEVER RUN ON SUPABASE PRODUCTION WITHOUT FRESH APPROVAL.
-- This is NOT a migration. Do not register the function as an Auth Hook yet.
-- Requires replacement of BOTH invalid placeholders in a reviewed deployment
-- copy, never in a public Git commit. For isolated PG17 test only use fake IDs.
--
-- Official inputs: event jsonb {user_id, claims, authentication_method}.
-- Supabase Auth signs the JWT after this hook; this function NEVER signs tokens.
-- Keep the standard aud='authenticated' (required by Supabase Data API).
-- OAuth client identification comes from Supabase-supplied signed client_id;
-- do not trust arbitrary request parameters, URLs, or user_metadata.
--
-- DEPLOYMENT BLOCKERS: real OAuth client registration, verified claim shape,
-- actual Auth user, separate shared-project Auth Hook approval, consent Site URL
-- review, and standalone Vercel pilot with NO service_role key.

begin;

create schema if not exists orotitan_oauth_internal;
revoke all on schema orotitan_oauth_internal from public;
grant usage on schema orotitan_oauth_internal to supabase_auth_admin;

create function orotitan_oauth_internal.mcp_resource_claim_hook(event jsonb)
returns jsonb
language sql
security invoker
set search_path = ''
as $mcp_hook$
  select jsonb_build_object(
    'claims',
    case when
      event ->> 'user_id' = '__APPROVED_SUPABASE_AUTH_USER_UUID__'
      and event -> 'claims' ->> 'sub' = '__APPROVED_SUPABASE_AUTH_USER_UUID__'
      and event -> 'claims' ->> 'client_id' = '__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__'
      and event -> 'claims' ->> 'role' = 'authenticated'
      and event -> 'claims' ->> 'aud' = 'authenticated'
      and event -> 'claims' ->> 'iss' = 'https://cugpgtzygqqlxetyeven.supabase.co/auth/v1'
      and event -> 'claims' ->> 'is_anonymous' = 'false'
      then jsonb_set(
        event -> 'claims', '{resource}',
        to_jsonb('https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'::text),
        true
      )
      else event -> 'claims'
    end
  );
$mcp_hook$;

-- Function privileges: caller is only the Supabase Auth hook execution role.
revoke all on function orotitan_oauth_internal.mcp_resource_claim_hook(jsonb)
  from public, anon, authenticated;
grant execute on function orotitan_oauth_internal.mcp_resource_claim_hook(jsonb)
  to supabase_auth_admin;

commit;
