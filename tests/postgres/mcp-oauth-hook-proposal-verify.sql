-- D27 isolated tests; run only from the loopback-only harness.
do $check$
begin
  if not has_function_privilege('supabase_auth_admin',
    'orotitan_oauth_internal.mcp_resource_claim_hook(jsonb)','EXECUTE') then
    raise exception 'supabase_auth_admin lacks hook execute';
  end if;
  if has_function_privilege('authenticated',
    'orotitan_oauth_internal.mcp_resource_claim_hook(jsonb)','EXECUTE') or
     has_function_privilege('anon',
    'orotitan_oauth_internal.mcp_resource_claim_hook(jsonb)','EXECUTE') then
    raise exception 'public roles must never execute the JWT hook';
  end if;
end;
$check$;

set role supabase_auth_admin;

do $matrix$
declare
  base jsonb := jsonb_build_object(
    'user_id','11111111-1111-4111-8111-111111111111',
    'authentication_method','oauth_provider/authorization_code',
    'claims',jsonb_build_object(
      'iss','https://cugpgtzygqqlxetyeven.supabase.co/auth/v1',
      'aud','authenticated',
      'sub','11111111-1111-4111-8111-111111111111',
      'role','authenticated',
      'client_id','approved-client-test-only',
      'is_anonymous',false,
      'email','user@example.org',
      'exp',1792500000,
      'iat',1792499000
    )
  );
  allowed_resource text := 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp';
  changed jsonb;
  candidate jsonb;
  forbidden jsonb[];
begin
  select orotitan_oauth_internal.mcp_resource_claim_hook(base) into changed;
  if changed->'claims'->>'resource' is distinct from allowed_resource then
    raise exception 'matching OAuth identity lacks resource claim';
  end if;
  if changed->'claims'->>'email' != 'user@example.org' or
     changed->'claims'->>'aud' != 'authenticated' or
     changed->'claims'->>'client_id' != 'approved-client-test-only' or
     changed->'claims'->>'sub' != base->'claims'->>'sub' or
     changed->'claims'->>'exp' != base->'claims'->>'exp' then
    raise exception 'required Supabase claims were modified';
  end if;

  -- A refresh must retain client/resource binding when Supabase includes
  -- the verified client_id; the hook should not check UI login method.
  candidate := jsonb_set(base,'{authentication_method}',to_jsonb('token_refresh'::text));
  select orotitan_oauth_internal.mcp_resource_claim_hook(candidate) into changed;
  if changed->'claims'->>'resource' != allowed_resource then
    raise exception 'authorized token refresh did not retain resource binding';
  end if;

  forbidden := array[
    jsonb_set(base,'{user_id}',to_jsonb('22222222-2222-4222-8222-222222222222'::text)),
    jsonb_set(base,'{claims,sub}',to_jsonb('22222222-2222-4222-8222-222222222222'::text)),
    jsonb_set(base,'{claims,client_id}',to_jsonb('unknown-client'::text)),
    base #- '{claims,client_id}',
    jsonb_set(base,'{claims,role}',to_jsonb('anon'::text)),
    jsonb_set(base,'{claims,aud}',to_jsonb('other'::text)),
    jsonb_set(base,'{claims,iss}',to_jsonb('https://untrusted.example.com'::text)),
    jsonb_set(base,'{claims,is_anonymous}','true'::jsonb),
    base #- '{claims,is_anonymous}',
    base #- '{claims,sub}'
  ];
  foreach candidate in array forbidden loop
    select orotitan_oauth_internal.mcp_resource_claim_hook(candidate) into changed;
    if changed->'claims' is distinct from candidate->'claims' then
      raise exception 'unauthorized event claims changed: %',candidate;
    end if;
    if changed->'claims' ? 'resource' then
      raise exception 'unauthorized OAuth event received resource grant';
    end if;
  end loop;
  -- Wrong inbound resource is forcibly replaced only for the approved pair.
  candidate := jsonb_set(base,'{claims,resource}',to_jsonb('https://attacker.invalid'::text));
  select orotitan_oauth_internal.mcp_resource_claim_hook(candidate) into changed;
  if changed->'claims'->>'resource' != allowed_resource then
    raise exception 'resource claim was not pinned';
  end if;
end;
$matrix$;
reset role;
