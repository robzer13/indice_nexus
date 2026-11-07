/**
 * Standalone, fail-closed gate for the single authorized hosted Research intake
 * PREVIEW on the existing SHADOW Supabase project.
 *
 * This is a request-receipt UI only. No Research run or publication privileges.
 * Never accept arbitrary project refs in hosted deployments.
 */
export type HostedShadowEnv = Readonly<Record<string, string | undefined>>;

export const SHADOW_RESEARCH_PROJECT_REF = 'awgsurdyvsyolcgpnygh';
export const SHADOW_RESEARCH_ORIGIN =
  'https://awgsurdyvsyolcgpnygh.supabase.co';

/** Presence reserves the deployment for C8; malformed values never fall back to Research. */
export function isC8ReadOnlyRequested(env: HostedShadowEnv): boolean {
  return env.OROTITAN_C8_READ_ONLY_ENABLED !== undefined;
}

export function assertOutsideC8ReadOnly(env: HostedShadowEnv): void {
  if (isC8ReadOnlyRequested(env)) throw new Error('C8_READ_ONLY_MUTATION_DENIED');
}

function isHostedShadowConfigurationValid(env: HostedShadowEnv): boolean {
  const secret = env.OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY;
  const appSecret = env.SUPABASE_SERVICE_ROLE_KEY;
  const appUrl = env.NEXT_PUBLIC_SUPABASE_URL;
  const sharedUrl = env.OROTITAN_SHARED_INBOX_TEST_URL;

  return env.NODE_ENV === 'production' &&
    env.VERCEL_ENV === 'preview' &&
    env.OROTITAN_SHARED_INBOX_TEST_PROJECT_REF === SHADOW_RESEARCH_PROJECT_REF &&
    env.OROTITAN_EXPECTED_SUPABASE_PROJECT_REF === SHADOW_RESEARCH_PROJECT_REF &&
    // Require BOTH the ordinary Screener client and Research intake client
    // to target SHADOW, not the existing production-linked Preview env.
    (appUrl === SHADOW_RESEARCH_ORIGIN || appUrl === SHADOW_RESEARCH_ORIGIN + '/') &&
    (sharedUrl === SHADOW_RESEARCH_ORIGIN || sharedUrl === SHADOW_RESEARCH_ORIGIN + '/') &&
    typeof secret === 'string' && secret.length >= 40 &&
    // Prevent a split credential context where the rest of the app can still
    // use a production service_role client while Research uses SHADOW.
    appSecret === secret &&
    typeof env.ADMIN_PASSWORD === 'string' && env.ADMIN_PASSWORD.length >= 16 &&
    typeof env.ADMIN_SESSION_SECRET === 'string' &&
    env.ADMIN_SESSION_SECRET.length >= 32;
}

export function isHostedC8ReadOnlyEnabled(env: HostedShadowEnv): boolean {
  return env.OROTITAN_C8_READ_ONLY_ENABLED === '1' &&
    isHostedShadowConfigurationValid(env);
}

export function isHostedShadowResearchEnabled(env: HostedShadowEnv): boolean {
  return !isC8ReadOnlyRequested(env) &&
    env.OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED === '1' &&
    env.OROTITAN_SHARED_INBOX_TEST_ENABLED === '1' &&
    isHostedShadowConfigurationValid(env);
}


/**
 * J1-only exception inside a C8-isolated SHADOW Preview.
 * This does not enable Research, stages, artifact writes, or publication.
 * The transport-level allowlist is enforced separately by the J1 client.
 */
export function isHostedJ1RunFactoryEnabled(env: HostedShadowEnv): boolean {
  return env.OROTITAN_J1_RUN_FACTORY_ENABLED === '1' &&
    env.OROTITAN_C8_ACCEPTANCE_GO_ENABLED === '0' &&
    isHostedC8ReadOnlyEnabled(env);
}


/**
 * Dedicated P0-C/P0-D identity onboarding Preview. No J0/C8 overrides,
 * no J1 Run creation, and no general hosted Research capability.
 * The associated Supabase transport grants read-only identity lookup and
 * precisely one governed onboarding RPC.
 */
export function isHostedShadowOnboardingOnlyEnabled(env:HostedShadowEnv):boolean{
  return !isC8ReadOnlyRequested(env) &&
    env.OROTITAN_HOSTED_SHADOW_ONBOARDING_ONLY_ENABLED === '1' &&
    env.OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED === '0' &&
    env.OROTITAN_SHARED_INBOX_TEST_ENABLED === '0' &&
    env.OROTITAN_J1_RUN_FACTORY_ENABLED === undefined &&
    isHostedShadowConfigurationValid(env);
}


const P0D_IDENTITY_READ_PATHS=new Set([
  '/rest/v1/issuers',
  '/rest/v1/securities',
  '/rest/v1/research_dossiers',
  '/rest/v1/companies',
  '/rest/v1/legacy_company_identity_map',
  '/rest/v1/orotitan_identity_onboarding_receipts_v1',
]);

/** Pure policy also used in tests. No generic PostgREST writes or Run RPCs. */
export function isP0DOnboardingTransportAllowed(url:URL, method:string):boolean{
  if(url.origin!==SHADOW_RESEARCH_ORIGIN||url.username||url.password||
     url.hash)return false;
  if((method==='GET'||method==='HEAD')&&P0D_IDENTITY_READ_PATHS.has(url.pathname))
    return true;
  return method==='POST'&&url.search===''&&
    url.pathname==='/rest/v1/rpc/orotitan_onboard_identity_shadow_v1';
}
