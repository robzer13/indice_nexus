import 'server-only';
import {createClient} from '@supabase/supabase-js';
import {
  isHostedShadowOnboardingOnlyEnabled,
  SHADOW_RESEARCH_ORIGIN,
} from './hosted-shadow-research-gate';

const identityReadPaths=new Set([
  '/rest/v1/issuers',
  '/rest/v1/securities',
  '/rest/v1/research_dossiers',
  '/rest/v1/companies',
  '/rest/v1/legacy_company_identity_map',
  '/rest/v1/orotitan_identity_onboarding_receipts_v1',
]);
const onboardingRpcPath='/rest/v1/rpc/orotitan_onboard_identity_shadow_v1';

/** SHADOW-only allowlist. Identity reads + exactly one P0-D atomic RPC. */
export function getP0DOnboardingShadowClient(){
  if(!isHostedShadowOnboardingOnlyEnabled(process.env))
    throw Error('P0D_SHADOW_CONFIG_DENIED');
  return createClient(SHADOW_RESEARCH_ORIGIN,
    process.env.OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY!,{
      auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},
      global:{fetch:async(input,init)=>{
        const url=new URL(typeof input==='string'?input:
          input instanceof URL?input.href:input.url);
        const method=(init?.method??
          (input instanceof Request?input.method:'GET')).toUpperCase();
        const readAllowed=(method==='GET'||method==='HEAD')&&
          identityReadPaths.has(url.pathname);
        const onboardAllowed=method==='POST'&&url.search===''&&
          url.pathname===onboardingRpcPath;
        if(!isHostedShadowOnboardingOnlyEnabled(process.env)||
           url.origin!==SHADOW_RESEARCH_ORIGIN||
           url.username||url.password||url.hash||
           !(readAllowed||onboardAllowed))
          throw Error('P0D_SHADOW_TRANSPORT_DENIED');
        return fetch(input,{...init,redirect:'error'});
      }},
    });
}
