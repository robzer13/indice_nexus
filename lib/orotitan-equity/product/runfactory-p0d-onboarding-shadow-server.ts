import 'server-only';
import {createClient} from '@supabase/supabase-js';
import {
  isHostedShadowOnboardingOnlyEnabled,isP0DOnboardingTransportAllowed,
  SHADOW_RESEARCH_ORIGIN,
} from './hosted-shadow-research-gate';


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
        if(!isHostedShadowOnboardingOnlyEnabled(process.env)||
           !isP0DOnboardingTransportAllowed(url,method))
          throw Error('P0D_SHADOW_TRANSPORT_DENIED');
        return fetch(input,{...init,redirect:'error'});
      }},
    });
}
