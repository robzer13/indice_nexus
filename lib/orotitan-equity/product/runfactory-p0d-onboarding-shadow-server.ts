import 'server-only';
import {createClient} from '@supabase/supabase-js';
import {pgrstClockSkewRetryFetch} from '@/lib/supabase/pgrst-clock-retry';
import {
  isHostedShadowOnboardingOnlyEnabled,isP0DOnboardingTransportAllowed,
  SHADOW_RESEARCH_ORIGIN,
} from './hosted-shadow-research-gate';


/** SHADOW-only allowlist. Identity reads + exactly one P0-D atomic RPC. */
export function getP0DOnboardingShadowClient(){
  if(!isHostedShadowOnboardingOnlyEnabled(process.env))
    throw Error('P0D_SHADOW_CONFIG_DENIED');
  const readWithClockSkewGuard=pgrstClockSkewRetryFetch(fetch,SHADOW_RESEARCH_ORIGIN);
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
        // Retry only a confirmed PGRST303 future-JWT 401 on a GET/HEAD.
        // Identity onboarding RPC POST is never retried or rewritten.
        return readWithClockSkewGuard(input,{...init,redirect:'error'});
      }},
    });
}
