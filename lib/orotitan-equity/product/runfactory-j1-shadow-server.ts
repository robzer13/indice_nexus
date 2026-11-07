import 'server-only';
import {createClient} from '@supabase/supabase-js';
import {pgrstClockSkewRetryFetch} from '@/lib/supabase/pgrst-clock-retry';
import {
  isHostedJ1RunFactoryEnabled,
  SHADOW_RESEARCH_ORIGIN,
} from './hosted-shadow-research-gate';

const readOnlyRegistryPaths = new Set([
  '/rest/v1/issuers',
  '/rest/v1/securities',
  '/rest/v1/research_dossiers',
  '/rest/v1/research_snapshots',
  '/rest/v1/orotitan_runs',
  '/rest/v1/orotitan_run_events',
  '/rest/v1/orotitan_run_stages',
  '/rest/v1/orotitan_p0e_run_admission_receipts_v1',
]);

/** Only the V3.1 atomic INITIAL Run RPC may write. Never permit stages/Research. */
export function getJ1RunFactoryShadowClient() {
  if(!isHostedJ1RunFactoryEnabled(process.env))
    throw Error('J1_SHADOW_CONFIG_DENIED');
  const safeReadFetch=pgrstClockSkewRetryFetch(fetch,SHADOW_RESEARCH_ORIGIN);
  return createClient(SHADOW_RESEARCH_ORIGIN,
    process.env.OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY!,{
      auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},
      global:{fetch:async(input,init)=>{
        const url=new URL(typeof input==='string'?input:
          input instanceof URL?input.href:input.url);
        const method=(init?.method??
          (input instanceof Request?input.method:'GET')).toUpperCase();
        const readAllowed=(method==='GET'||method==='HEAD')&&
          readOnlyRegistryPaths.has(url.pathname);
        const runAdmissionAllowed=method==='POST'&&url.search===''&&
          url.pathname==='/rest/v1/rpc/orotitan_shadow_create_initial_run_v3_1';
        if(!isHostedJ1RunFactoryEnabled(process.env)||
           url.origin!==SHADOW_RESEARCH_ORIGIN||
           url.username||url.password||url.hash||
           !(readAllowed||runAdmissionAllowed))
          throw Error('J1_SHADOW_TRANSPORT_DENIED');
        // Only confirmed PGRST303 future-JWT GET/HEAD reads may retry.
        // Admission RPC POST is never retried automatically.
        return safeReadFetch(input,{...init,redirect:'error'});
      }},
    });
}
