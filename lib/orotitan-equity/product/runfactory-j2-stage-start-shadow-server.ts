import 'server-only';
import {createClient} from '@supabase/supabase-js';
import {pgrstClockSkewRetryFetch} from '@/lib/supabase/pgrst-clock-retry';
import {isHostedJ2ResearchStageStartEnabled,isHostedJ2ResearchPreviewEnabled,SHADOW_RESEARCH_ORIGIN} from
 './hosted-shadow-research-gate';

const readOnlyPaths=new Set([
 '/rest/v1/issuers','/rest/v1/securities','/rest/v1/research_dossiers',
 '/rest/v1/research_snapshots','/rest/v1/orotitan_runs',
 '/rest/v1/orotitan_run_events','/rest/v1/orotitan_run_stages',
 '/rest/v1/orotitan_p0e_run_admission_receipts_v1',
]);

/** The J2 Preview can read the necessary Registry rows and call exactly one
 * canonical stage-start RPC. All other PostgREST mutations are denied.
 * No client request may supply a project URL or credential.
 * NEVER mount this transport on J1/C8 or a non-isolated Vercel project.
 */
export function getJ2ResearchStageStartShadowClient(){
 if(!isHostedJ2ResearchPreviewEnabled(process.env))
  throw Error('J2_RESEARCH_PREVIEW_CONFIG_DENIED');
 const safeRead=pgrstClockSkewRetryFetch(fetch,SHADOW_RESEARCH_ORIGIN);
 return createClient(SHADOW_RESEARCH_ORIGIN,
  process.env.OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY!,{
   auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},
   global:{fetch:async(input,init)=>{
    const url=new URL(typeof input==='string'?input:
     input instanceof URL?input.href:input.url);
    const method=(init?.method??
     (input instanceof Request?input.method:'GET')).toUpperCase();
    const read=(method==='GET'||method==='HEAD')&&
     readOnlyPaths.has(url.pathname);
    const stageStart=isHostedJ2ResearchStageStartEnabled(process.env)&&
     method==='POST'&&url.search===''&&
     url.pathname==='/rest/v1/rpc/start_orotitan_stage';
    if(!isHostedJ2ResearchPreviewEnabled(process.env)||
       url.origin!==SHADOW_RESEARCH_ORIGIN||
       url.username||url.password||url.hash||
       !(read||stageStart))
     throw Error('J2_RESEARCH_STAGE_START_TRANSPORT_DENIED');
    // The bounded clock-skew read workaround must never replay a POST.
    return safeRead(input,{...init,redirect:'error'});
   }},
  });
}
