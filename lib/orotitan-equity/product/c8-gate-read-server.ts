import 'server-only';
import {createClient} from '@supabase/supabase-js';
import {
  isC8ReadOnlyRequested, isHostedC8ReadOnlyEnabled,
  SHADOW_RESEARCH_ORIGIN,
} from './hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from './research-inbox-pilot-server';

const readPaths = new Set([
  '/rest/v1/orotitan_j0_c8_acceptance_control_v1',
  '/rest/v1/orotitan_j0_c8_milestone_activation_v1',
  '/rest/v1/orotitan_j0_c8_reviews_v1',
  '/rest/v1/orotitan_j0_c8_decisions_v1',
]);

/** Only C8 readback traffic can leave this client, including a GET-only read RPC. */
export function getC8GateReadClient() {
  if (!isC8ReadOnlyRequested(process.env)) return getHostedShadowRecoveryClient();
  if (!isHostedC8ReadOnlyEnabled(process.env)) throw Error('C8_READ_ONLY_CONFIG_DENIED');
  return createClient(SHADOW_RESEARCH_ORIGIN,
    process.env.OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY!, {
      auth: {persistSession:false, autoRefreshToken:false, detectSessionInUrl:false},
      global: {fetch: async (input, init) => {
        const url = new URL(typeof input === 'string' ? input :
          input instanceof URL ? input.href : input.url);
        const method = (init?.method ?? (input instanceof Request ? input.method : 'GET')).toUpperCase();
        const readRpc = url.pathname === '/rest/v1/rpc/orotitan_j0_c8_read_gate_v1' &&
          method === 'GET' && /^J[0-8]$/.test(url.searchParams.get('p_milestone') ?? '') &&
          [...url.searchParams.keys()].every(key => key === 'p_milestone');
        if (!isHostedC8ReadOnlyEnabled(process.env) ||
            url.origin !== SHADOW_RESEARCH_ORIGIN || url.username || url.password ||
            !((readPaths.has(url.pathname) && ['GET','HEAD'].includes(method)) || readRpc)) {
          throw Error('C8_READ_ONLY_TRANSPORT_DENIED');
        }
        return fetch(input, {...init, redirect:'error'});
      }},
    });
}
