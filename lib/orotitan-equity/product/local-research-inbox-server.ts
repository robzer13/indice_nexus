import 'server-only';
import { join } from 'node:path';
import {
  enqueueLocalResearchRequest, listLocalResearchRequests,
  type ResearchInboxCompany,
} from './local-research-inbox-v1';

/**
 * Hard fail-closed pilot gate. Never writes any file in a production build,
 * regardless of environment variables set on Vercel.
 */
export function isLocalResearchInboxEnabled(): boolean {
  return process.env.NODE_ENV === 'development' &&
    process.env.OROTITAN_LOCAL_REQUESTS_ENABLED === '1';
}
function requirePilot(): string {
  if (!isLocalResearchInboxEnabled()) throw new Error('PRODUCT_PILOT_DISABLED');
  return join(process.cwd(), '.orotitan-local');
}
export async function loadLocalResearchInbox() {
  return listLocalResearchRequests(requirePilot());
}
export async function requestLocalResearchForCompany(
  companyId: string, companies: readonly ResearchInboxCompany[],
) {
  return enqueueLocalResearchRequest({
    directory: requirePilot(), companyId, companies,
  });
}
