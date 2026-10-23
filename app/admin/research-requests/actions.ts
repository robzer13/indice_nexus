'use server';

import { redirect } from 'next/navigation';
import { requireAdminSession } from '@/lib/auth/admin-session';
import {
  researchInboxPilotMode, requestPilotResearch,
} from '@/lib/orotitan-equity/product/research-inbox-pilot-server';

/** A LOCAL intake receipt only; no Research runner or Registry mutation. */
export async function requestResearchPilotAction(form: FormData): Promise<void> {
  await requireAdminSession();
  if (researchInboxPilotMode() === 'disabled') redirect('/admin');

  const companyId = form.get('company_id');
  if (typeof companyId !== 'string') {
    redirect('/admin/research-requests?error=missing-company');
  }
  let outcome: Awaited<ReturnType<typeof requestPilotResearch>>;
  try {
    outcome = await requestPilotResearch(companyId);
  } catch {
    // Do not expose filesystem/DB details to browser query parameters.
    redirect('/admin/research-requests?error=intake-failed');
  }
  const params = new URLSearchParams({
    request: outcome.request.requestId,
    result: outcome.duplicate ? 'existing' : 'created',
  });
  redirect('/admin/research-requests?' + params.toString());
}
