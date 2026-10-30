'use server';
import { redirect } from 'next/navigation';
import { requireAdminSession } from '@/lib/auth/admin-session';
import { executeShadowRecoveryCycle } from '@/lib/orotitan-equity/product/research-recovery-shadow-server';

export async function runResearchRecoveryAction(form: FormData): Promise<void> {
  await requireAdminSession();
  const key=form.get('source');
  if (typeof key !== 'string' ||
      (key !== 'asml-observation' && !/^[0-9a-f-]{36}$/i.test(key))) {
    redirect('/admin/research-recovery?error=invalid');
  }
  let status: string;
  try {
    const result=await executeShadowRecoveryCycle(key);
    status=result.report.status.toLowerCase();
  } catch {
    redirect('/admin/research-recovery?source='+encodeURIComponent(key)+'&error=failed');
  }
  redirect('/admin/research-recovery?source='+encodeURIComponent(key)+'&result='+encodeURIComponent(status));
}
