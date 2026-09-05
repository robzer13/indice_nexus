import { redirect } from 'next/navigation';
import { AdminNav } from '@/components/admin/admin-nav';
import { CompanyForm } from '@/components/admin/company-form';
import { isAdminAuthenticated } from '@/lib/auth/admin-session';

export const dynamic = 'force-dynamic';
type SearchParams = Promise<Record<string, string | string[] | undefined>>;

export default async function NewCompanyPage({ searchParams }: { searchParams: SearchParams }) {
  if (!(await isAdminAuthenticated())) redirect('/admin');
  const query = await searchParams;
  const error = typeof query.error === 'string' ? query.error : null;
  return <div className="space-y-6"><div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Référentiel</div><h1 className="mt-2 text-3xl font-semibold text-white">Nouvelle société</h1></div><AdminNav/>{error ? <div className="rounded-lg border border-rose-900 bg-rose-950/30 p-3 text-sm text-rose-200">{error}</div> : null}<CompanyForm/></div>;
}
