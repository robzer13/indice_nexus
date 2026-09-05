import { notFound, redirect } from 'next/navigation';
import { AdminNav } from '@/components/admin/admin-nav';
import { CompanyForm } from '@/components/admin/company-form';
import { isAdminAuthenticated } from '@/lib/auth/admin-session';
import { getCompanyRecordBySlug } from '@/lib/data/company-admin';

export const dynamic = 'force-dynamic';
type SearchParams = Promise<Record<string, string | string[] | undefined>>;

export default async function EditCompanyPage({ params, searchParams }: { params: Promise<{ slug: string }>; searchParams: SearchParams }) {
  if (!(await isAdminAuthenticated())) redirect('/admin');
  const { slug } = await params;
  const [company, query] = await Promise.all([getCompanyRecordBySlug(slug), searchParams]);
  if (!company) notFound();
  const error = typeof query.error === 'string' ? query.error : null;
  const success = typeof query.success === 'string' ? query.success : null;
  return <div className="space-y-6"><div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Référentiel</div><h1 className="mt-2 text-3xl font-semibold text-white">{company.name}</h1><p className="mt-2 text-sm text-slate-500">Modification des métadonnées uniquement.</p></div><AdminNav/>{error ? <div className="rounded-lg border border-rose-900 bg-rose-950/30 p-3 text-sm text-rose-200">{error}</div> : null}{success ? <div className="rounded-lg border border-emerald-900 bg-emerald-950/25 p-3 text-sm text-emerald-200">{success}</div> : null}<CompanyForm company={company}/></div>;
}
