import Link from 'next/link';
import { redirect } from 'next/navigation';
import { AdminNav } from '@/components/admin/admin-nav';
import { isAdminAuthenticated } from '@/lib/auth/admin-session';
import { getAllCompanies } from '@/lib/data/company-admin';

export const dynamic = 'force-dynamic';

type SearchParams = Promise<Record<string, string | string[] | undefined>>;

export default async function CompaniesAdminPage({ searchParams }: { searchParams: SearchParams }) {
  if (!(await isAdminAuthenticated())) redirect('/admin');
  const [companies, query] = await Promise.all([getAllCompanies(), searchParams]);
  const success = typeof query.success === 'string' ? query.success : null;

  return <div className="space-y-6">
    <div className="flex items-end justify-between gap-4"><div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Référentiel</div><h1 className="mt-2 text-3xl font-semibold text-white">Sociétés</h1><p className="mt-2 text-sm text-slate-400">Métadonnées administrables. Aucun snapshot ni cours historique n’est modifié.</p></div><Link href="/admin/companies/new" className="rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-slate-950">Nouvelle société</Link></div>
    <AdminNav/>
    {success ? <div className="rounded-lg border border-emerald-900 bg-emerald-950/25 p-3 text-sm text-emerald-200">{success}</div> : null}
    <div className="overflow-x-auto rounded-xl border border-slate-800">
      <table className="min-w-[900px] w-full text-left text-sm"><thead className="border-b border-slate-800 bg-slate-900/80 text-xs uppercase tracking-wide text-slate-500"><tr><th className="px-4 py-3">Société</th><th className="px-4 py-3">Ticker</th><th className="px-4 py-3">Marché</th><th className="px-4 py-3">Devise/unité</th><th className="px-4 py-3">Symbole data</th><th className="px-4 py-3">État</th><th className="px-4 py-3"></th></tr></thead>
      <tbody className="divide-y divide-slate-800 bg-slate-950/50">{companies.map((company) => <tr key={company.id}><td className="px-4 py-4 font-medium text-white">{company.name}</td><td className="px-4 py-4 font-mono text-slate-400">{company.ticker}</td><td className="px-4 py-4 text-slate-400">{company.exchange}</td><td className="px-4 py-4 text-slate-400">{company.currency} / {company.quote_unit}</td><td className="px-4 py-4 font-mono text-xs text-slate-400">{company.market_data_symbol ?? '—'}</td><td className="px-4 py-4"><span className={company.active ? 'text-emerald-300' : 'text-slate-600'}>{company.active ? 'Active' : 'Inactive'}</span></td><td className="px-4 py-4 text-right"><Link href={`/admin/companies/${company.slug}`} className="text-cyan-300 hover:text-cyan-200">Éditer</Link></td></tr>)}</tbody></table>
    </div>
  </div>;
}
