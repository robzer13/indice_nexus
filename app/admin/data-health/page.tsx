import Link from 'next/link';
import { redirect } from 'next/navigation';
import { AdminNav } from '@/components/admin/admin-nav';
import { MetricCard } from '@/components/metric-card';
import { isAdminAuthenticated } from '@/lib/auth/admin-session';
import { getCompanyStates } from '@/lib/data/companies';
import { summarizeDataHealth } from '@/lib/domain/data-health';

export const dynamic = 'force-dynamic';

export default async function DataHealthPage() {
  if (!(await isAdminAuthenticated())) redirect('/admin');
  const companies = await getCompanyStates();
  const health = summarizeDataHealth(companies);
  return <div className="space-y-6">
    <div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Data governance</div><h1 className="mt-2 text-3xl font-semibold text-white">Data Health</h1><p className="mt-2 text-sm text-slate-400">Contrôle déterministe des données manquantes, périmées ou non calibrées. Aucune donnée n’est inventée.</p></div>
    <AdminNav/>
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><MetricCard label="Sociétés" value={companies.length}/><MetricCard label="Propres" value={health.clean}/><MetricCard label="Avec erreur" value={health.withErrors}/><MetricCard label="Issues totales" value={health.totalIssues}/></div>
    <div className="space-y-3">{health.rows.map(({ company, issues }) => <div key={company.id} className="rounded-xl border border-slate-800 bg-slate-900/55 p-4"><div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"><div><div className="font-semibold text-white">{company.name} <span className="font-mono text-xs text-slate-500">{company.ticker}</span></div><div className="mt-2 flex flex-wrap gap-2">{issues.length === 0 ? <span className="rounded-full border border-emerald-900 bg-emerald-950/25 px-2 py-1 text-xs text-emerald-300">Aucune anomalie</span> : issues.map((issue) => <span key={issue.code} className={`rounded-full border px-2 py-1 text-xs ${issue.severity === 'error' ? 'border-rose-900 bg-rose-950/25 text-rose-300' : issue.severity === 'warning' ? 'border-amber-900 bg-amber-950/25 text-amber-300' : 'border-slate-700 bg-slate-950 text-slate-400'}`}>{issue.label}</span>)}</div></div><div className="flex gap-2"><Link href={`/company/${company.slug}`} className="rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300">Voir</Link><Link href={`/admin/companies/${company.slug}`} className="rounded-lg border border-cyan-900 px-3 py-2 text-xs text-cyan-300">Métadonnées</Link></div></div></div>)}</div>
  </div>;
}
