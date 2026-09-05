import type { Metadata } from 'next';
import Link from 'next/link';
import { AdminNav } from '@/components/admin/admin-nav';
import { MetricCard } from '@/components/metric-card';
import { loginAction, logoutAction } from '@/app/admin/actions';
import { isAdminAuthenticated } from '@/lib/auth/admin-session';
import { getCompanyStates } from '@/lib/data/companies';
import { getRecentMarketSyncRuns } from '@/lib/data/market-prices';
import { summarizeDataHealth } from '@/lib/domain/data-health';

export const metadata: Metadata = { title: 'Admin' };
export const dynamic = 'force-dynamic';

type SearchParams = Promise<Record<string, string | string[] | undefined>>;

function messageValue(value: string | string[] | undefined): string | null {
  return typeof value === 'string' ? value : null;
}

export default async function AdminPage({ searchParams }: { searchParams: SearchParams }) {
  const query = await searchParams;
  const error = messageValue(query.error);
  const authenticated = await isAdminAuthenticated();

  if (!authenticated) {
    return <div className="mx-auto max-w-md rounded-xl border border-slate-800 bg-slate-900/70 p-6 shadow-panel">
      <div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Zone privée</div>
      <h1 className="mt-2 text-2xl font-semibold text-white">Administration OroTitan</h1>
      <p className="mt-2 text-sm leading-6 text-slate-400">Connexion serveur, session signée et cookie HTTP-only.</p>
      {error ? <div className="mt-4 rounded-lg border border-rose-900 bg-rose-950/30 p-3 text-sm text-rose-200">{error}</div> : null}
      <form action={loginAction} className="mt-5 space-y-4">
        <label className="block text-sm text-slate-400">Mot de passe<input name="password" type="password" required autoComplete="current-password" className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:border-cyan-500"/></label>
        <button className="w-full rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-400">Se connecter</button>
      </form>
    </div>;
  }

  const [companies, syncRuns] = await Promise.all([getCompanyStates(), getRecentMarketSyncRuns(1)]);
  const health = summarizeDataHealth(companies);
  const latestRun = syncRuns[0] ?? null;

  return <div className="space-y-6">
    <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Operational cockpit</div><h1 className="mt-2 text-3xl font-semibold text-white">Administration V1.1</h1><p className="mt-2 text-sm text-slate-400">Sociétés, analyses immutables, cours et qualité des données.</p></div>
      <form action={logoutAction}><button className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:bg-slate-900">Déconnexion</button></form>
    </div>
    <AdminNav/>
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <MetricCard label="Sociétés actives" value={companies.length} hint="base suivie"/>
      <MetricCard label="Data Health" value={health.totalIssues} hint="anomalies ou avertissements"/>
      <MetricCard label="Erreurs critiques" value={health.withErrors} hint="sociétés concernées"/>
      <MetricCard label="Dernière synchro" value={latestRun ? `${latestRun.inserted}/${latestRun.companies}` : '—'} hint={latestRun ? new Date(latestRun.finished_at).toLocaleString('fr-FR') : 'aucun journal V1.1'}/>
    </div>
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      <Quick href="/admin/companies/new" title="Ajouter une société" text="Créer les métadonnées sans inventer d’analyse."/>
      <Quick href="/admin/snapshots/new" title="Ajouter une analyse" text="Formulaire structuré ou import JSON avec prévalidation."/>
      <Quick href="/admin/prices" title="Actualiser les cours" text="Lancer Twelve Data et contrôler les erreurs société par société."/>
      <Quick href="/admin/data-health" title="Contrôler la base" text="Repérer cours périmés, O90 absents et métadonnées incomplètes."/>
    </div>
  </div>;
}

function Quick({ href, title, text }: { href: string; title: string; text: string }) {
  return <Link href={href} className="rounded-xl border border-slate-800 bg-slate-900/55 p-5 transition hover:border-cyan-900 hover:bg-slate-900"><h2 className="font-semibold text-white">{title}</h2><p className="mt-2 text-sm leading-6 text-slate-500">{text}</p></Link>;
}
