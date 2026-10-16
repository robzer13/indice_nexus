import { redirect } from 'next/navigation';
import { AdminNav } from '@/components/admin/admin-nav';
import { PriceDisplay } from '@/components/price-display';
import { RefreshPricesButton } from '@/components/admin/refresh-prices-button';
import { refreshPricesAction } from '@/app/admin/actions';
import { isAdminAuthenticated } from '@/lib/auth/admin-session';
import { getCompanyStates } from '@/lib/data/companies';
import { getRecentMarketSyncRuns } from '@/lib/data/market-prices';
import { getFreshness } from '@/lib/domain/freshness';
import { summarizeMarketSyncHealth, type SyncSourceHealth } from '@/lib/domain/market-sync-health';

export const dynamic = 'force-dynamic';
type SearchParams = Promise<Record<string, string | string[] | undefined>>;

function formatRunDate(value: string): string {
  const time = Date.parse(value);
  return Number.isFinite(time)
    ? new Date(time).toLocaleString('fr-FR', { timeZone: 'Europe/Paris' }) + ' (Paris)'
    : 'Horodatage invalide';
}

function SyncHealthPanel({ health, sampledRuns }: { health: SyncSourceHealth; sampledRuns: number }) {
  const kind = health.source === 'CRON' ? 'Automatique — CRON' : 'Manuelle — ADMIN';
  const labels = {
    NO_RUN: 'Aucune exécution retrouvée',
    INVALID_RUN: 'Journal incohérent',
    SUCCESS: 'Exécution sans échec',
    PARTIAL: 'Succès partiel',
    FAILED: 'Toutes les cotations ont échoué',
  };
  const statusColor = health.status === 'SUCCESS' ? 'text-emerald-300'
    : health.status === 'NO_RUN' || health.status === 'INVALID_RUN' ? 'text-slate-300'
    : 'text-amber-300';
  const run = health.run;

  return <section className="rounded-xl border border-slate-800 bg-slate-900/65 p-5">
    <h2 className="text-lg font-semibold text-white">{kind}</h2>
    <p className={'mt-2 text-sm font-semibold ' + statusColor}>{labels[health.status]}</p>
    {!run
      ? <p className="mt-3 text-sm text-slate-400">Aucun journal {health.source} parmi les {sampledRuns} dernières exécutions consultées. État inconnu, pas de succès présumé.</p>
      : <>
        <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
          <div><dt className="text-slate-500">Début</dt><dd className="text-slate-200">{formatRunDate(run.started_at)}</dd></div>
          <div><dt className="text-slate-500">Fin</dt><dd className="text-slate-200">{formatRunDate(run.finished_at)}</dd></div>
          <div><dt className="text-slate-500">Durée</dt><dd className="text-slate-200">{health.durationSeconds === null ? '—' : health.durationSeconds + ' s'}</dd></div>
          <div><dt className="text-slate-500">Sociétés</dt><dd className="text-slate-200">{run.companies}</dd></div>
          <div><dt className="text-slate-500">Insérées / échecs</dt><dd className="text-slate-200">{run.inserted} / {run.failed}</dd></div>
          <div><dt className="text-slate-500">Taux de réussite</dt><dd className="text-slate-200">{health.successRatePct === null ? 'Non calculable' : health.successRatePct.toLocaleString('fr-FR') + ' %'}</dd></div>
        </dl>
        {health.status === 'INVALID_RUN'
          ? <p className="mt-3 text-sm text-amber-200">Compteurs, détails ou horodatages incohérents : aucun taux fiable ne peut être calculé.</p>
          : null}
        {health.errors.length > 0
          ? <div className="mt-4 space-y-2 border-t border-slate-800 pt-3">
            <h3 className="text-sm font-semibold text-slate-200">Erreurs regroupées ({run.failed} échecs déclarés)</h3>
            {health.errors.map(group => <details key={group.message} className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2">
              <summary className="cursor-pointer text-xs text-slate-300">
                <span className="font-semibold text-amber-300">{group.count} ×</span>{' '}
                <span className="break-all">{group.message}</span>
              </summary>
              <p className="mt-2 break-words text-xs text-slate-500">{group.companies.length > 0 ? group.companies.join(', ') : 'Identités des sociétés indisponibles'}</p>
            </details>)}
          </div>
          : run.failed > 0
            ? <p className="mt-4 text-sm text-amber-200">Détails des échecs indisponibles ou invalides.</p>
            : null}
      </>}
  </section>;
}

export default async function AdminPricesPage({ searchParams }: { searchParams: SearchParams }) {
  if (!(await isAdminAuthenticated())) redirect('/admin');
  const [companies, runs, query] = await Promise.all([getCompanyStates(), getRecentMarketSyncRuns(40), searchParams]);
  const error = typeof query.error === 'string' ? query.error : null;
  const success = typeof query.success === 'string' ? query.success : null;
  const warning = typeof query.warning === 'string' ? query.warning : null;
  const syncHealth = summarizeMarketSyncHealth(runs);
  const visibleRuns = runs.slice(0, 10);

  return <div className="space-y-6">
    <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between"><div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Market data</div><h1 className="mt-2 text-3xl font-semibold text-white">Cours & synchronisations</h1><p className="mt-2 text-sm text-slate-400">Collecte Yahoo Finance en priorité, Twelve Data en repli optionnel. Les cotations non admissibles sont rejetées ; les points conservés restent append-only.</p></div><form action={refreshPricesAction}><RefreshPricesButton/></form></div>
    <AdminNav/>
    {error ? <div className="rounded-lg border border-rose-900 bg-rose-950/30 p-3 text-sm text-rose-200">{error}</div> : null}
    {success ? <div className="rounded-lg border border-emerald-900 bg-emerald-950/25 p-3 text-sm text-emerald-200">{success}</div> : null}
    {warning ? <div className="rounded-lg border border-amber-900 bg-amber-950/25 p-3 text-sm text-amber-200">{warning}</div> : null}

    <div className="grid gap-4 lg:grid-cols-2"><SyncHealthPanel health={syncHealth.cron} sampledRuns={runs.length}/><SyncHealthPanel health={syncHealth.admin} sampledRuns={runs.length}/></div>

    <div className="overflow-x-auto rounded-xl border border-slate-800"><table className="min-w-[900px] w-full text-left text-sm"><thead className="border-b border-slate-800 bg-slate-900/80 text-xs uppercase text-slate-500"><tr><th className="px-4 py-3">Société</th><th className="px-4 py-3">Symbole</th><th className="px-4 py-3">Cours</th><th className="px-4 py-3">Dernier point</th><th className="px-4 py-3">Fraîcheur</th><th className="px-4 py-3">Source</th></tr></thead><tbody className="divide-y divide-slate-800 bg-slate-950/50">{companies.map((company) => { const freshness = getFreshness(company.price_as_of); return <tr key={company.id}><td className="px-4 py-4 font-medium text-white">{company.name}</td><td className="px-4 py-4 font-mono text-xs text-slate-400">{company.market_data_symbol ?? '—'}</td><td className="px-4 py-4 text-slate-200"><PriceDisplay value={company.price} currency={company.currency} quoteUnit={company.quote_unit} priceDecimals={company.price_decimals}/></td><td className="px-4 py-4 text-xs text-slate-400">{company.price_as_of ? new Date(company.price_as_of).toLocaleString('fr-FR') : '—'}</td><td className={`px-4 py-4 ${freshness.stale ? 'text-amber-300' : 'text-emerald-300'}`}>{freshness.label}</td><td className="px-4 py-4 text-slate-500">{company.price_source ?? '—'}</td></tr>; })}</tbody></table></div>

    <section className="space-y-3"><h2 className="text-xl font-semibold text-white">Journal des synchronisations</h2>{runs.length === 0 ? <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 text-sm text-slate-500">Aucun journal V1.1. Exécute la migration puis lance une actualisation.</div> : <div className="space-y-3">{visibleRuns.map((run) => <details key={run.id} className="rounded-xl border border-slate-800 bg-slate-900/50 p-4"><summary className="cursor-pointer text-sm text-slate-200"><span className={run.failed === 0 ? 'text-emerald-300' : 'text-amber-300'}>{run.trigger_source}</span> · {new Date(run.finished_at).toLocaleString('fr-FR')} · {run.inserted}/{run.companies} insérés · {run.failed} échec(s)</summary><pre className="mt-3 overflow-x-auto rounded-lg bg-slate-950 p-3 text-xs text-slate-400">{JSON.stringify(run.results, null, 2)}</pre></details>)}</div>}</section>
  </div>;
}
