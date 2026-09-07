import type { Metadata } from 'next';
import { ScreenerTable } from '@/components/screener-table';
import { Panel } from '@/components/ui/panel';
import { getCompanyStates } from '@/lib/data/companies';

export const metadata: Metadata = { title: 'Screener' };
export const dynamic = 'force-dynamic';

export default async function ScreenerPage() {
  const companies = await getCompanyStates();
  return <div className="space-y-7"><header className="flex flex-col gap-5 border-b border-slate-700/60 pb-7 sm:flex-row sm:items-end sm:justify-between"><div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-state-accent">Base analysée · V1.2</div><h1 className="mt-2 text-3xl font-semibold tracking-tight text-ink-primary sm:text-4xl">Screener OroTitan</h1><p className="mt-3 max-w-3xl text-sm leading-6 text-ink-secondary">Identifiez les sociétés les plus proches d’un point d’entrée OroTitan, avec une lecture disciplinée des cours, seuils et scores.</p></div><div className="shrink-0 rounded-control border border-slate-700/60 bg-cockpit-panel/70 px-4 py-3 sm:text-right"><div className="font-mono text-xl font-semibold tabular-nums text-ink-primary">{companies.length}</div><div className="mt-1 text-xs uppercase tracking-[0.16em] text-ink-muted">sociétés actives</div></div></header>{companies.length === 0 ? <Panel className="p-6 text-sm text-ink-muted">Aucune société active dans la base.</Panel> : <ScreenerTable companies={companies}/>}</div>;
}
