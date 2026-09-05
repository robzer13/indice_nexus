import type { Metadata } from 'next';
import { ScreenerTable } from '@/components/screener-table';
import { getCompanyStates } from '@/lib/data/companies';

export const metadata: Metadata = { title: 'Screener' };
export const dynamic = 'force-dynamic';

export default async function ScreenerPage() {
  const companies = await getCompanyStates();
  return <div className="space-y-6"><div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Base analysée</div><h1 className="mt-2 text-3xl font-semibold text-white">Screener OroTitan</h1><p className="mt-2 text-sm text-slate-400">Tri initial par distance O90 décroissante : seuil atteint d’abord, puis sociétés les plus proches depuis le dessus. Les `NULL` restent non calibrés.</p></div>{companies.length === 0 ? <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 text-slate-500">Aucune société active dans la base.</div> : <ScreenerTable companies={companies}/>}</div>;
}
