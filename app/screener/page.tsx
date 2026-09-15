import type { Metadata } from 'next';
import { ScreenerTable } from '@/components/screener-table';
import { getCompanyStates } from '@/lib/data/companies';

export const metadata: Metadata = { title: 'Screener' };
export const dynamic = 'force-dynamic';

export default async function ScreenerPage() {
  const companies = await getCompanyStates();
  return <div className="space-y-6"><div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Univers canonique publié · V1</div><h1 className="mt-2 text-3xl font-semibold text-white">Screener OroTitan</h1><p className="mt-2 max-w-4xl text-sm leading-6 text-slate-400">Seuls les dossiers disposant d’un <code>current_snapshot_id</code> canonique sont visibles. Les scores OQS, OVS et Investment ainsi que les seuils de prix proviennent du snapshot publié courant.</p></div>{companies.length === 0 ? <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 text-slate-500">Aucune société publiée dans OroTitan.</div> : <ScreenerTable companies={companies}/>}</div>;
}
