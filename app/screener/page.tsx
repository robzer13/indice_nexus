import type { Metadata } from 'next';
import { ScreenerTable } from '@/components/screener-table';
import { getCompanyStates } from '@/lib/data/companies';

export const metadata: Metadata = { title: 'Screener' };
export const dynamic = 'force-dynamic';

export default async function ScreenerPage() {
  const companies = await getCompanyStates();

  return <div className="space-y-6 pb-10">
    <div className="rounded-3xl border border-cyan-900/35 bg-[linear-gradient(145deg,rgba(8,29,45,.88),rgba(4,13,24,.75))] p-6 sm:p-7">
      <div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Univers canonique publié · V2</div>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">Screener OroTitan</h1>
      <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-400">Compare les sociétés sur leur <strong className="font-medium text-slate-200">qualité certifiée</strong> et leur <strong className="font-medium text-slate-200">valorisation au dernier cours disponible</strong>. L’OQS reste figé par la recherche ; l’OVS et le Score investissement marché sont recalculés de manière indicative avec la méthodologie I2, sans modifier le snapshot publié.</p>
      <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-500">
        <span className="rounded-full border border-slate-700 bg-slate-950/40 px-3 py-1.5">OQS · certifié</span>
        <span className="rounded-full border border-violet-800/50 bg-violet-950/20 px-3 py-1.5 text-violet-200">OVS · adaptatif au cours</span>
        <span className="rounded-full border border-emerald-800/50 bg-emerald-950/20 px-3 py-1.5 text-emerald-200">Score marché · adaptatif</span>
      </div>
    </div>

    {companies.length === 0
      ? <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 text-slate-500">Aucune société publiée dans OroTitan.</div>
      : <ScreenerTable companies={companies}/>}
  </div>;
}
