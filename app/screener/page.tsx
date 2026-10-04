import type { Metadata } from 'next';
import { ScreenerTable } from '@/components/screener-table';
import { getCompanyStates } from '@/lib/data/companies';

export const metadata: Metadata = { title: 'Screener' };
export const dynamic = 'force-dynamic';

export default async function ScreenerPage() {
  const companies = await getCompanyStates();
  return <div className="space-y-6">
    <section className="border-b border-slate-800 pb-6">
      <div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Screener dynamique · univers publié</div>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">Screener OroTitan</h1>
      <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-400">Le cours est le dernier point de marché disponible. L’OQS reste figé par la recherche certifiée ; l’OVS et le score d’investissement live se recalculent au cours affiché, sans modifier le snapshot publié.</p>
    </section>
    {companies.length === 0 ? <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 text-slate-500">Aucune société publiée dans OroTitan.</div> : <ScreenerTable companies={companies}/>}
  </div>;
}
