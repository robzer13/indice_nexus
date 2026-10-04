import type { Metadata } from 'next';
import { ScreenerTable } from '@/components/screener-table';
import { getCompanyStates } from '@/lib/data/companies';

export const metadata: Metadata = { title: 'Screener' };
export const dynamic = 'force-dynamic';

export default async function ScreenerPage() {
  const companies = await getCompanyStates();

  return (
    <div className="space-y-6">
      <header className="max-w-5xl">
        <div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Univers publié · OroTitan V2</div>
        <h1 className="mt-2 text-3xl font-semibold text-white sm:text-4xl">Screener actions</h1>
        <p className="mt-3 text-sm leading-6 text-slate-400 sm:text-base">
          Le cours est le dernier prix de marché disponible. L’<strong className="font-medium text-slate-200">OVS actuel</strong> et le
          <strong className="font-medium text-slate-200"> score actuel</strong> sont recalculés en mode prix-only lorsque le cours évolue.
          Les scores canoniques du snapshot restent conservés séparément pour l’audit.
        </p>
      </header>
      {companies.length === 0
        ? <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 text-slate-500">Aucune société publiée dans OroTitan.</div>
        : <ScreenerTable companies={companies}/>}
    </div>
  );
}
