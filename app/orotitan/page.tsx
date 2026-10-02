import type { Metadata } from 'next';
import { CompanySearch } from '@/components/orotitan/company-search';

export const metadata: Metadata = {
  title: 'Equity Research',
};

export default function OroTitanEntryPage() {
  return (
    <div className="py-12 sm:py-20">
      <section className="mx-auto max-w-3xl text-center">
        <div className="text-xs font-semibold uppercase tracking-[0.24em] text-cyan-400">OroTitan Equity Research</div>
        <h1 className="mt-4 text-4xl font-semibold tracking-tight text-white sm:text-5xl">Quelle société voulez-vous analyser ?</h1>
        <p className="mx-auto mt-4 max-w-2xl text-sm leading-6 text-slate-400">
          Interface V1 de consultation. Le mock respecte la forme du contrat LOAD_RESULT gelé et n'effectue aucune lecture directe Supabase.
        </p>
      </section>
      <div className="mt-10"><CompanySearch /></div>
      <section className="mx-auto mt-12 max-w-3xl rounded-2xl border border-slate-800 bg-slate-900/40 p-5 text-left">
        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Périmètre V1</div>
        <div className="mt-3 grid gap-3 text-sm text-slate-400 sm:grid-cols-3">
          <div>Lecture uniquement</div>
          <div>LOAD_RESULT mocké</div>
          <div>Aucune mutation</div>
        </div>
      </section>
    </div>
  );
}
