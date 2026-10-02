import type { Metadata } from 'next';
import Link from 'next/link';
import { CompanySearch } from '@/components/orotitan/company-search';
import { VEOLIA_MOCK_DOSSIER } from '@/lib/orotitan-ui/mock';
import { buildRunHref, formatCutoff, runStatusLabel, stageLabel } from '@/lib/orotitan-ui/presentation';

export const metadata: Metadata = {
  title: 'Equity Research',
};

export default function OroTitanEntryPage() {
  const dossier = VEOLIA_MOCK_DOSSIER;
  const run = dossier.runSummaries.find((item) => item.runId === dossier.primaryRunId);

  return (
    <div className="mx-auto max-w-4xl py-10 sm:py-16">
      <section>
        <div className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-400">Equity Research</div>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
          Rechercher une société
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
          Ouvrez un dossier OroTitan existant par nom, ticker ou symbole de marché.
        </p>
      </section>

      <div className="mt-8">
        <CompanySearch />
      </div>

      {run ? (
        <section className="mt-12 border-t border-slate-800 pt-7">
          <div className="mb-3 flex items-center justify-between gap-4">
            <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Dossier récent</h2>
            <span className="text-xs text-slate-700">Environnement de démonstration</span>
          </div>
          <Link
            href={buildRunHref('/orotitan/' + dossier.identity.slug, run.runId)}
            className="group grid gap-3 border-y border-slate-800/90 py-4 transition hover:border-slate-700 sm:grid-cols-[minmax(0,1fr)_120px_120px_120px]"
          >
            <div>
              <div className="font-medium text-slate-100 group-hover:text-white">{dossier.identity.legalName}</div>
              <div className="mt-1 text-xs text-slate-600">{dossier.identity.ticker} · {dossier.identity.marketDataSymbol}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wide text-slate-700">Étape</div>
              <div className="mt-1 text-sm text-slate-300">{stageLabel(run.currentStage)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wide text-slate-700">État</div>
              <div className="mt-1 text-sm text-rose-300">{runStatusLabel(run.runStatus)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wide text-slate-700">Données</div>
              <div className="mt-1 text-sm text-slate-300">{formatCutoff(run.dataCutoff)}</div>
            </div>
          </Link>
        </section>
      ) : null}
    </div>
  );
}
