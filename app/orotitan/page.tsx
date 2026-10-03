import type { Metadata } from 'next';
import Link from 'next/link';
import { CompanySearch } from '@/components/orotitan/company-search';
import { getUiDossierShell } from '@/lib/orotitan-ui/server-data';

export const metadata: Metadata = {
  title: 'Equity Research',
};

export default async function OroTitanEntryPage() {
  const dossier = await getUiDossierShell('veolia');

  return (
    <div className="mx-auto max-w-5xl py-10 sm:py-14">
      <section className="text-center">
        <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-cyan-400">OroTitan</div>
        <h1 className="mt-4 text-4xl font-semibold tracking-tight text-white sm:text-5xl">Equity Research</h1>
        <p className="mx-auto mt-4 max-w-xl text-sm leading-6 text-slate-400">
          Recherchez un dossier existant par nom, ticker ou symbole de marché.
        </p>
      </section>

      <div className="mx-auto mt-8 max-w-3xl">
        <CompanySearch />
      </div>

      {dossier ? (
        <section className="mt-11 sm:mt-12">
          <div className="mb-3 flex items-center justify-between gap-4">
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Dossier disponible</h2>
            <span className="text-[10px] uppercase tracking-[0.15em] text-slate-600">Données réelles · lecture seule</span>
          </div>

          <Link
            href={'/orotitan/' + dossier.identity.slug}
            className="group grid gap-4 rounded-xl border border-[rgba(123,173,214,.16)] bg-[rgba(8,22,36,.72)] px-5 py-5 shadow-[0_18px_48px_rgba(0,0,0,.16)] transition hover:border-cyan-400/25 hover:bg-[rgba(10,26,42,.78)] md:grid-cols-[minmax(0,1fr)_160px_160px]"
          >
            <div>
              <div className="text-base font-medium text-slate-100 group-hover:text-white">{dossier.identity.legalName}</div>
              <div className="mt-1 text-xs text-slate-500">{dossier.identity.ticker} · {dossier.identity.marketDataSymbol}</div>
            </div>
            <Meta label="Analyses actives" value={String(dossier.runSummaries.length)} />
            <Meta label="Action" value="Choisir une analyse →" accent />
          </Link>
        </section>
      ) : null}
    </div>
  );
}

function Meta({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">{label}</div>
      <div className={'mt-1.5 text-sm ' + (accent ? 'text-cyan-300' : 'text-slate-300')}>{value}</div>
    </div>
  );
}
