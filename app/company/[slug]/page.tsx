import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { CompanyStatusBadge } from '@/components/company-status-badge';
import { EntryZoneBadge } from '@/components/entry-zone-badge';
import { OroTitanDistance } from '@/components/orotitan-distance';
import { PriceDisplay } from '@/components/price-display';
import { PriceHistoryChart } from '@/components/price-history-chart';
import { ScoreBadge } from '@/components/score-badge';
import { ScoreComponents } from '@/components/score-components';
import { SnapshotComparison } from '@/components/snapshot-comparison';
import { SnapshotHistory } from '@/components/snapshot-history';
import { getCompanyStateBySlug, getSnapshotHistory } from '@/lib/data/companies';
import { getMarketPriceHistory } from '@/lib/data/market-prices';
import { getDistanceO90 } from '@/lib/domain/distance';
import { getEntryZone } from '@/lib/domain/entry-zone';
import { getFreshness } from '@/lib/domain/freshness';

export const dynamic = 'force-dynamic';

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  return { title: slug };
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><div className="text-[11px] uppercase tracking-[0.12em] text-slate-500">{label}</div><div className="mt-1.5 text-sm text-slate-200">{children}</div></div>;
}

function TextBlock({ title, value }: { title: string; value: string | null }) {
  return <div className="rounded-2xl border border-slate-800/90 bg-slate-900/55 p-5"><h3 className="text-sm font-semibold text-slate-100">{title}</h3><p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-400">{value ?? 'Non renseigné'}</p></div>;
}

function pct(value: number | null, digits = 1) {
  if (value === null || !Number.isFinite(value)) return '—';
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)} %`;
}

function scoreValue(value: number | null) {
  return value === null ? '—' : value.toFixed(0);
}

function labelPea(value: string | null | undefined) {
  if (value === 'YES') return 'Éligible';
  if (value === 'NO') return 'Non éligible';
  if (value === 'UNKNOWN') return 'À confirmer';
  return 'Non renseigné';
}

export default async function CompanyPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const company = await getCompanyStateBySlug(slug);
  if (!company) notFound();

  const [history, priceHistory] = await Promise.all([
    getSnapshotHistory(company.id),
    getMarketPriceHistory(company.id, company.security_id ?? null, 180),
  ]);

  const distance = getDistanceO90(company.price, company.price_o90);
  const zone = getEntryZone(distance);
  const freshness = getFreshness(company.price_as_of);
  const priceProps = { currency: company.currency, quoteUnit: company.quote_unit, priceDecimals: company.price_decimals };
  const economicExposureRegions = company.economic_exposure_regions ?? [];
  const thresholds = [
    ['Potentiel OroTitan', company.price_o85],
    ['Rendement requis 10 %', company.price_o90],
    ['Rendement fort 12,5 %', company.price_o92],
    ['Rendement exceptionnel 15 %', company.price_o95],
  ] as const;

  const adaptiveOvs = company.market_valuation_score ?? company.valuation_score;
  const adaptiveInvestment = company.market_investment_score ?? company.investment_score;
  const marketMode = Boolean(company.market_score_is_live);

  return <div className="space-y-7 pb-10">
    <section className="overflow-hidden rounded-3xl border border-cyan-900/40 bg-[linear-gradient(145deg,rgba(9,31,48,.96),rgba(4,13,24,.92))] p-6 shadow-2xl shadow-cyan-950/10 sm:p-7">
      <div className="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
        <div className="max-w-3xl">
          <div className="font-mono text-xs font-semibold tracking-[0.18em] text-cyan-300">{company.ticker} · {company.exchange}</div>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">{company.name}</h1>
          <p className="mt-3 text-sm leading-6 text-slate-400">{company.business_description_short ?? company.sector ?? 'Analyse OroTitan publiée.'}</p>
          <div className="mt-4 flex flex-wrap gap-2"><CompanyStatusBadge status={company.status}/><EntryZoneBadge zone={zone}/>{company.pea_eligibility ? <span className="rounded-full border border-slate-700 bg-slate-950/45 px-2.5 py-1 text-xs text-slate-300">PEA · {labelPea(company.pea_eligibility)}</span> : null}</div>
        </div>

        <div className="min-w-[280px] rounded-2xl border border-cyan-800/35 bg-slate-950/45 p-5">
          <div className="flex items-center justify-between gap-4">
            <span className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Cours actuel</span>
            <span className={`rounded-full px-2 py-1 text-[11px] font-medium ${freshness.stale ? 'bg-amber-400/10 text-amber-200' : 'bg-emerald-400/10 text-emerald-200'}`}>{freshness.label}</span>
          </div>
          <div className="mt-2 text-4xl font-semibold tracking-tight text-white"><PriceDisplay value={company.price} {...priceProps}/></div>
          <div className="mt-2 flex items-center justify-between gap-4 text-xs">
            <span className="text-slate-500">vs prix de référence</span>
            <span className={(company.market_price_change_vs_reference_pct ?? null) !== null && (company.market_price_change_vs_reference_pct ?? 0) <= 0 ? 'font-mono text-emerald-300' : 'font-mono text-amber-200'}>{pct(company.market_price_change_vs_reference_pct ?? null)}</span>
          </div>
          <div className="mt-4 border-t border-slate-800 pt-3 text-xs leading-5 text-slate-500">
            MAJ : {company.price_as_of ? new Date(company.price_as_of).toLocaleString('fr-FR') : 'indisponible'}<br/>
            Source : {company.price_source ?? 'indisponible'}
          </div>
        </div>
      </div>
    </section>

    <section>
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <div><div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-400">Lecture OroTitan</div><h2 className="mt-1 text-2xl font-semibold text-white">Qualité figée, valorisation sensible au cours</h2></div>
        <div className="text-xs text-slate-500">{marketMode ? 'Projection marché recalculée avec I2' : 'Projection au prix de référence certifié'}</div>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-2xl border border-cyan-900/40 bg-cyan-950/15 p-5">
          <div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Qualité · OQS certifié</div>
          <div className="mt-3 flex items-end justify-between"><div className="text-4xl font-semibold text-cyan-100">{scoreValue(company.business_quality_score)}</div><span className="text-xs text-slate-500">/100</span></div>
          <p className="mt-3 text-sm leading-5 text-slate-400">La qualité fondamentale reste celle de la recherche certifiée. Elle ne bouge pas avec le cours.</p>
        </div>

        <div className="rounded-2xl border border-violet-900/45 bg-violet-950/15 p-5">
          <div className="flex items-center justify-between gap-3"><div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Valorisation · OVS marché</div>{marketMode ? <span className="rounded-full bg-violet-400/10 px-2 py-1 text-[10px] font-semibold text-violet-200">ADAPTATIF</span> : null}</div>
          <div className="mt-3 flex items-end justify-between"><div className="text-4xl font-semibold text-violet-100">{scoreValue(adaptiveOvs)}</div><span className="text-xs text-slate-500">/100</span></div>
          <div className="mt-3 flex justify-between text-xs text-slate-500"><span>OVS certifié au snapshot</span><span className="font-mono text-slate-300">{scoreValue(company.valuation_score)}</span></div>
          <p className="mt-3 text-sm leading-5 text-slate-400">Recalcul indicatif au cours courant avec les règles I2. Les caps de fiabilité et de certification restent ceux du snapshot.</p>
        </div>

        <div className="rounded-2xl border border-emerald-900/45 bg-emerald-950/10 p-5">
          <div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Score investissement marché</div>
          <div className="mt-3 flex items-end justify-between"><div className="text-4xl font-semibold text-emerald-100">{scoreValue(adaptiveInvestment)}</div><span className="text-xs text-slate-500">/100</span></div>
          <div className="mt-3 flex justify-between text-xs text-slate-500"><span>Score certifié au snapshot</span><span className="font-mono text-slate-300">{scoreValue(company.investment_score)}</span></div>
          <p className="mt-3 text-sm leading-5 text-slate-400">Combine l’OQS certifié et l’OVS marché, avec les mêmes caps méthodologiques OroTitan.</p>
        </div>
      </div>
    </section>

    <section className="grid gap-4 xl:grid-cols-[1.15fr_.85fr]">
      <div className="rounded-2xl border border-slate-800 bg-slate-900/55 p-5">
        <div className="flex items-center justify-between gap-3"><h2 className="text-lg font-semibold text-white">Rendement attendu au cours actuel</h2><span className="text-xs text-slate-500">horizon certifié</span></div>
        <div className="mt-5 grid gap-4 sm:grid-cols-3">
          <Field label="Scénario principal"><span className="text-lg font-semibold text-white">{pct(company.market_primary_expected_return ?? null)}</span></Field>
          <Field label="Normalisation"><span className="text-lg font-semibold text-white">{pct(company.market_normalization_expected_return ?? null)}</span></Field>
          <Field label="Distance au seuil 10 %"><OroTitanDistance value={distance}/></Field>
        </div>
        <div className="mt-5 border-t border-slate-800 pt-4 text-xs leading-5 text-slate-500">Le recalcul marché change uniquement les sorties dépendantes du prix. Il ne constitue pas une nouvelle certification fondamentale.</div>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900/55 p-5">
        <h2 className="text-lg font-semibold text-white">Repères de prix</h2>
        <div className="mt-4 space-y-3">{thresholds.slice(1).map(([label, value]) => <div key={label} className="flex items-center justify-between gap-4 border-b border-slate-800/70 pb-3 last:border-0 last:pb-0"><span className="text-sm text-slate-400">{label}</span><span className="font-mono text-sm font-semibold text-slate-100">{value === null ? 'Non disponible' : <PriceDisplay value={value} {...priceProps}/>}</span></div>)}</div>
      </div>
    </section>

    <section className="space-y-4">
      <div><h2 className="text-xl font-semibold text-white">Cours et zones d’entrée</h2><p className="mt-1 text-sm text-slate-500">Le marché évolue ; les seuils restent ceux de la valorisation certifiée jusqu’à une nouvelle analyse.</p></div>
      <PriceHistoryChart points={priceHistory} thresholds={thresholds.map(([label, value]) => ({ label, value }))} {...priceProps}/>
    </section>

    <section className="space-y-4">
      <div><div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-400">Thèse d’investissement</div><h2 className="mt-1 text-xl font-semibold text-white">Ce qu’il faut retenir</h2></div>
      <div className="grid gap-4 lg:grid-cols-3"><TextBlock title="Cas qualité" value={company.quality_case ?? null}/><TextBlock title="Cas valorisation" value={company.valuation_case ?? null}/><TextBlock title="Risque principal" value={company.key_risk ?? null}/></div>
    </section>

    <details className="group rounded-2xl border border-slate-800 bg-slate-900/45">
      <summary className="cursor-pointer list-none px-5 py-4 text-sm font-semibold text-slate-200">Voir la classification et les détails de recherche <span className="float-right text-slate-500 group-open:rotate-180">⌄</span></summary>
      <div className="border-t border-slate-800 p-5">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Pays émetteur">{company.country ?? 'Non renseigné'}</Field><Field label="Secteur">{company.sector ?? 'Non renseigné'}</Field><Field label="Industrie">{company.industry_group ?? 'Non renseignée'}</Field><Field label="Modèle économique">{company.business_model_primary ?? 'Non renseigné'}</Field><Field label="Modèle secondaire">{company.business_model_secondary ?? 'Non renseigné'}</Field><Field label="Expositions économiques">{economicExposureRegions.length > 0 ? economicExposureRegions.join(', ') : 'Non renseignées'}</Field><Field label="PEA">{labelPea(company.pea_eligibility)}</Field><Field label="Taxonomie">{company.taxonomy_version ?? 'Non renseignée'}</Field>
        </div>
        <div className="mt-6 grid gap-4 lg:grid-cols-2"><TextBlock title="Déclencheurs d’invalidation" value={company.invalidation}/><TextBlock title="État du dossier / prochaine action" value={company.notes}/></div>
        <div className="mt-5 grid gap-4 border-t border-slate-800 pt-5 sm:grid-cols-2 lg:grid-cols-4"><Field label="Autorité">{company.source_title ?? 'Non renseignée'}</Field><Field label="Version">{company.model_version ?? 'Non renseignée'}</Field><Field label="Date d’analyse">{company.analysis_date ?? 'Non renseignée'}</Field><Field label="Prix de référence">{(company.reference_price ?? null) === null ? '—' : <><PriceDisplay value={company.reference_price ?? null} {...priceProps}/> · {company.reference_price_date ?? 'date inconnue'}</>}</Field></div>
      </div>
    </details>

    <details className="group rounded-2xl border border-slate-800 bg-slate-900/45">
      <summary className="cursor-pointer list-none px-5 py-4 text-sm font-semibold text-slate-200">Voir les scores détaillés et l’historique canonique <span className="float-right text-slate-500 group-open:rotate-180">⌄</span></summary>
      <div className="space-y-6 border-t border-slate-800 p-5">
        <div><h3 className="text-base font-semibold text-white">Composants du score certifié</h3><p className="mt-1 text-xs text-slate-500">Sorties canoniques du snapshot publié.</p><div className="mt-3"><ScoreComponents value={company.score_components}/></div></div>
        <SnapshotComparison snapshots={history} {...priceProps}/>
        <div><h3 className="text-base font-semibold text-white">Historique des snapshots</h3><div className="mt-3 rounded-xl border border-slate-800 bg-slate-950/40 p-2"><SnapshotHistory snapshots={history} currency={company.currency} quoteUnit={company.quote_unit} priceDecimals={company.price_decimals}/></div></div>
      </div>
    </details>
  </div>;
}
