import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { CompanyStatusBadge } from '@/components/company-status-badge';
import { EntryZoneBadge } from '@/components/entry-zone-badge';
import { EngineBadge, ResearchFreshnessBadge } from '@/components/engine-provenance-badges';
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
  const company = await getCompanyStateBySlug(slug);
  return { title: company ? company.name : slug };
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><div className="text-[11px] uppercase tracking-[0.12em] text-slate-500">{label}</div><div className="mt-1 text-sm text-slate-200">{children}</div></div>;
}

function TextBlock({ title, value, tone = 'neutral' }: { title: string; value: string | null; tone?: 'neutral' | 'risk' }) {
  return (
    <div className={'rounded-2xl border p-5 ' + (tone === 'risk' ? 'border-rose-950/70 bg-rose-950/10' : 'border-slate-800 bg-slate-900/55')}>
      <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
      <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-400">{value ?? 'Non renseigné'}</p>
    </div>
  );
}

function MetricCard({ label, value, sub, emphasis = false }: { label: string; value: React.ReactNode; sub?: React.ReactNode; emphasis?: boolean }) {
  return (
    <div className={'rounded-2xl border p-4 ' + (emphasis ? 'border-cyan-800/60 bg-cyan-950/20' : 'border-slate-800 bg-slate-900/55')}>
      <div className="text-xs text-slate-500">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
      {sub ? <div className="mt-2 text-xs leading-5 text-slate-500">{sub}</div> : null}
    </div>
  );
}

function formatPct(value: number | null, digits = 1): string {
  if (value === null || !Number.isFinite(value)) return '—';
  return (value >= 0 ? '+' : '') + value.toFixed(digits) + ' %';
}

function priceSourceLabel(source: string | null): string {
  if (source === 'YAHOO_FINANCE' || source === 'YAHOO_FINANCE_LIVE') return source === 'YAHOO_FINANCE_LIVE' ? 'Yahoo Finance · direct' : 'Yahoo Finance';
  if (source === 'TWELVE_DATA') return 'Twelve Data';
  if (source === 'CANONICAL_REFERENCE_PRICE') return 'Prix de référence de l’analyse';
  return source ?? 'Source inconnue';
}

function frenchValue(value: string | null | undefined): string {
  if (!value) return 'Non renseigné';
  const labels: Record<string, string> = {
    YES: 'Oui',
    NO: 'Non',
    UNKNOWN: 'Inconnu',
    NONE: 'Aucune',
    LOW: 'Faible',
    MEDIUM: 'Moyenne',
    HIGH: 'Élevée',
    GLOBAL: 'Mondiale',
    HEALTH_CARE: 'Santé',
    MEDICAL_DEVICES: 'Dispositifs médicaux',
    CONSUMABLES_RAZOR_BLADE: 'Équipement + consommables récurrents',
    AFTERMARKET_SERVICE: 'Services après-vente',
  };
  return labels[value] ?? value.replaceAll('_', ' ');
}

export default async function CompanyPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const company = await getCompanyStateBySlug(slug);
  if (!company) notFound();

  const [history, priceHistory] = await Promise.all([
    getSnapshotHistory(company.id),
    getMarketPriceHistory(company.id, 180),
  ]);

  const displayedPriceHistory = [...priceHistory];
  const lastPersistedPrice = displayedPriceHistory.at(-1);
  if (
    company.price !== null &&
    company.price_as_of &&
    (!lastPersistedPrice || Date.parse(company.price_as_of) > Date.parse(lastPersistedPrice.as_of))
  ) {
    displayedPriceHistory.push({
      id: -1,
      company_id: company.id,
      price: company.price,
      as_of: company.price_as_of,
      source: company.price_source ?? 'CURRENT_DISPLAY_PRICE',
      raw: null,
      created_at: company.price_as_of,
    });
  }

  const distance = getDistanceO90(company.price, company.price_o90);
  const zone = getEntryZone(distance);
  const freshness = getFreshness(company.price_as_of);
  const priceProps = { currency: company.currency, quoteUnit: company.quote_unit, priceDecimals: company.price_decimals };
  const economicExposureRegions = company.economic_exposure_regions ?? [];
  const marketPriceAvailable = company.price_source !== null && company.price_source !== 'CANONICAL_REFERENCE_PRICE';
  const currentOvs = marketPriceAvailable ? (company.live_valuation_score ?? company.valuation_score) : company.valuation_score;
  const currentInvestmentScore = marketPriceAvailable ? (company.live_investment_score ?? company.investment_score) : company.investment_score;
  const currentExpectedReturn = marketPriceAvailable ? company.live_primary_expected_return_pct : company.canonical_primary_expected_return_pct;
  const dynamicChanged = marketPriceAvailable && (
    (company.live_valuation_score !== null && company.valuation_score !== null && Math.abs(company.live_valuation_score - company.valuation_score) > 0.05) ||
    (company.live_investment_score !== null && company.investment_score !== null && Math.abs(company.live_investment_score - company.investment_score) > 0.05)
  );

  const thresholds = [
    ['Potentiel OroTitan', company.price_o85],
    ['Rendement requis', company.price_o90],
    ['Rendement fort', company.price_o92],
    ['Rendement exceptionnel', company.price_o95],
  ] as const;

  return (
    <div className="space-y-7">
      <section className="rounded-3xl border border-slate-800 bg-gradient-to-br from-slate-900/90 via-slate-950/80 to-cyan-950/15 p-5 sm:p-7">
        <div className="flex flex-col gap-6 xl:flex-row xl:items-start xl:justify-between">
          <div className="max-w-3xl">
            <div className="font-mono text-xs font-semibold tracking-[0.12em] text-cyan-400">{company.ticker} · {company.exchange}</div>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">{company.name}</h1>
            <p className="mt-3 text-sm leading-6 text-slate-400">{frenchValue(company.country)} · {frenchValue(company.sector)} · {frenchValue(company.industry_group)}</p>
            {company.business_description_short ? <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-300">{company.business_description_short}</p> : null}
            <div className="mt-5 flex flex-wrap gap-2">
              <EntryZoneBadge zone={zone}/><CompanyStatusBadge status={company.status}/>
              <EngineBadge status={company.engine_status ?? 'UNKNOWN'} shortFingerprint={company.engine_short_fingerprint}/>
              <ResearchFreshnessBadge status={company.research_freshness_status ?? 'UNKNOWN'} ageDays={company.analysis_age_days}/>
            </div>
          </div>

          <div className="min-w-[260px] rounded-2xl border border-slate-700/80 bg-slate-950/70 p-5">
            <div className="text-xs uppercase tracking-[0.15em] text-slate-500">{marketPriceAvailable ? 'Cours de marché' : 'Prix de référence'}</div>
            <div className="mt-2 text-4xl font-semibold text-white"><PriceDisplay value={company.price} {...priceProps}/></div>
            <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
              {company.live_price_change_vs_reference_pct !== null && marketPriceAvailable
                ? <span className={company.live_price_change_vs_reference_pct <= 0 ? 'text-emerald-300' : 'text-amber-300'}>{formatPct(company.live_price_change_vs_reference_pct)} vs analyse</span>
                : null}
              <span className={freshness.stale ? 'text-amber-300' : 'text-slate-500'}>{freshness.label}</span>
            </div>
            <div className="mt-3 border-t border-slate-800 pt-3 text-xs leading-5 text-slate-500">
              <div>{company.price_as_of ? new Date(company.price_as_of).toLocaleString('fr-FR') : 'Date indisponible'}</div>
              <div>{priceSourceLabel(company.price_source)}</div>
            </div>
          </div>
        </div>
      </section>

      {!marketPriceAvailable ? (
        <div className="rounded-2xl border border-amber-900/60 bg-amber-950/20 px-4 py-3 text-sm leading-6 text-amber-200">
          Aucun cours de marché plus récent n’est encore disponible. Les scores affichés ci-dessous restent donc ceux du snapshot publié.
        </div>
      ) : null}

      <section>
        <div className="mb-3 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white">Décision au cours actuel</h2>
            <p className="mt-1 text-sm text-slate-500">La qualité reste figée par la recherche ; la valorisation s’adapte au dernier cours disponible.</p>
          </div>
          {dynamicChanged ? <div className="text-xs text-cyan-300">Recalcul prix-only actif</div> : null}
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <MetricCard label="Qualité · OQS" value={<ScoreBadge score={company.business_quality_score}/>} sub="Score fondamental canonique"/>
          <MetricCard label={marketPriceAvailable ? 'Valorisation actuelle · OVS' : 'Valorisation publiée · OVS'} value={<ScoreBadge score={currentOvs}/>} sub={dynamicChanged && company.valuation_score !== null ? <>Publié : {company.valuation_score.toFixed(0)}</> : 'Selon le snapshot courant'} emphasis/>
          <MetricCard label={marketPriceAvailable ? 'Score actuel' : 'Score publié'} value={<ScoreBadge score={currentInvestmentScore}/>} sub={dynamicChanged && company.investment_score !== null ? <>Publié : {company.investment_score.toFixed(0)}</> : 'Qualité + valorisation'}/>
          <MetricCard label="Rendement attendu" value={formatPct(currentExpectedReturn)} sub={marketPriceAvailable ? 'Recalculé au cours actuel' : 'Scénario principal publié'}/>
          <MetricCard label="Écart au seuil requis" value={<OroTitanDistance value={distance}/>} sub={company.price_o90 === null ? 'Seuil non disponible' : <>Seuil : <PriceDisplay value={company.price_o90} {...priceProps}/></>}/>
        </div>
        <div className="mt-3 rounded-xl border border-slate-800 bg-slate-950/50 px-4 py-3 text-xs leading-5 text-slate-500">
          Le recalcul « prix-only » conserve les fondamentaux, le niveau de marge de sécurité, la fiabilité de valorisation et les autres caps du snapshot publié. Il ne remplace pas une nouvelle certification de recherche.
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold text-white">Lecture rapide de la thèse</h2>
        <div className="grid gap-3 lg:grid-cols-3">
          <TextBlock title="Cas qualité" value={company.quality_case ?? null}/>
          <TextBlock title="Cas valorisation" value={company.valuation_case ?? null}/>
          <TextBlock title="Risque clé" value={company.key_risk ?? null} tone="risk"/>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
        <div className="space-y-3">
          <div>
            <h2 className="text-xl font-semibold text-white">Cours et seuils de valorisation</h2>
            <p className="mt-1 text-sm text-slate-500">Le cours de marché évolue ; l’échelle de prix reste celle du dernier snapshot certifié.</p>
          </div>
          <PriceHistoryChart points={displayedPriceHistory} thresholds={thresholds.map(([label, value]) => ({ label, value }))} {...priceProps}/>
        </div>
        <div>
          <h2 className="text-xl font-semibold text-white">Échelle de prix</h2>
          <div className="mt-3 space-y-2">
            {thresholds.filter(([label]) => label !== 'Potentiel OroTitan').map(([label, value], index) => (
              <div key={label} className={'flex items-center justify-between gap-4 rounded-xl border p-4 ' + (index === 0 ? 'border-cyan-800/60 bg-cyan-950/20' : 'border-slate-800 bg-slate-900/50')}>
                <div>
                  <div className="text-sm font-medium text-slate-200">{label}</div>
                  <div className="mt-1 text-xs text-slate-500">{index === 0 ? 'Objectif de rendement ' + (company.required_return_h_pct ?? 10) + ' %' : index === 1 ? 'Rendement fort' : 'Rendement exceptionnel'}</div>
                </div>
                <div className="text-lg font-semibold text-white">{value === null ? '—' : <PriceDisplay value={value} {...priceProps}/>}</div>
              </div>
            ))}
          </div>
          <div className="mt-3 rounded-xl border border-slate-800 bg-slate-900/40 p-4 text-sm text-slate-400">
            <div className="flex justify-between gap-4"><span>Marge de sécurité</span><span className="text-slate-200">{frenchValue(company.margin_of_safety)}</span></div>
            <div className="mt-2 flex justify-between gap-4"><span>Fiabilité de valorisation</span><span className="text-slate-200">{frenchValue(company.valuation_reliability)}</span></div>
          </div>
        </div>
      </section>

      <SnapshotComparison snapshots={history} {...priceProps}/>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
        <h2 className="text-xl font-semibold text-white">Profil de l’entreprise</h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Pays émetteur">{frenchValue(company.country)}</Field>
          <Field label="Secteur">{frenchValue(company.sector)}</Field>
          <Field label="Industrie">{frenchValue(company.industry_group)}</Field>
          <Field label="Modèle économique">{frenchValue(company.business_model_primary)}</Field>
          <Field label="Modèle secondaire">{frenchValue(company.business_model_secondary)}</Field>
          <Field label="Exposition économique">{economicExposureRegions.length > 0 ? economicExposureRegions.map(frenchValue).join(', ') : 'Non renseignée'}</Field>
          <Field label="Éligibilité PEA">{frenchValue(company.pea_eligibility)}</Field>
          <Field label="Taxonomie">{company.taxonomy_version ?? 'Non renseignée'}</Field>
        </div>
      </section>

      <section className="grid gap-3 lg:grid-cols-2">
        <TextBlock title="Facteurs d’invalidation" value={company.invalidation}/>
        <TextBlock title="État du dossier / prochaine action" value={company.notes}/>
      </section>

      <details className="rounded-2xl border border-slate-800 bg-slate-900/40">
        <summary className="cursor-pointer select-none px-5 py-4 text-sm font-semibold text-slate-200">Détails du scoring et audit</summary>
        <div className="space-y-6 border-t border-slate-800 p-5">
          <div>
            <h3 className="font-semibold text-white">Composants de score</h3>
            <p className="mt-1 text-sm text-slate-500">Valeurs canoniques publiées. L’OVS dynamique de marché est présenté séparément en haut de la fiche.</p>
            <div className="mt-3"><ScoreComponents value={company.score_components}/></div>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Field label="Source d’autorité">{company.source_title ?? 'Non renseignée'}</Field>
            <Field label="Version analytique">{company.model_version ?? 'Non renseignée'}</Field>
            <Field label="Moteur">{company.engine_status === 'CURRENT' ? 'Actuel' : company.engine_status === 'PREVIOUS' ? 'Précédent' : company.engine_status === 'LEGACY' ? 'Legacy' : 'Inconnu'} · process {company.process_version ?? '—'}</Field>
            <Field label="Fingerprint moteur">{company.engine_short_fingerprint ? company.engine_short_fingerprint + '…' : 'Non disponible'}</Field>
            <Field label="Data cutoff">{company.data_cutoff ?? 'Non renseigné'}</Field>
            <Field label="Âge recherche">{company.analysis_age_days === null ? 'Inconnu' : company.analysis_age_days + ' jours'}</Field>
            <Field label="Publication">{company.published_at ? new Date(company.published_at).toLocaleString('fr-FR') : 'Non renseignée'}</Field>
            <Field label="Prix de référence">{company.canonical_reference_price === null ? 'Non disponible' : <PriceDisplay value={company.canonical_reference_price} {...priceProps}/>}</Field>
          </div>
          <div>
            <h3 className="font-semibold text-white">Historique canonique</h3>
            <p className="mt-1 text-sm text-slate-500">Snapshots immuables du dossier ; les variations de cours n’écrasent pas l’analyse publiée.</p>
            <div className="mt-3 rounded-xl border border-slate-800 bg-slate-950/50 p-2"><SnapshotHistory snapshots={history} currency={company.currency} quoteUnit={company.quote_unit} priceDecimals={company.price_decimals}/></div>
          </div>
        </div>
      </details>
    </div>
  );
}
