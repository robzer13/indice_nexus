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
  return <div><div className="text-[11px] uppercase tracking-[0.08em] text-slate-500">{label}</div><div className="mt-1.5 text-sm text-slate-200">{children}</div></div>;
}

function TextBlock({ title, value }: { title: string; value: string | null }) {
  return <div className="rounded-2xl border border-slate-800 bg-slate-900/55 p-5"><h3 className="text-sm font-semibold text-slate-100">{title}</h3><p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-400">{value ?? 'Non renseigné'}</p></div>;
}

function MetricBox({ label, value, detail, accent = false }: { label: string; value: React.ReactNode; detail?: React.ReactNode; accent?: boolean }) {
  return <div className={`rounded-2xl border p-4 ${accent ? 'border-cyan-800 bg-cyan-950/20' : 'border-slate-800 bg-slate-900/55'}`}><div className="text-[11px] uppercase tracking-[0.08em] text-slate-500">{label}</div><div className="mt-2 text-2xl font-semibold text-white">{value}</div>{detail ? <div className="mt-2 text-xs leading-5 text-slate-500">{detail}</div> : null}</div>;
}

function fmtPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(digits)} %`;
}

function priceTime(value: string | null): string {
  if (!value) return 'Non disponible';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Date invalide';
  return date.toLocaleString('fr-FR', { dateStyle: 'short', timeStyle: 'short' });
}

function sourceLabel(value: string | null): string {
  if (!value) return 'Non disponible';
  if (value === 'TWELVE_DATA') return 'Twelve Data';
  if (value === 'YAHOO_FINANCE') return 'Yahoo Finance';
  if (value === 'CANONICAL_REFERENCE_PRICE') return 'Prix de référence publié';
  return value.replaceAll('_', ' ');
}

function taxonomyLabel(value: string | null | undefined): string {
  if (!value) return 'Non renseigné';
  const known: Record<string, string> = {
    US: 'États-Unis',
    HEALTH_CARE: 'Santé',
    MEDICAL_DEVICES: 'Dispositifs médicaux',
    CONSUMABLES_RAZOR_BLADE: 'Équipement + consommables récurrents',
    AFTERMARKET_SERVICE: 'Services après-vente',
    GLOBAL: 'Mondial',
  };
  return known[value] ?? value.replaceAll('_', ' ').toLowerCase().replace(/^./, (letter) => letter.toUpperCase());
}

function peaLabel(value: string | null | undefined): string {
  if (value === 'YES') return 'Éligible';
  if (value === 'NO') return 'Non éligible';
  if (value === 'UNKNOWN') return 'Inconnu';
  return 'Non renseigné';
}

export default async function CompanyPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const company = await getCompanyStateBySlug(slug);
  if (!company) notFound();

  const [history, priceHistory] = await Promise.all([
    getSnapshotHistory(company.id),
    getMarketPriceHistory(company.id, 180),
  ]);

  const distance = getDistanceO90(company.price, company.price_o90);
  const zone = getEntryZone(distance);
  const freshness = getFreshness(company.price_as_of);
  const priceProps = { currency: company.currency, quoteUnit: company.quote_unit, priceDecimals: company.price_decimals };
  const economicExposureRegions = company.economic_exposure_regions ?? [];

  const publishedOvs = company.snapshot_valuation_score ?? company.valuation_score;
  const liveOvs = company.live_valuation_score ?? publishedOvs;
  const publishedInvestment = company.snapshot_investment_score ?? company.investment_score;
  const liveInvestment = company.live_investment_score ?? publishedInvestment;

  const thresholds = [
    ['Potentiel OroTitan', company.price_o85],
    ['Rendement requis H', company.price_o90],
    ['Rendement fort', company.price_o92],
    ['Rendement exceptionnel', company.price_o95],
  ] as const;

  return <div className="space-y-9">
    <section className="grid gap-5 border-b border-slate-800 pb-7 lg:grid-cols-[1fr_auto] lg:items-end">
      <div>
        <div className="font-mono text-xs text-cyan-400">{company.ticker} · {company.exchange}</div>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">{company.name}</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">{company.business_description_short ?? 'Description du business non renseignée.'}</p>
        <div className="mt-4 flex flex-wrap gap-2"><EntryZoneBadge zone={zone}/><CompanyStatusBadge status={company.status}/></div>
      </div>
      <div className="min-w-[250px] rounded-2xl border border-cyan-900/70 bg-gradient-to-br from-cyan-950/35 to-slate-950 p-5">
        <div className="text-xs uppercase tracking-[0.12em] text-slate-500">Dernier cours disponible</div>
        <div className="mt-2 text-4xl font-semibold text-white"><PriceDisplay value={company.price} {...priceProps}/></div>
        <div className={`mt-3 text-xs ${freshness.stale ? 'text-amber-300' : 'text-emerald-300'}`}>{freshness.stale ? '● Cours ancien' : '● Cours à jour'}</div>
        <div className="mt-1 text-xs text-slate-500">{priceTime(company.price_as_of)} · {sourceLabel(company.price_source)}</div>
      </div>
    </section>

    <section className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold text-white">Lecture immédiate</h2>
        <p className="mt-1 max-w-4xl text-sm leading-6 text-slate-500">La qualité fondamentale reste celle de l’analyse publiée. La valorisation et le score d’investissement ci-dessous s’adaptent au dernier cours sans réécrire le snapshot certifié.</p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricBox label="OQS · qualité" value={<ScoreBadge score={company.business_quality_score}/>} detail="Fixe jusqu’à une nouvelle analyse fondamentale."/>
        <MetricBox label="OVS · valorisation live" value={<ScoreBadge score={liveOvs}/>} detail={<>Publié : {publishedOvs ?? '—'} · recalculé au cours affiché.</>} accent/>
        <MetricBox label="Score investissement live" value={<ScoreBadge score={liveInvestment}/>} detail={<>Publié : {publishedInvestment ?? '—'}.</>}/>
        <MetricBox label="Rendement annualisé live" value={<span className="font-mono">{fmtPct(company.live_primary_expected_return)}</span>} detail={<>Objectif H : {fmtPct(company.required_return_h, 1)}.</>}/>
        <MetricBox label="Écart au seuil H" value={<OroTitanDistance value={distance}/>} detail={company.price_o90 === null ? 'Seuil non calibré.' : <>Seuil H : <PriceDisplay value={company.price_o90} {...priceProps}/></>}/>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-950/55 px-4 py-3 text-xs leading-5 text-slate-500">
        <span className="font-semibold text-slate-300">Méthode live :</span> le résultat économique terminal certifié est conservé ; seul le prix d’entrée est remplacé par le dernier cours. Les caps de marge de sécurité, fiabilité de valorisation et permission de score du snapshot restent inchangés. {company.live_valuation_reason ? <span className="text-amber-300"> Limite : {company.live_valuation_reason}</span> : null}
      </div>
    </section>

    <section className="space-y-4">
      <div><h2 className="text-xl font-semibold text-white">Thèse en 30 secondes</h2><p className="mt-1 text-sm text-slate-500">Les trois idées à lire avant le détail du dossier.</p></div>
      <div className="grid gap-4 lg:grid-cols-3">
        <TextBlock title="Cas qualité" value={company.quality_case ?? null}/>
        <TextBlock title="Cas de valorisation" value={company.valuation_case ?? null}/>
        <TextBlock title="Risque principal" value={company.key_risk ?? null}/>
      </div>
    </section>

    <section className="space-y-5">
      <div><h2 className="text-xl font-semibold text-white">Valorisation et points d’entrée</h2><p className="mt-1 text-sm text-slate-500">Le marché évolue en continu ; les seuils restent ceux de la valorisation certifiée jusqu’à une nouvelle analyse.</p></div>
      <PriceHistoryChart points={priceHistory} thresholds={thresholds.map(([label, value]) => ({ label, value }))} {...priceProps}/>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{thresholds.map(([label, value]) => <div key={label} className={`rounded-xl border p-4 ${label === 'Rendement requis H' ? 'border-cyan-800 bg-cyan-950/20' : 'border-slate-800 bg-slate-900/50'}`}><div className="text-xs font-semibold text-slate-500">{label}</div><div className="mt-2 text-lg font-semibold text-slate-100">{value === null ? <span className="text-sm font-medium text-slate-500">Non disponible</span> : <PriceDisplay value={value} {...priceProps}/>}</div></div>)}</div>

      <div className="grid gap-3 rounded-2xl border border-slate-800 bg-slate-900/45 p-5 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="Cours de référence">{company.reference_price === null || company.reference_price === undefined ? '—' : <PriceDisplay value={company.reference_price} {...priceProps}/>}</Field>
        <Field label="Date de référence">{company.reference_price_date ?? '—'}</Field>
        <Field label="Variation depuis l’analyse">{fmtPct(company.price_change_vs_reference_pct)}</Field>
        <Field label="Rendement requis H">{fmtPct(company.required_return_h, 1)}</Field>
      </div>
    </section>

    <section className="space-y-4">
      <div><h2 className="text-xl font-semibold text-white">Qualité fondamentale</h2><p className="mt-1 text-sm text-slate-500">Les dimensions ci-dessous sont issues du snapshot publié et ne bougent pas avec le cours de Bourse.</p></div>
      <ScoreComponents value={company.score_components}/>
    </section>

    <section className="space-y-4">
      <div><h2 className="text-xl font-semibold text-white">Risques et prochaine action</h2><p className="mt-1 text-sm text-slate-500">Ce qui ferait changer la thèse et ce que le système recommande aujourd’hui.</p></div>
      <div className="grid gap-4 lg:grid-cols-2">
        <TextBlock title="Déclencheurs d’invalidation" value={company.invalidation}/>
        <TextBlock title="État du dossier / prochaine action" value={company.notes}/>
      </div>
    </section>

    <section className="rounded-2xl border border-slate-800 bg-slate-900/45 p-5">
      <div className="text-xs uppercase tracking-[0.15em] text-slate-500">Carte d’identité de la société</div>
      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="Pays émetteur">{taxonomyLabel(company.country)}</Field>
        <Field label="Secteur">{taxonomyLabel(company.sector)}</Field>
        <Field label="Industrie">{taxonomyLabel(company.industry_group)}</Field>
        <Field label="Business model">{taxonomyLabel(company.business_model_primary)}</Field>
        <Field label="Modèle secondaire">{taxonomyLabel(company.business_model_secondary)}</Field>
        <Field label="Expositions économiques">{economicExposureRegions.length > 0 ? economicExposureRegions.map(taxonomyLabel).join(', ') : 'Non renseignées'}</Field>
        <Field label="PEA">{peaLabel(company.pea_eligibility)}</Field>
        <Field label="Taxonomie">{company.taxonomy_version ?? 'Non renseignée'}</Field>
      </div>
    </section>

    <SnapshotComparison snapshots={history} {...priceProps}/>

    <section className="space-y-4">
      <div><h2 className="text-xl font-semibold text-white">Historique canonique et traçabilité</h2><p className="mt-1 text-sm text-slate-500">Les snapshots restent immuables. Le prix live est une couche de marché séparée.</p></div>
      <div className="grid gap-4 rounded-xl border border-slate-800 bg-slate-900/55 p-5 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="Autorité">{company.source_title ?? 'Non renseignée'}</Field>
        <Field label="Version">{company.model_version ?? 'Non renseignée'}</Field>
        <Field label="Date d’analyse">{company.analysis_date ?? 'Non renseignée'}</Field>
        <Field label="Statut OroTitan certifié">{company.quality_orotitan === null ? 'Non renseigné' : company.quality_orotitan ? 'Oui' : 'Non'}</Field>
      </div>
      <div className="rounded-xl border border-slate-800 bg-slate-900/55 p-2"><SnapshotHistory snapshots={history} currency={company.currency} quoteUnit={company.quote_unit} priceDecimals={company.price_decimals}/></div>
    </section>
  </div>;
}
