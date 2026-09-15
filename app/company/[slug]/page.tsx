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
  return <div><div className="text-xs uppercase tracking-wide text-slate-500">{label}</div><div className="mt-1 text-sm text-slate-200">{children}</div></div>;
}

function TextBlock({ title, value }: { title: string; value: string | null }) {
  return <div className="rounded-xl border border-slate-800 bg-slate-900/55 p-5"><h3 className="text-sm font-semibold text-slate-200">{title}</h3><p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-400">{value ?? 'Non renseigné'}</p></div>;
}

export default async function CompanyPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const company = await getCompanyStateBySlug(slug);
  if (!company) notFound();
  const [history, priceHistory] = await Promise.all([getSnapshotHistory(company.id), getMarketPriceHistory(company.id, 180)]);
  const distance = getDistanceO90(company.price, company.price_o90);
  const zone = getEntryZone(distance);
  const freshness = getFreshness(company.price_as_of);
  const priceProps = { currency: company.currency, quoteUnit: company.quote_unit, priceDecimals: company.price_decimals };
  const economicExposureRegions = company.economic_exposure_regions ?? [];
  const thresholds = [
    ['Potentiel OroTitan', company.price_o85],
    ['Rendement requis H', company.price_o90],
    ['Rendement fort', company.price_o92],
    ['Rendement exceptionnel', company.price_o95],
  ] as const;

  return <div className="space-y-8">
    <section className="flex flex-col gap-5 border-b border-slate-800 pb-7 lg:flex-row lg:items-end lg:justify-between"><div><div className="font-mono text-xs text-cyan-400">{company.ticker} · {company.exchange}</div><h1 className="mt-2 text-3xl font-semibold text-white sm:text-4xl">{company.name}</h1><p className="mt-2 text-sm text-slate-500">{company.country ?? 'Pays non renseigné'} · {company.sector ?? 'Secteur non renseigné'} · snapshot canonique publié</p></div><div className="flex flex-wrap gap-2"><EntryZoneBadge zone={zone}/><CompanyStatusBadge status={company.status}/></div></section>

    <section className="rounded-xl border border-slate-800 bg-slate-900/55 p-5"><div className="text-xs uppercase tracking-[0.15em] text-slate-500">Classification canonique</div><div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><Field label="Pays émetteur">{company.country ?? 'Non renseigné'}</Field><Field label="Secteur">{company.sector ?? 'Non renseigné'}</Field><Field label="Industrie">{company.industry_group ?? 'Non renseignée'}</Field><Field label="Business model">{company.business_model_primary ?? 'Non renseigné'}</Field><Field label="Business model secondaire">{company.business_model_secondary ?? 'Non renseigné'}</Field><Field label="Expositions économiques">{economicExposureRegions.length > 0 ? economicExposureRegions.join(', ') : 'Non renseignées'}</Field><Field label="PEA">{company.pea_eligibility ?? 'Non renseigné'}</Field><Field label="Taxonomie">{company.taxonomy_version ?? 'V1 historique / non applicable'}</Field></div><div className="mt-5 border-t border-slate-800 pt-4"><Field label="Description courte du business">{company.business_description_short ?? 'Non renseignée pour ce snapshot'}</Field></div></section>

    <section className="grid gap-4 lg:grid-cols-[1.4fr_1fr]"><div className="rounded-xl border border-slate-800 bg-slate-900/65 p-5"><div className="text-xs uppercase tracking-[0.15em] text-slate-500">État canonique actuel</div><div className="mt-4 flex flex-wrap items-end gap-6"><div><div className="text-xs text-slate-500">Cours</div><div className="mt-1 text-3xl font-semibold text-white"><PriceDisplay value={company.price} {...priceProps}/></div></div><div><div className="text-xs text-slate-500">Seuil rendement H</div><div className="mt-1 text-xl text-slate-200">{company.price_o90 === null ? 'Non calibré' : <PriceDisplay value={company.price_o90} {...priceProps}/>}</div></div><div><div className="text-xs text-slate-500">Distance seuil H</div><div className="mt-1"><OroTitanDistance value={distance}/></div></div><div><div className="text-xs text-slate-500">Investment Score</div><div className="mt-1"><ScoreBadge score={company.investment_score}/></div></div></div><div className="mt-5 grid gap-4 border-t border-slate-800 pt-4 sm:grid-cols-2 lg:grid-cols-4"><Field label="Date du cours">{company.price_as_of ? new Date(company.price_as_of).toLocaleString('fr-FR') : 'Non disponible'}</Field><Field label="Source du cours">{company.price_source ?? 'Non disponible'}</Field><Field label="Fraîcheur"><span className={freshness.stale ? 'text-amber-200' : 'text-emerald-200'}>{freshness.label}</span></Field><Field label="OroTitan terminal">{company.quality_orotitan === null ? 'Non renseigné' : company.quality_orotitan ? 'YES' : 'NO'}</Field></div></div><div className="rounded-xl border border-slate-800 bg-slate-900/65 p-5"><div className="text-xs uppercase tracking-[0.15em] text-slate-500">Scores canoniques</div><div className="mt-4 grid grid-cols-2 gap-4"><Field label="OQS">{company.business_quality_score ?? '—'}</Field><Field label="OVS">{company.valuation_score ?? '—'}</Field><Field label="Investment">{company.investment_score ?? '—'}</Field><Field label="Statut">{company.status ?? '—'}</Field></div></div></section>

    <section className="space-y-4"><div><h2 className="text-xl font-semibold text-white">Historique de marché</h2><p className="mt-1 text-sm text-slate-500">Les cours de marché restent séparés de l’analyse. Les seuils affichés sont issus du snapshot canonique courant.</p></div><PriceHistoryChart points={priceHistory} thresholds={thresholds.map(([label, value]) => ({ label, value }))} {...priceProps}/></section>

    <section className="space-y-4"><div><h2 className="text-xl font-semibold text-white">Price ladder canonique</h2><p className="mt-1 text-sm text-slate-500">Les seuils proviennent directement de la valorisation publiée. Un seuil absent reste explicitement non disponible.</p></div><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{thresholds.map(([label, value]) => <div key={label} className={`rounded-lg border p-4 ${label === 'Rendement requis H' ? 'border-cyan-800 bg-cyan-950/20' : 'border-slate-800 bg-slate-900/50'}`}><div className="text-xs font-semibold text-slate-500">{label}</div><div className="mt-2 text-lg font-semibold text-slate-100">{value === null ? <span className="text-sm font-medium text-slate-500">Non disponible</span> : <PriceDisplay value={value} {...priceProps}/>}</div></div>)}</div></section>

    <SnapshotComparison snapshots={history} {...priceProps}/>

    <section className="space-y-4"><h2 className="text-xl font-semibold text-white">Thèse structurée V2</h2><div className="grid gap-4 lg:grid-cols-3"><TextBlock title="Quality case" value={company.quality_case ?? null}/><TextBlock title="Valuation case" value={company.valuation_case ?? null}/><TextBlock title="Key risk" value={company.key_risk ?? null}/></div></section>

    <section className="space-y-4"><h2 className="text-xl font-semibold text-white">État analytique publié</h2><div className="grid gap-4 lg:grid-cols-2"><TextBlock title="Triggers d’invalidation" value={company.invalidation}/><TextBlock title="Readiness / prochaine action" value={company.notes}/></div><div className="grid gap-4 rounded-xl border border-slate-800 bg-slate-900/55 p-5 sm:grid-cols-2 lg:grid-cols-4"><Field label="Autorité">{company.source_title ?? 'Non renseignée'}</Field><Field label="Version">{company.model_version ?? 'Non renseignée'}</Field><Field label="Date d’analyse">{company.analysis_date ?? 'Non renseignée'}</Field><Field label="Snapshot">Courant publié</Field></div></section>

    <section className="space-y-4"><div><h2 className="text-xl font-semibold text-white">Composants de score</h2><p className="mt-1 text-sm text-slate-500">Projection directe des sorties canoniques OQS, OVS, Investment et dimensions sous-jacentes. La taxonomie V2 n’a aucune autorité de scoring.</p></div><ScoreComponents value={company.score_components}/></section>

    <section className="space-y-4"><div><h2 className="text-xl font-semibold text-white">Historique canonique</h2><p className="mt-1 text-sm text-slate-500">Seuls les snapshots de <code>research_snapshots</code> du dossier canonique sont listés.</p></div><div className="rounded-xl border border-slate-800 bg-slate-900/55 p-2"><SnapshotHistory snapshots={history} currency={company.currency} quoteUnit={company.quote_unit} priceDecimals={company.price_decimals}/></div></section>
  </div>;
}
