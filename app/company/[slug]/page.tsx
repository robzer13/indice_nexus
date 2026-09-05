import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { CompanyStatusBadge } from '@/components/company-status-badge';
import { FairValueRange } from '@/components/fair-value-range';
import { OroTitanDistance } from '@/components/orotitan-distance';
import { PriceDisplay } from '@/components/price-display';
import { ScoreBadge } from '@/components/score-badge';
import { ScoreComponents } from '@/components/score-components';
import { SnapshotHistory } from '@/components/snapshot-history';
import { getCompanyStateBySlug, getSnapshotHistory } from '@/lib/data/companies';
import { getDistanceO90 } from '@/lib/domain/distance';
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
  const history = await getSnapshotHistory(company.id);
  const distance = getDistanceO90(company.price, company.price_o90);
  const freshness = getFreshness(company.price_as_of);
  const priceProps = { currency: company.currency, quoteUnit: company.quote_unit, priceDecimals: company.price_decimals };
  const thresholds = [
    ['O85', company.price_o85],
    ['O90', company.price_o90],
    ['O92', company.price_o92],
    ['O95', company.price_o95],
  ] as const;

  return <div className="space-y-8">
    <section className="flex flex-col gap-5 border-b border-slate-800 pb-7 lg:flex-row lg:items-end lg:justify-between"><div><div className="font-mono text-xs text-cyan-400">{company.ticker} · {company.exchange}</div><h1 className="mt-2 text-3xl font-semibold text-white sm:text-4xl">{company.name}</h1><p className="mt-2 text-sm text-slate-500">{company.country ?? 'Pays non renseigné'} · {company.sector ?? 'Secteur non renseigné'}</p></div><CompanyStatusBadge status={company.status}/></section>

    <section className="grid gap-4 lg:grid-cols-[1.4fr_1fr]"><div className="rounded-xl border border-slate-800 bg-slate-900/65 p-5"><div className="text-xs uppercase tracking-[0.15em] text-slate-500">État actuel</div><div className="mt-4 flex flex-wrap items-end gap-6"><div><div className="text-xs text-slate-500">Cours</div><div className="mt-1 text-3xl font-semibold text-white"><PriceDisplay value={company.price} {...priceProps}/></div></div><div><div className="text-xs text-slate-500">O90</div><div className="mt-1 text-xl text-slate-200">{company.price_o90 === null ? 'Non calibré' : <PriceDisplay value={company.price_o90} {...priceProps}/>}</div></div><div><div className="text-xs text-slate-500">Distance O90</div><div className="mt-1"><OroTitanDistance value={distance}/></div></div><div><div className="text-xs text-slate-500">OroTitan Score</div><div className="mt-1"><ScoreBadge score={company.orotitan_score}/></div></div></div><div className="mt-5 grid gap-4 border-t border-slate-800 pt-4 sm:grid-cols-2 lg:grid-cols-4"><Field label="Date du cours">{company.price_as_of ? new Date(company.price_as_of).toLocaleString('fr-FR') : 'Non disponible'}</Field><Field label="Source du cours">{company.price_source ?? 'Non disponible'}</Field><Field label="Fraîcheur"><span className={freshness.stale ? 'text-amber-200' : 'text-emerald-200'}>{freshness.label}</span></Field><Field label="Qualité OroTitan">{company.quality_orotitan === null ? 'Non renseigné' : company.quality_orotitan ? 'Oui' : 'Non'}</Field></div></div><div className="rounded-xl border border-slate-800 bg-slate-900/65 p-5"><div className="text-xs uppercase tracking-[0.15em] text-slate-500">Scores & confiance</div><div className="mt-4 grid grid-cols-2 gap-4"><Field label="Business quality">{company.business_quality_score ?? '—'}</Field><Field label="Investment">{company.investment_score ?? '—'}</Field><Field label="Valuation">{company.valuation_score ?? '—'}</Field><Field label="Confiance /10">{company.confidence_score ?? '—'}</Field></div></div></section>

    <section className="space-y-4"><div><h2 className="text-xl font-semibold text-white">Valorisation</h2><p className="mt-1 text-sm text-slate-500">Fair values et seuils proviennent exclusivement du snapshot courant sélectionné. Aucun mélange silencieux entre modèles.</p></div><FairValueRange company={company}/><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{thresholds.map(([label, value]) => <div key={label} className={`rounded-lg border p-4 ${label === 'O90' ? 'border-cyan-800 bg-cyan-950/20' : 'border-slate-800 bg-slate-900/50'}`}><div className="text-xs font-semibold text-slate-500">{label}</div><div className="mt-2 text-lg font-semibold text-slate-100">{value === null ? <span className="text-sm font-medium text-slate-500">Non calibré</span> : <PriceDisplay value={value} {...priceProps}/>}</div></div>)}</div></section>

    <section className="space-y-4"><h2 className="text-xl font-semibold text-white">Analyse</h2><div className="grid gap-4 lg:grid-cols-3"><TextBlock title="Thèse" value={company.thesis}/><TextBlock title="Risque principal" value={company.main_risk}/><TextBlock title="Invalidation" value={company.invalidation}/></div><div className="grid gap-4 rounded-xl border border-slate-800 bg-slate-900/55 p-5 sm:grid-cols-2 lg:grid-cols-4"><Field label="Source">{company.source_title ?? 'Non renseignée'}</Field><Field label="Model version">{company.model_version ?? 'Non renseignée'}</Field><Field label="Date d’analyse">{company.analysis_date ?? 'Non renseignée'}</Field><Field label="Notes">{company.notes ?? 'Aucune'}</Field></div></section>

    <section className="space-y-4"><div><h2 className="text-xl font-semibold text-white">Score components</h2><p className="mt-1 text-sm text-slate-500">Le JSON est affiché tel qu’il existe, sans supposer une structure universelle.</p></div><ScoreComponents value={company.score_components}/></section>

    <section className="space-y-4"><div><h2 className="text-xl font-semibold text-white">Historique des snapshots</h2><p className="mt-1 text-sm text-slate-500">Chaque ligne conserve son modèle, sa date et ses sorties propres.</p></div><div className="rounded-xl border border-slate-800 bg-slate-900/55 p-2"><SnapshotHistory snapshots={history} currency={company.currency} quoteUnit={company.quote_unit} priceDecimals={company.price_decimals}/></div></section>
  </div>;
}
