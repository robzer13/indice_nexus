import Link from 'next/link';
import { MetricCard } from '@/components/metric-card';
import { OroTitanDistance } from '@/components/orotitan-distance';
import { PriceDisplay } from '@/components/price-display';
import { ScoreBadge } from '@/components/score-badge';
import { CompanyStatusBadge } from '@/components/company-status-badge';
import { getCompanyStates } from '@/lib/data/companies';
import { getDistanceO90 } from '@/lib/domain/distance';
import { prioritizeCompanies } from '@/lib/domain/prioritization';
import { getFreshness } from '@/lib/domain/freshness';

export const dynamic = 'force-dynamic';

export default async function DashboardPage() {
  const companies = await getCompanyStates();
  const priorities = prioritizeCompanies(companies, 6);
  const calibrated = companies.filter((company) => company.price_o90 !== null).length;
  const reached = companies.filter((company) => { const distance = getDistanceO90(company.price, company.price_o90); return distance !== null && distance >= 0; }).length;
  const newestPrice = companies.map((company) => company.price_as_of).filter((value): value is string => Boolean(value)).sort().at(-1) ?? null;
  const freshness = getFreshness(newestPrice);

  return <div className="space-y-8">
    <section className="flex flex-col gap-5 border-b border-slate-800 pb-7 md:flex-row md:items-end md:justify-between"><div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Cockpit d’entrée</div><h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">Quelles sociétés sont les plus proches d’un point d’entrée OroTitan ?</h1><p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">Base fermée de sociétés déjà analysées. Les analyses restent immutables ; seuls de nouveaux snapshots de marché sont ajoutés.</p></div><Link href="/screener" className="inline-flex shrink-0 items-center justify-center rounded-lg border border-cyan-700 bg-cyan-950/40 px-4 py-2.5 text-sm font-semibold text-cyan-100 hover:bg-cyan-950/70">Ouvrir le screener</Link></section>

    <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><MetricCard label="Sociétés suivies" value={companies.length} hint="sociétés actives"/><MetricCard label="O90 calibré" value={calibrated} hint="snapshot courant"/><MetricCard label="À O90 ou mieux" value={reached} hint="distance O90 ≥ 0 %"/><MetricCard label="Fraîcheur marché" value={<span className={freshness.stale ? 'text-amber-200' : 'text-emerald-200'}>{freshness.label}</span>} hint={newestPrice ? `Dernier cours: ${new Date(newestPrice).toLocaleString('fr-FR')}` : 'Aucun cours disponible'}/></section>

    <section><div className="mb-4 flex items-end justify-between"><div><h2 className="text-xl font-semibold text-white">Six priorités du moment</h2><p className="mt-1 text-sm text-slate-500">O90 atteint d’abord, puis sociétés les plus proches du seuil depuis le dessus. Aucun O90 absent n’est transformé en distance.</p></div></div>{priorities.length === 0 ? <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 text-sm text-slate-500">Aucune priorité calculable : aucun couple cours/O90 valide.</div> : <div className="grid gap-4 lg:grid-cols-2">{priorities.map((company, index) => { const priceProps = { currency: company.currency, quoteUnit: company.quote_unit, priceDecimals: company.price_decimals }; return <Link key={company.id} href={`/company/${company.slug}`} className="group rounded-xl border border-slate-800 bg-slate-900/65 p-5 shadow-panel transition hover:border-slate-700 hover:bg-slate-900"><div className="flex items-start justify-between gap-4"><div><div className="text-xs font-mono text-slate-600">#{index + 1} · {company.ticker}</div><h3 className="mt-1 text-lg font-semibold text-slate-100 group-hover:text-white">{company.name}</h3></div><CompanyStatusBadge status={company.status}/></div><div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4"><div><div className="text-xs text-slate-500">Cours</div><div className="mt-1 text-base font-semibold text-white"><PriceDisplay value={company.price} {...priceProps}/></div></div><div><div className="text-xs text-slate-500">O90</div><div className="mt-1 text-slate-200"><PriceDisplay value={company.price_o90} {...priceProps}/></div></div><div><div className="text-xs text-slate-500">Distance</div><div className="mt-1"><OroTitanDistance value={company.distance_o90_pct} compact/></div></div><div><div className="text-xs text-slate-500">Score</div><div className="mt-1"><ScoreBadge score={company.orotitan_score}/></div></div></div></Link>; })}</div>}</section>
  </div>;
}
