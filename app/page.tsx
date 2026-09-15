import Link from 'next/link';
import { CompanyStatusBadge } from '@/components/company-status-badge';
import { EntryZoneBadge } from '@/components/entry-zone-badge';
import { MetricCard } from '@/components/metric-card';
import { OroTitanDistance } from '@/components/orotitan-distance';
import { PriceDisplay } from '@/components/price-display';
import { ScoreBadge } from '@/components/score-badge';
import { getCompanyStates } from '@/lib/data/companies';
import { summarizeDataHealth } from '@/lib/domain/data-health';
import { getDistanceO90 } from '@/lib/domain/distance';
import { getEntryZone } from '@/lib/domain/entry-zone';
import { prioritizeCompanies } from '@/lib/domain/prioritization';

export const dynamic = 'force-dynamic';

export default async function DashboardPage() {
  const companies = await getCompanyStates();
  const priorities = prioritizeCompanies(companies, 6);
  const enriched = companies.map((company) => {
    const distance = getDistanceO90(company.price, company.price_o90);
    return { ...company, distance, zone: getEntryZone(distance) };
  });
  const reached = enriched.filter((company) => company.zone === 'AT_OR_BELOW_O90').length;
  const within5 = enriched.filter((company) => company.zone === 'WITHIN_5').length;
  const within10 = enriched.filter((company) => company.zone === 'WITHIN_10').length;
  const within20 = enriched.filter((company) => company.zone === 'WITHIN_20').length;
  const uncalibrated = enriched.filter((company) => company.zone === 'UNCALIBRATED').length;
  const health = summarizeDataHealth(companies);
  const stale = health.rows.filter((row) => row.issues.some((issue) => issue.code === 'STALE_PRICE' || issue.code === 'MISSING_PRICE')).length;

  const zoneCounts = [
    ['AT_OR_BELOW_O90', reached],
    ['WITHIN_5', within5],
    ['WITHIN_10', within10],
    ['WITHIN_20', within20],
    ['UNCALIBRATED', uncalibrated],
  ] as const;

  return <div className="space-y-8">
    <section className="flex flex-col gap-5 border-b border-slate-800 pb-7 md:flex-row md:items-end md:justify-between"><div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Canonical cockpit · V1</div><h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">Sociétés officiellement publiées dans OroTitan</h1><p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">La liste est dérivée uniquement des dossiers dont le pointeur canonique <code>current_snapshot_id</code> est renseigné. Les analyses legacy non publiées ne sont plus exposées.</p></div><Link href="/screener" className="inline-flex shrink-0 items-center justify-center rounded-lg border border-cyan-700 bg-cyan-950/40 px-4 py-2.5 text-sm font-semibold text-cyan-100 hover:bg-cyan-950/70">Ouvrir le screener</Link></section>

    <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6"><MetricCard label="Publiées" value={companies.length} hint="snapshots canoniques"/><MetricCard label="Seuil H atteint" value={reached} hint="cours ≤ prix requis"/><MetricCard label="À < 5 %" value={within5} hint="du seuil H"/><MetricCard label="À 5–10 %" value={within10} hint="du seuil H"/><MetricCard label="Cours à traiter" value={stale} hint="manquants ou périmés"/><MetricCard label="Data Health" value={health.totalIssues} hint="issues détectées"/></section>

    <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-5"><div><h2 className="text-lg font-semibold text-white">Radar d’entrée</h2><p className="mt-1 text-sm text-slate-500">Répartition selon la distance au prix permettant d’atteindre le rendement requis H du snapshot canonique.</p></div><div className="mt-4 flex flex-wrap gap-3">{zoneCounts.map(([zone, count]) => <div key={zone} className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2"><EntryZoneBadge zone={zone}/><span className="font-mono text-sm text-slate-300">{count}</span></div>)}</div></section>

    <section><div className="mb-4"><h2 className="text-xl font-semibold text-white">Priorités publiées</h2><p className="mt-1 text-sm text-slate-500">Classement sur les seules sociétés canoniques publiées, selon leur distance au seuil H puis leur score d’investissement.</p></div>{priorities.length === 0 ? <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 text-sm text-slate-500">Aucune société publiée.</div> : <div className="grid gap-4 lg:grid-cols-2">{priorities.map((company, index) => { const priceProps = { currency: company.currency, quoteUnit: company.quote_unit, priceDecimals: company.price_decimals }; const zone = getEntryZone(company.distance_o90_pct); return <Link key={company.id} href={`/company/${company.slug}`} className="group rounded-xl border border-slate-800 bg-slate-900/65 p-5 shadow-panel transition hover:border-slate-700 hover:bg-slate-900"><div className="flex items-start justify-between gap-4"><div><div className="text-xs font-mono text-slate-600">#{index + 1} · {company.ticker}</div><h3 className="mt-1 text-lg font-semibold text-slate-100">{company.name}</h3></div><CompanyStatusBadge status={company.status}/></div><div className="mt-4"><EntryZoneBadge zone={zone}/></div><div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4"><div><div className="text-xs text-slate-500">Cours</div><div className="mt-1 font-semibold text-white"><PriceDisplay value={company.price} {...priceProps}/></div></div><div><div className="text-xs text-slate-500">Seuil H</div><div className="mt-1 text-slate-200"><PriceDisplay value={company.price_o90} {...priceProps}/></div></div><div><div className="text-xs text-slate-500">Distance</div><div className="mt-1"><OroTitanDistance value={company.distance_o90_pct} compact/></div></div><div><div className="text-xs text-slate-500">Investment</div><div className="mt-1"><ScoreBadge score={company.investment_score}/></div></div></div></Link>; })}</div>}</section>
  </div>;
}
