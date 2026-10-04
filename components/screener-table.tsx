'use client';

import { useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { CompanyStatusBadge } from '@/components/company-status-badge';
import { EntryZoneBadge } from '@/components/entry-zone-badge';
import { EngineBadge, ResearchFreshnessBadge } from '@/components/engine-provenance-badges';
import { OroTitanDistance } from '@/components/orotitan-distance';
import { PriceDisplay } from '@/components/price-display';
import { ScoreBadge } from '@/components/score-badge';
import { getDistanceO90 } from '@/lib/domain/distance';
import { getEntryZone, type EntryZone } from '@/lib/domain/entry-zone';
import { getFreshness } from '@/lib/domain/freshness';
import type { CompanyState, CompanyStatus } from '@/lib/domain/types';

type SortKey = 'currentScore' | 'valuation' | 'oqs' | 'distance' | 'analysisDate';
type SortDirection = 'asc' | 'desc';
type FocusFilter = 'ALL' | 'ENTRY' | 'NEAR' | 'SCORE70';

type Row = CompanyState & {
  distance_o90_pct: number | null;
  entry_zone: EntryZone;
  stale: boolean;
  freshness_label: string;
  current_valuation_score: number | null;
  current_investment_score: number | null;
};

function compareNullableNumber(a: number | null, b: number | null, direction: SortDirection): number {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return direction === 'asc' ? a - b : b - a;
}

function compareRow(a: Row, b: Row, key: SortKey, direction: SortDirection): number {
  if (key === 'currentScore') return compareNullableNumber(a.current_investment_score, b.current_investment_score, direction);
  if (key === 'valuation') return compareNullableNumber(a.current_valuation_score, b.current_valuation_score, direction);
  if (key === 'oqs') return compareNullableNumber(a.business_quality_score, b.business_quality_score, direction);
  if (key === 'distance') return compareNullableNumber(a.distance_o90_pct, b.distance_o90_pct, direction);
  const left = a.analysis_date ?? '';
  const right = b.analysis_date ?? '';
  return direction === 'asc' ? left.localeCompare(right) : right.localeCompare(left);
}

function formatPct(value: number | null, digits = 1): string {
  if (value === null || !Number.isFinite(value)) return '—';
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)} %`;
}

export function ScreenerTable({ companies }: { companies: CompanyState[] }) {
  const router = useRouter();
  const [search, setSearch] = useState('');
  const [focus, setFocus] = useState<FocusFilter>('ALL');
  const [status, setStatus] = useState<'ALL' | CompanyStatus>('ALL');
  const [sector, setSector] = useState('ALL');
  const [country, setCountry] = useState('ALL');
  const [industryGroup, setIndustryGroup] = useState('ALL');
  const [businessModel, setBusinessModel] = useState('ALL');
  const [engine, setEngine] = useState<'ALL' | NonNullable<CompanyState['engine_status']>>('ALL');
  const [researchFreshness, setResearchFreshness] = useState<'ALL' | NonNullable<CompanyState['research_freshness_status']>>('ALL');
  const [freshness, setFreshness] = useState<'ALL' | 'FRESH' | 'STALE'>('ALL');
  const [scoreMin, setScoreMin] = useState('');
  const [distanceMin, setDistanceMin] = useState('');
  const [distanceMax, setDistanceMax] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('currentScore');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const rows = useMemo<Row[]>(() => companies.map((company) => {
    const distance = getDistanceO90(company.price, company.price_o90);
    const freshnessState = getFreshness(company.price_as_of);
    return {
      ...company,
      distance_o90_pct: distance,
      entry_zone: getEntryZone(distance),
      stale: freshnessState.stale,
      freshness_label: freshnessState.label,
      current_valuation_score: company.live_valuation_score ?? company.valuation_score,
      current_investment_score: company.live_investment_score ?? company.investment_score,
    };
  }), [companies]);

  const sectors = useMemo(() => [...new Set(rows.map((row) => row.sector).filter((value): value is string => Boolean(value)))].sort(), [rows]);
  const countries = useMemo(() => [...new Set(rows.map((row) => row.country).filter((value): value is string => Boolean(value)))].sort(), [rows]);
  const industryGroups = useMemo(() => [...new Set(rows.map((row) => row.industry_group).filter((value): value is string => Boolean(value)))].sort(), [rows]);
  const businessModels = useMemo(() => [...new Set(rows.map((row) => row.business_model_primary).filter((value): value is string => Boolean(value)))].sort(), [rows]);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const minScore = scoreMin.trim() ? Number(scoreMin) : null;
    const minDistance = distanceMin.trim() ? Number(distanceMin) : null;
    const maxDistance = distanceMax.trim() ? Number(distanceMax) : null;

    return rows
      .filter((row) => !needle || row.name.toLowerCase().includes(needle) || row.ticker.toLowerCase().includes(needle))
      .filter((row) => status === 'ALL' || row.status === status)
      .filter((row) => sector === 'ALL' || row.sector === sector)
      .filter((row) => country === 'ALL' || row.country === country)
      .filter((row) => industryGroup === 'ALL' || row.industry_group === industryGroup)
      .filter((row) => businessModel === 'ALL' || row.business_model_primary === businessModel)
      .filter((row) => engine === 'ALL' || (row.engine_status ?? 'UNKNOWN') === engine)
      .filter((row) => researchFreshness === 'ALL' || (row.research_freshness_status ?? 'UNKNOWN') === researchFreshness)
      .filter((row) => freshness === 'ALL' || (freshness === 'FRESH' && !row.stale) || (freshness === 'STALE' && row.stale))
      .filter((row) => minScore === null || (row.current_investment_score !== null && row.current_investment_score >= minScore))
      .filter((row) => minDistance === null || (row.distance_o90_pct !== null && row.distance_o90_pct >= minDistance))
      .filter((row) => maxDistance === null || (row.distance_o90_pct !== null && row.distance_o90_pct <= maxDistance))
      .filter((row) => {
        if (focus === 'ENTRY') return row.distance_o90_pct !== null && row.distance_o90_pct >= 0;
        if (focus === 'NEAR') return row.distance_o90_pct !== null && row.distance_o90_pct >= -10;
        if (focus === 'SCORE70') return row.current_investment_score !== null && row.current_investment_score >= 70;
        return true;
      })
      .sort((a, b) => compareRow(a, b, sortKey, sortDirection));
  }, [rows, search, focus, status, sector, country, industryGroup, businessModel, engine, researchFreshness, freshness, scoreMin, distanceMin, distanceMax, sortKey, sortDirection]);

  const stats = useMemo(() => ({
    count: filtered.length,
    fresh: filtered.filter((row) => !row.stale && row.price_source !== 'CANONICAL_REFERENCE_PRICE').length,
    entry: filtered.filter((row) => row.distance_o90_pct !== null && row.distance_o90_pct >= 0).length,
    attractive: filtered.filter((row) => row.current_investment_score !== null && row.current_investment_score >= 70).length,
  }), [filtered]);

  function setSort(next: SortKey) {
    if (next === sortKey) setSortDirection((current) => current === 'asc' ? 'desc' : 'asc');
    else {
      setSortKey(next);
      setSortDirection('desc');
    }
  }

  function reset() {
    setSearch('');
    setFocus('ALL');
    setStatus('ALL');
    setSector('ALL');
    setCountry('ALL');
    setIndustryGroup('ALL');
    setBusinessModel('ALL');
    setEngine('ALL');
    setResearchFreshness('ALL');
    setFreshness('ALL');
    setScoreMin('');
    setDistanceMin('');
    setDistanceMax('');
    setSortKey('currentScore');
    setSortDirection('desc');
  }

  const sortMark = (key: SortKey) => key === sortKey ? (sortDirection === 'asc' ? ' ↑' : ' ↓') : '';

  return (
    <div className="space-y-5">
      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Metric label="Sociétés affichées" value={String(stats.count)} hint="univers filtré"/>
        <Metric label="Cours marché frais" value={String(stats.fresh)} hint="hors prix de référence"/>
        <Metric label="Seuil H atteint" value={String(stats.entry)} hint="prix ≤ seuil de rendement"/>
        <Metric label="Score actuel ≥ 70" value={String(stats.attractive)} hint="après repricing du cours"/>
      </section>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/55 p-4 shadow-sm shadow-black/10">
        <div className="grid gap-3 lg:grid-cols-[minmax(240px,1.5fr)_1fr_1fr_auto]">
          <label className="text-sm text-slate-400">
            Rechercher
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Société ou ticker" className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100 outline-none transition focus:border-cyan-500"/>
          </label>
          <Select label="Statut" value={status} onChange={(value) => setStatus(value as 'ALL' | CompanyStatus)} options={[['ALL','Tous'],['OROTITAN','OroTitan'],['FINALIST','Finaliste'],['PRICE_WAIT','Attendre le prix'],['WATCHLIST','Liste de suivi'],['REJECTED','Rejetée']]}/>
          <Select label="Secteur" value={sector} onChange={setSector} options={[['ALL','Tous'],...sectors.map((value) => [value,value] as [string,string])]}/>
          <button onClick={reset} className="self-end rounded-xl border border-slate-700 px-4 py-2.5 text-sm text-slate-300 transition hover:border-slate-600 hover:bg-slate-800">Réinitialiser</button>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          <QuickFilter active={focus === 'ALL'} onClick={() => setFocus('ALL')}>Toutes</QuickFilter>
          <QuickFilter active={focus === 'ENTRY'} onClick={() => setFocus('ENTRY')}>Seuil H atteint</QuickFilter>
          <QuickFilter active={focus === 'NEAR'} onClick={() => setFocus('NEAR')}>À moins de 10 % du seuil</QuickFilter>
          <QuickFilter active={focus === 'SCORE70'} onClick={() => setFocus('SCORE70')}>Score actuel ≥ 70</QuickFilter>
        </div>

        <details className="mt-4 border-t border-slate-800 pt-4">
          <summary className="cursor-pointer select-none text-sm font-medium text-slate-300">Filtres avancés</summary>
          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <Select label="Pays" value={country} onChange={setCountry} options={[['ALL','Tous'],...countries.map((value) => [value,value] as [string,string])]}/>
            <Select label="Industrie" value={industryGroup} onChange={setIndustryGroup} options={[['ALL','Toutes'],...industryGroups.map((value) => [value,value] as [string,string])]}/>
            <Select label="Modèle économique" value={businessModel} onChange={setBusinessModel} options={[['ALL','Tous'],...businessModels.map((value) => [value,value] as [string,string])]}/>
            <Select label="Génération moteur" value={engine} onChange={(value) => setEngine(value as typeof engine)} options={[['ALL','Toutes'],['CURRENT','Moteur actuel'],['PREVIOUS','Moteur précédent'],['LEGACY','Legacy'],['UNKNOWN','Inconnu']]}/>
            <Select label="Fraîcheur recherche" value={researchFreshness} onChange={(value) => setResearchFreshness(value as typeof researchFreshness)} options={[['ALL','Toutes'],['RECENT','Récentes'],['AGING','Vieillissantes'],['STALE','À rafraîchir'],['UNKNOWN','Inconnues']]}/>
            <Select label="Fraîcheur du cours" value={freshness} onChange={(value) => setFreshness(value as typeof freshness)} options={[['ALL','Toutes'],['FRESH','Récentes'],['STALE','Périmées']]}/>
            <NumericFilter label="Score actuel min." value={scoreMin} onChange={setScoreMin} placeholder="ex. 70"/>
            <NumericFilter label="Écart H min. %" value={distanceMin} onChange={setDistanceMin} placeholder="ex. -10"/>
            <NumericFilter label="Écart H max. %" value={distanceMax} onChange={setDistanceMax} placeholder="ex. 5"/>
            <Select label="Trier par" value={sortKey} onChange={(value) => setSortKey(value as SortKey)} options={[['currentScore','Score actuel'],['valuation','OVS actuel'],['oqs','Qualité OQS'],['distance','Écart au seuil H'],['analysisDate','Date d’analyse']]}/>
          </div>
        </details>
      </section>

      <div className="grid gap-3 md:hidden">
        {filtered.map((row) => <MobileCompanyCard key={row.id} row={row} onOpen={() => router.push(`/company/${row.slug}`)}/>)}
      </div>

      <div className="hidden overflow-x-auto rounded-2xl border border-slate-800 bg-slate-950/60 md:block">
        <table className="min-w-[1120px] w-full text-left text-sm">
          <thead className="border-b border-slate-800 bg-slate-900/85 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-4 py-3">Société</th>
              <th className="px-4 py-3">Cours actuel</th>
              <th className="px-4 py-3"><button onClick={() => setSort('oqs')}>Qualité OQS{sortMark('oqs')}</button></th>
              <th className="px-4 py-3"><button onClick={() => setSort('valuation')}>OVS actuel{sortMark('valuation')}</button></th>
              <th className="px-4 py-3"><button onClick={() => setSort('currentScore')}>Score actuel{sortMark('currentScore')}</button></th>
              <th className="px-4 py-3">Seuil H</th>
              <th className="px-4 py-3"><button onClick={() => setSort('distance')}>Écart H{sortMark('distance')}</button></th>
              <th className="px-4 py-3">Statut</th>
              <th className="px-4 py-3"><button onClick={() => setSort('analysisDate')}>Analyse{sortMark('analysisDate')}</button></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {filtered.map((row) => {
              const priceProps = { currency: row.currency, quoteUnit: row.quote_unit, priceDecimals: row.price_decimals };
              const dynamicOvs = row.live_valuation_score;
              const dynamicInvestment = row.live_investment_score;
              return (
                <tr key={row.id} tabIndex={0} role="link" onClick={() => router.push(`/company/${row.slug}`)} onKeyDown={(event) => { if (event.key === 'Enter') router.push(`/company/${row.slug}`); }} className="cursor-pointer transition hover:bg-slate-900/70 focus:bg-slate-900/70 focus:outline-none">
                  <td className="px-4 py-4">
                    <div className="font-medium text-slate-100">{row.name}</div>
                    <div className="mt-1 font-mono text-xs text-slate-500">{row.ticker} · {row.exchange}</div>
                    <div className="mt-2 flex flex-wrap gap-1.5"><EngineBadge status={row.engine_status ?? 'UNKNOWN'} shortFingerprint={row.engine_short_fingerprint}/><ResearchFreshnessBadge status={row.research_freshness_status ?? 'UNKNOWN'} ageDays={row.analysis_age_days}/></div>
                  </td>
                  <td className="px-4 py-4">
                    <div className="font-semibold text-slate-100"><PriceDisplay value={row.price} {...priceProps}/></div>
                    <div className={`mt-1 text-[11px] ${row.stale ? 'text-amber-300' : 'text-slate-500'}`}>{row.freshness_label}</div>
                  </td>
                  <td className="px-4 py-4"><ScoreBadge score={row.business_quality_score}/></td>
                  <td className="px-4 py-4">
                    <ScoreBadge score={row.current_valuation_score}/>
                    {dynamicOvs !== null && row.valuation_score !== null && Math.abs(dynamicOvs - row.valuation_score) > 0.05
                      ? <div className="mt-1 text-[11px] text-slate-600">canonique {row.valuation_score.toFixed(0)}</div>
                      : null}
                  </td>
                  <td className="px-4 py-4">
                    <ScoreBadge score={row.current_investment_score}/>
                    {dynamicInvestment !== null && row.investment_score !== null && Math.abs(dynamicInvestment - row.investment_score) > 0.05
                      ? <div className="mt-1 text-[11px] text-slate-600">canonique {row.investment_score.toFixed(0)}</div>
                      : null}
                  </td>
                  <td className="px-4 py-4 text-slate-200">{row.price_o90 === null ? <span className="text-slate-500">Non calibré</span> : <PriceDisplay value={row.price_o90} {...priceProps}/>}</td>
                  <td className="px-4 py-4"><OroTitanDistance value={row.distance_o90_pct} compact/></td>
                  <td className="px-4 py-4"><CompanyStatusBadge status={row.status}/></td>
                  <td className="px-4 py-4 font-mono text-xs text-slate-500">{row.analysis_date ?? '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {filtered.length === 0 ? <div className="p-8 text-center text-sm text-slate-500">Aucune société ne correspond aux filtres.</div> : null}
      </div>

      <div className="flex flex-col gap-1 text-xs text-slate-500 sm:flex-row sm:items-center sm:justify-between">
        <span>{filtered.length} société{filtered.length > 1 ? 's' : ''} affichée{filtered.length > 1 ? 's' : ''}.</span>
        <span>OVS actuel = recalcul prix-only ; qualité et caps analytiques restent ceux du snapshot publié.</span>
      </div>
    </div>
  );
}

function MobileCompanyCard({ row, onOpen }: { row: Row; onOpen: () => void }) {
  const priceProps = { currency: row.currency, quoteUnit: row.quote_unit, priceDecimals: row.price_decimals };
  return (
    <button onClick={onOpen} className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 text-left transition hover:border-cyan-900 hover:bg-slate-900">
      <div className="flex items-start justify-between gap-4">
        <div><div className="font-semibold text-white">{row.name}</div><div className="mt-1 font-mono text-xs text-slate-500">{row.ticker} · {row.exchange}</div><div className="mt-2 flex flex-wrap gap-1.5"><EngineBadge status={row.engine_status ?? 'UNKNOWN'} shortFingerprint={row.engine_short_fingerprint}/><ResearchFreshnessBadge status={row.research_freshness_status ?? 'UNKNOWN'} ageDays={row.analysis_age_days}/></div></div>
        <CompanyStatusBadge status={row.status}/>
      </div>
      <div className="mt-4 flex items-end justify-between gap-4 border-b border-slate-800 pb-4">
        <div><div className="text-xs text-slate-500">Cours actuel</div><div className="mt-1 text-xl font-semibold text-white"><PriceDisplay value={row.price} {...priceProps}/></div><div className={`mt-1 text-[11px] ${row.stale ? 'text-amber-300' : 'text-slate-500'}`}>{row.freshness_label}</div></div>
        <EntryZoneBadge zone={row.entry_zone}/>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-3">
        <MiniScore label="OQS" value={row.business_quality_score}/>
        <MiniScore label="OVS actuel" value={row.current_valuation_score}/>
        <MiniScore label="Score actuel" value={row.current_investment_score}/>
      </div>
      <div className="mt-4 flex items-center justify-between text-xs text-slate-500">
        <span>Seuil H : {row.price_o90 === null ? 'non calibré' : <PriceDisplay value={row.price_o90} {...priceProps}/>}</span>
        <span>{formatPct(row.distance_o90_pct)}</span>
      </div>
    </button>
  );
}

function Metric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4"><div className="text-xs uppercase tracking-wide text-slate-500">{label}</div><div className="mt-2 text-2xl font-semibold text-white">{value}</div><div className="mt-1 text-xs text-slate-600">{hint}</div></div>;
}

function MiniScore({ label, value }: { label: string; value: number | null }) {
  return <div><div className="text-[11px] text-slate-500">{label}</div><div className="mt-1 font-mono text-lg font-semibold text-slate-100">{value === null ? '—' : value.toFixed(0)}</div></div>;
}

function QuickFilter({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return <button onClick={onClick} className={`rounded-full border px-3 py-1.5 text-xs font-medium transition ${active ? 'border-cyan-500/50 bg-cyan-500/10 text-cyan-200' : 'border-slate-700 bg-slate-950 text-slate-400 hover:text-slate-200'}`}>{children}</button>;
}

function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: Array<[string,string]> }) {
  return <label className="text-sm text-slate-400">{label}<select value={value} onChange={(event) => onChange(event.target.value)} className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100 outline-none focus:border-cyan-500">{options.map(([optionValue, optionLabel]) => <option key={optionValue} value={optionValue}>{optionLabel}</option>)}</select></label>;
}

function NumericFilter({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (value: string) => void; placeholder: string }) {
  return <label className="text-sm text-slate-400">{label}<input value={value} onChange={(event) => onChange(event.target.value)} type="number" step="0.1" placeholder={placeholder} className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100 outline-none focus:border-cyan-500"/></label>;
}
