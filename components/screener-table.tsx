'use client';

import { useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { CompanyStatusBadge } from '@/components/company-status-badge';
import { EntryZoneBadge } from '@/components/entry-zone-badge';
import { OroTitanDistance } from '@/components/orotitan-distance';
import { PriceDisplay } from '@/components/price-display';
import { ScoreBadge } from '@/components/score-badge';
import { getDistanceO90 } from '@/lib/domain/distance';
import { getEntryZone, type EntryZone } from '@/lib/domain/entry-zone';
import { getFreshness } from '@/lib/domain/freshness';
import type { CompanyState, CompanyStatus } from '@/lib/domain/types';

type SortKey = 'distance' | 'marketScore' | 'quality' | 'valuation' | 'analysisDate' | 'price';
type SortDirection = 'asc' | 'desc';
type Row = CompanyState & {
  distance_o90_pct: number | null;
  entry_zone: EntryZone;
  stale: boolean;
  display_valuation_score: number | null;
  display_investment_score: number | null;
};

const statusOptions: Array<[string, string]> = [
  ['ALL', 'Tous'],
  ['OROTITAN', 'OroTitan'],
  ['FINALIST', 'Finaliste'],
  ['PRICE_WAIT', 'Attente prix'],
  ['TIER_1', 'Niveau 1'],
  ['WATCHLIST', 'Surveillance'],
  ['REJECTED', 'Rejetée'],
];

function compareNullableNumber(a: number | null, b: number | null, direction: SortDirection): number {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return direction === 'asc' ? a - b : b - a;
}

function compareRow(a: Row, b: Row, key: SortKey, direction: SortDirection): number {
  if (key === 'distance') return compareNullableNumber(a.distance_o90_pct, b.distance_o90_pct, direction);
  if (key === 'marketScore') return compareNullableNumber(a.display_investment_score, b.display_investment_score, direction);
  if (key === 'quality') return compareNullableNumber(a.business_quality_score, b.business_quality_score, direction);
  if (key === 'valuation') return compareNullableNumber(a.display_valuation_score, b.display_valuation_score, direction);
  if (key === 'price') return compareNullableNumber(a.price, b.price, direction);
  const left = a.analysis_date ?? '';
  const right = b.analysis_date ?? '';
  return direction === 'asc' ? left.localeCompare(right) : right.localeCompare(left);
}

function formatUpdate(value: string | null): string {
  if (!value) return 'Pas de cours';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Date inconnue';
  return date.toLocaleString('fr-FR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
}

export function ScreenerTable({ companies }: { companies: CompanyState[] }) {
  const router = useRouter();
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState<'ALL' | CompanyStatus>('ALL');
  const [entryZone, setEntryZone] = useState<'ALL' | EntryZone>('ALL');
  const [marketScoreMin, setMarketScoreMin] = useState('');
  const [country, setCountry] = useState('ALL');
  const [sector, setSector] = useState('ALL');
  const [industryGroup, setIndustryGroup] = useState('ALL');
  const [businessModel, setBusinessModel] = useState('ALL');
  const [freshness, setFreshness] = useState<'ALL' | 'FRESH' | 'STALE'>('ALL');
  const [qualityMin, setQualityMin] = useState('');
  const [valuationMin, setValuationMin] = useState('');
  const [distanceMax, setDistanceMax] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('marketScore');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const rows = useMemo<Row[]>(() => companies.map((company) => {
    const distance = getDistanceO90(company.price, company.price_o90);
    return {
      ...company,
      distance_o90_pct: distance,
      entry_zone: getEntryZone(distance),
      stale: getFreshness(company.price_as_of).stale,
      display_valuation_score: company.market_valuation_score ?? company.valuation_score,
      display_investment_score: company.market_investment_score ?? company.investment_score,
    };
  }), [companies]);

  const countries = useMemo(() => [...new Set(rows.map((row) => row.country).filter((value): value is string => Boolean(value)))].sort(), [rows]);
  const sectors = useMemo(() => [...new Set(rows.map((row) => row.sector).filter((value): value is string => Boolean(value)))].sort(), [rows]);
  const industryGroups = useMemo(() => [...new Set(rows.map((row) => row.industry_group).filter((value): value is string => Boolean(value)))].sort(), [rows]);
  const businessModels = useMemo(() => [...new Set(rows.map((row) => row.business_model_primary).filter((value): value is string => Boolean(value)))].sort(), [rows]);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const minMarketScore = marketScoreMin.trim() ? Number(marketScoreMin) : null;
    const minQuality = qualityMin.trim() ? Number(qualityMin) : null;
    const minValuation = valuationMin.trim() ? Number(valuationMin) : null;
    const maxDistance = distanceMax.trim() ? Number(distanceMax) : null;

    return rows
      .filter((row) => !needle || row.name.toLowerCase().includes(needle) || row.ticker.toLowerCase().includes(needle))
      .filter((row) => status === 'ALL' || row.status === status)
      .filter((row) => entryZone === 'ALL' || row.entry_zone === entryZone)
      .filter((row) => country === 'ALL' || row.country === country)
      .filter((row) => sector === 'ALL' || row.sector === sector)
      .filter((row) => industryGroup === 'ALL' || row.industry_group === industryGroup)
      .filter((row) => businessModel === 'ALL' || row.business_model_primary === businessModel)
      .filter((row) => freshness === 'ALL' || (freshness === 'FRESH' && !row.stale) || (freshness === 'STALE' && row.stale))
      .filter((row) => minMarketScore === null || (row.display_investment_score !== null && row.display_investment_score >= minMarketScore))
      .filter((row) => minQuality === null || (row.business_quality_score !== null && row.business_quality_score >= minQuality))
      .filter((row) => minValuation === null || (row.display_valuation_score !== null && row.display_valuation_score >= minValuation))
      .filter((row) => maxDistance === null || (row.distance_o90_pct !== null && row.distance_o90_pct <= maxDistance))
      .sort((a, b) => compareRow(a, b, sortKey, sortDirection));
  }, [rows, search, status, entryZone, country, sector, industryGroup, businessModel, freshness, marketScoreMin, qualityMin, valuationMin, distanceMax, sortKey, sortDirection]);

  function setSort(next: SortKey) {
    if (next === sortKey) setSortDirection((current) => current === 'asc' ? 'desc' : 'asc');
    else {
      setSortKey(next);
      setSortDirection(next === 'distance' ? 'asc' : 'desc');
    }
  }

  const sortMark = (key: SortKey) => key === sortKey ? (sortDirection === 'asc' ? ' ↑' : ' ↓') : '';

  function reset() {
    setSearch('');
    setStatus('ALL');
    setEntryZone('ALL');
    setMarketScoreMin('');
    setCountry('ALL');
    setSector('ALL');
    setIndustryGroup('ALL');
    setBusinessModel('ALL');
    setFreshness('ALL');
    setQualityMin('');
    setValuationMin('');
    setDistanceMax('');
    setSortKey('marketScore');
    setSortDirection('desc');
  }

  return <div className="space-y-4">
    <div className="rounded-2xl border border-slate-800 bg-slate-900/55 p-4 shadow-xl shadow-slate-950/20">
      <div className="grid gap-3 lg:grid-cols-[1.5fr_1fr_1fr_.8fr_auto]">
        <label className="text-sm text-slate-400">Rechercher
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Société ou ticker" className="mt-1.5 w-full rounded-xl border border-slate-700 bg-slate-950/80 px-3 py-2.5 text-slate-100 placeholder:text-slate-600"/>
        </label>
        <Select label="Statut" value={status} onChange={(value) => setStatus(value as 'ALL' | CompanyStatus)} options={statusOptions}/>
        <Select label="Zone de prix" value={entryZone} onChange={(value) => setEntryZone(value as typeof entryZone)} options={[['ALL','Toutes'],['AT_OR_BELOW_O90','Seuil 10 % atteint'],['WITHIN_5','À moins de 5 %'],['WITHIN_10','À 5–10 %'],['WITHIN_20','À 10–20 %'],['ABOVE_20','À plus de 20 %'],['UNCALIBRATED','Non calibrée']]}/>
        <NumericFilter label="Score marché min." value={marketScoreMin} onChange={setMarketScoreMin} placeholder="ex. 70"/>
        <button onClick={reset} className="self-end rounded-xl border border-slate-700 px-4 py-2.5 text-sm text-slate-400 transition hover:border-slate-600 hover:bg-slate-800 hover:text-white">Réinitialiser</button>
      </div>

      <details className="group mt-3 border-t border-slate-800 pt-3">
        <summary className="cursor-pointer list-none text-sm font-medium text-slate-400 hover:text-slate-200">Filtres avancés <span className="ml-1 text-slate-600 group-open:rotate-180">⌄</span></summary>
        <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <Select label="Pays" value={country} onChange={setCountry} options={[['ALL','Tous'],...countries.map((value) => [value,value] as [string,string])]}/>
          <Select label="Secteur" value={sector} onChange={setSector} options={[['ALL','Tous'],...sectors.map((value) => [value,value] as [string,string])]}/>
          <Select label="Industrie" value={industryGroup} onChange={setIndustryGroup} options={[['ALL','Toutes'],...industryGroups.map((value) => [value,value] as [string,string])]}/>
          <Select label="Modèle économique" value={businessModel} onChange={setBusinessModel} options={[['ALL','Tous'],...businessModels.map((value) => [value,value] as [string,string])]}/>
          <Select label="Fraîcheur du cours" value={freshness} onChange={(value) => setFreshness(value as typeof freshness)} options={[['ALL','Toutes'],['FRESH','Récentes'],['STALE','À actualiser']]}/>
          <NumericFilter label="OQS min." value={qualityMin} onChange={setQualityMin} placeholder="ex. 80"/>
          <NumericFilter label="OVS marché min." value={valuationMin} onChange={setValuationMin} placeholder="ex. 60"/>
          <NumericFilter label="Distance seuil max. %" value={distanceMax} onChange={setDistanceMax} placeholder="ex. 20"/>
        </div>
      </details>
    </div>

    <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/55">
      <div className="overflow-x-auto">
        <table className="min-w-[1240px] w-full text-left text-sm">
          <thead className="sticky top-0 z-10 border-b border-slate-800 bg-[#08121f] text-[11px] uppercase tracking-[0.08em] text-slate-500">
            <tr>
              <th className="px-4 py-3.5">Société</th>
              <th className="px-4 py-3.5"><button onClick={() => setSort('price')}>Cours{sortMark('price')}</button></th>
              <th className="px-4 py-3.5">Dernière MAJ</th>
              <th className="px-4 py-3.5"><button onClick={() => setSort('quality')}>Qualité OQS{sortMark('quality')}</button></th>
              <th className="px-4 py-3.5"><button onClick={() => setSort('valuation')}>Valorisation{sortMark('valuation')}</button></th>
              <th className="px-4 py-3.5"><button onClick={() => setSort('marketScore')}>Score marché{sortMark('marketScore')}</button></th>
              <th className="px-4 py-3.5">Seuil 10 %</th>
              <th className="px-4 py-3.5"><button onClick={() => setSort('distance')}>Distance{sortMark('distance')}</button></th>
              <th className="px-4 py-3.5">Statut</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80">{filtered.map((row) => {
            const priceProps = { currency: row.currency, quoteUnit: row.quote_unit, priceDecimals: row.price_decimals };
            const updateFreshness = getFreshness(row.price_as_of);
            return <tr key={row.id} tabIndex={0} role="link" onClick={() => router.push(`/company/${row.slug}`)} onKeyDown={(event) => { if (event.key === 'Enter') router.push(`/company/${row.slug}`); }} className="cursor-pointer transition hover:bg-cyan-950/10 focus:bg-cyan-950/10 focus:outline-none">
              <td className="px-4 py-4">
                <div className="font-semibold text-slate-100">{row.name}</div>
                <div className="mt-1 font-mono text-xs text-slate-500">{row.ticker} · {row.exchange}</div>
              </td>
              <td className="px-4 py-4">
                <div className="font-semibold text-white"><PriceDisplay value={row.price} {...priceProps}/></div>
                {row.market_price_change_vs_reference_pct !== null && row.market_price_change_vs_reference_pct !== undefined ? <div className={`mt-1 font-mono text-[11px] ${row.market_price_change_vs_reference_pct <= 0 ? 'text-emerald-300' : 'text-amber-200'}`}>{row.market_price_change_vs_reference_pct >= 0 ? '+' : ''}{row.market_price_change_vs_reference_pct.toFixed(1)} % réf.</div> : null}
              </td>
              <td className="px-4 py-4">
                <div className={`text-xs ${updateFreshness.stale ? 'text-amber-200' : 'text-emerald-300'}`}>{updateFreshness.stale ? 'À actualiser' : 'À jour'}</div>
                <div className="mt-1 text-[11px] text-slate-500">{formatUpdate(row.price_as_of)}</div>
              </td>
              <td className="px-4 py-4"><ScoreBadge score={row.business_quality_score} compact/></td>
              <td className="px-4 py-4">
                <ScoreBadge score={row.display_valuation_score} compact/>
                <div className="mt-1 text-[10px] text-slate-600">{row.market_score_is_live ? 'marché' : 'certifié'}</div>
              </td>
              <td className="px-4 py-4">
                <ScoreBadge score={row.display_investment_score} compact/>
                {row.market_score_is_live && row.investment_score !== null ? <div className="mt-1 text-[10px] text-slate-600">snapshot {row.investment_score.toFixed(0)}</div> : null}
              </td>
              <td className="px-4 py-4 text-slate-200">{row.price_o90 === null ? <span className="text-slate-600">Non calibré</span> : <PriceDisplay value={row.price_o90} {...priceProps}/>}</td>
              <td className="px-4 py-4"><OroTitanDistance value={row.distance_o90_pct} compact/></td>
              <td className="px-4 py-4"><div className="space-y-2"><CompanyStatusBadge status={row.status}/><EntryZoneBadge zone={row.entry_zone}/></div></td>
            </tr>;
          })}</tbody>
        </table>
      </div>
      {filtered.length === 0 ? <div className="p-10 text-center text-sm text-slate-500">Aucune société ne correspond aux filtres.</div> : null}
    </div>

    <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-slate-500">
      <span>{filtered.length} société{filtered.length > 1 ? 's' : ''} affichée{filtered.length > 1 ? 's' : ''}.</span>
      <span>OVS et Score marché = projection indicative au dernier cours disponible · OQS = certifié.</span>
    </div>
  </div>;
}

function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: Array<[string,string]> }) {
  return <label className="text-sm text-slate-400">{label}<select value={value} onChange={(event) => onChange(event.target.value)} className="mt-1.5 w-full rounded-xl border border-slate-700 bg-slate-950/80 px-3 py-2.5 text-slate-100">{options.map(([optionValue, optionLabel]) => <option key={optionValue} value={optionValue}>{optionLabel}</option>)}</select></label>;
}

function NumericFilter({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (value: string) => void; placeholder: string }) {
  return <label className="text-sm text-slate-400">{label}<input value={value} onChange={(event) => onChange(event.target.value)} type="number" step="0.1" placeholder={placeholder} className="mt-1.5 w-full rounded-xl border border-slate-700 bg-slate-950/80 px-3 py-2.5 text-slate-100 placeholder:text-slate-600"/></label>;
}
