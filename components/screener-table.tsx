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

type SortKey = 'investment' | 'valuation' | 'return' | 'distance' | 'analysisDate';
type SortDirection = 'asc' | 'desc';
type Row = CompanyState & { distance_o90_pct: number | null; entry_zone: EntryZone; stale: boolean };

function liveInvestment(row: CompanyState): number | null {
  return row.live_investment_score ?? row.investment_score;
}

function liveValuation(row: CompanyState): number | null {
  return row.live_valuation_score ?? row.valuation_score;
}

function liveReturn(row: CompanyState): number | null {
  return row.live_primary_expected_return ?? null;
}

function compareNullableNumber(a: number | null, b: number | null, direction: SortDirection): number {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return direction === 'asc' ? a - b : b - a;
}

function compareRow(a: Row, b: Row, key: SortKey, direction: SortDirection): number {
  if (key === 'investment') return compareNullableNumber(liveInvestment(a), liveInvestment(b), direction);
  if (key === 'valuation') return compareNullableNumber(liveValuation(a), liveValuation(b), direction);
  if (key === 'return') return compareNullableNumber(liveReturn(a), liveReturn(b), direction);
  if (key === 'distance') return compareNullableNumber(a.distance_o90_pct, b.distance_o90_pct, direction);
  const left = a.analysis_date ?? '';
  const right = b.analysis_date ?? '';
  return direction === 'asc' ? left.localeCompare(right) : right.localeCompare(left);
}

function fmtPct(value: number | null, digits = 1): string {
  if (value === null || !Number.isFinite(value)) return '—';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(digits)} %`;
}

function fmtPriceTime(value: string | null): string {
  if (!value) return 'Cours non daté';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Date inconnue';
  return date.toLocaleString('fr-FR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
}

export function ScreenerTable({ companies }: { companies: CompanyState[] }) {
  const router = useRouter();
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState<'ALL' | CompanyStatus>('ALL');
  const [sector, setSector] = useState('ALL');
  const [entryZone, setEntryZone] = useState<'ALL' | EntryZone>('ALL');
  const [freshness, setFreshness] = useState<'ALL' | 'FRESH' | 'STALE'>('ALL');
  const [country, setCountry] = useState('ALL');
  const [industryGroup, setIndustryGroup] = useState('ALL');
  const [pea, setPea] = useState<'ALL' | 'YES' | 'NO' | 'UNKNOWN'>('ALL');
  const [scoreMin, setScoreMin] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('investment');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const rows = useMemo<Row[]>(() => companies.map((company) => {
    const distance = getDistanceO90(company.price, company.price_o90);
    return {
      ...company,
      distance_o90_pct: distance,
      entry_zone: getEntryZone(distance),
      stale: getFreshness(company.price_as_of).stale,
    };
  }), [companies]);

  const sectors = useMemo(() => [...new Set(rows.map((row) => row.sector).filter((value): value is string => Boolean(value)))].sort(), [rows]);
  const countries = useMemo(() => [...new Set(rows.map((row) => row.country).filter((value): value is string => Boolean(value)))].sort(), [rows]);
  const industryGroups = useMemo(() => [...new Set(rows.map((row) => row.industry_group).filter((value): value is string => Boolean(value)))].sort(), [rows]);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const minScore = scoreMin.trim() ? Number(scoreMin) : null;
    return rows
      .filter((row) => !needle || row.name.toLowerCase().includes(needle) || row.ticker.toLowerCase().includes(needle))
      .filter((row) => status === 'ALL' || row.status === status)
      .filter((row) => sector === 'ALL' || row.sector === sector)
      .filter((row) => country === 'ALL' || row.country === country)
      .filter((row) => industryGroup === 'ALL' || row.industry_group === industryGroup)
      .filter((row) => pea === 'ALL' || row.pea_eligibility === pea)
      .filter((row) => entryZone === 'ALL' || row.entry_zone === entryZone)
      .filter((row) => freshness === 'ALL' || (freshness === 'FRESH' && !row.stale) || (freshness === 'STALE' && row.stale))
      .filter((row) => minScore === null || (liveInvestment(row) !== null && liveInvestment(row)! >= minScore))
      .sort((a, b) => compareRow(a, b, sortKey, sortDirection));
  }, [rows, search, status, sector, country, industryGroup, pea, entryZone, freshness, scoreMin, sortKey, sortDirection]);

  function setSort(next: SortKey) {
    if (next === sortKey) setSortDirection((current) => current === 'asc' ? 'desc' : 'asc');
    else {
      setSortKey(next);
      setSortDirection('desc');
    }
  }

  const sortMark = (key: SortKey) => key === sortKey ? (sortDirection === 'asc' ? ' ↑' : ' ↓') : '';

  return <div className="space-y-4">
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 shadow-panel">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <label className="text-sm text-slate-400">Recherche
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Société ou ticker" className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100 outline-none transition focus:border-cyan-700"/>
        </label>
        <Select label="Statut" value={status} onChange={(value) => setStatus(value as 'ALL' | CompanyStatus)} options={[['ALL','Tous'],['OROTITAN','OroTitan'],['FINALIST','Finaliste'],['PRICE_WAIT','Attente de prix'],['WATCHLIST','À surveiller'],['REJECTED','Écartée']]}/>
        <Select label="Zone d’entrée" value={entryZone} onChange={(value) => setEntryZone(value as typeof entryZone)} options={[['ALL','Toutes'],['AT_OR_BELOW_O90','Seuil H atteint'],['WITHIN_5','À moins de 5 %'],['WITHIN_10','À 5–10 %'],['WITHIN_20','À 10–20 %'],['ABOVE_20','À plus de 20 %'],['UNCALIBRATED','Non calibrée']]}/>
        <Select label="Fraîcheur du cours" value={freshness} onChange={(value) => setFreshness(value as typeof freshness)} options={[['ALL','Tous'],['FRESH','Récent'],['STALE','Ancien']]}/>
      </div>

      <details className="mt-3 rounded-xl border border-slate-800/80 bg-slate-950/35 p-3">
        <summary className="cursor-pointer text-sm font-medium text-slate-300">Filtres avancés</summary>
        <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          <Select label="Secteur" value={sector} onChange={setSector} options={[['ALL','Tous'],...sectors.map((value) => [value,value] as [string,string])]}/>
          <Select label="Pays" value={country} onChange={setCountry} options={[['ALL','Tous'],...countries.map((value) => [value,value] as [string,string])]}/>
          <Select label="Industrie" value={industryGroup} onChange={setIndustryGroup} options={[['ALL','Toutes'],...industryGroups.map((value) => [value,value] as [string,string])]}/>
          <Select label="PEA" value={pea} onChange={(value) => setPea(value as typeof pea)} options={[['ALL','Tous'],['YES','Éligible'],['NO','Non éligible'],['UNKNOWN','Inconnu']]}/>
          <NumericFilter label="Score investissement live min." value={scoreMin} onChange={setScoreMin} placeholder="ex. 70"/>
        </div>
      </details>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-500">
        <span>{filtered.length} société{filtered.length > 1 ? 's' : ''} affichée{filtered.length > 1 ? 's' : ''}</span>
        <button onClick={() => {
          setSearch(''); setStatus('ALL'); setSector('ALL'); setEntryZone('ALL'); setFreshness('ALL'); setCountry('ALL'); setIndustryGroup('ALL'); setPea('ALL'); setScoreMin(''); setSortKey('investment'); setSortDirection('desc');
        }} className="rounded-lg border border-slate-700 px-3 py-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200">Réinitialiser</button>
      </div>
    </div>

    <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-950/65 shadow-panel">
      <table className="min-w-[1180px] w-full text-left text-sm">
        <thead className="border-b border-slate-800 bg-slate-900/90 text-[11px] uppercase tracking-[0.08em] text-slate-500">
          <tr>
            <th className="px-4 py-3">Société</th>
            <th className="px-4 py-3">Cours actuel</th>
            <th className="px-4 py-3">OQS</th>
            <th className="px-4 py-3"><button onClick={() => setSort('valuation')}>OVS live{sortMark('valuation')}</button></th>
            <th className="px-4 py-3"><button onClick={() => setSort('return')}>Rendement live{sortMark('return')}</button></th>
            <th className="px-4 py-3"><button onClick={() => setSort('investment')}>Investissement live{sortMark('investment')}</button></th>
            <th className="px-4 py-3">Seuil H</th>
            <th className="px-4 py-3"><button onClick={() => setSort('distance')}>Écart seuil H{sortMark('distance')}</button></th>
            <th className="px-4 py-3">Signal</th>
            <th className="px-4 py-3"><button onClick={() => setSort('analysisDate')}>Analyse{sortMark('analysisDate')}</button></th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">{filtered.map((row) => {
          const priceProps = { currency: row.currency, quoteUnit: row.quote_unit, priceDecimals: row.price_decimals };
          const freshnessState = getFreshness(row.price_as_of);
          const liveOvs = liveValuation(row);
          const publishedOvs = row.snapshot_valuation_score ?? row.valuation_score;
          const liveInvestmentScore = liveInvestment(row);
          const publishedInvestment = row.snapshot_investment_score ?? row.investment_score;
          return <tr key={row.id} tabIndex={0} role="link" onClick={() => router.push(`/company/${row.slug}`)} onKeyDown={(event) => { if (event.key === 'Enter') router.push(`/company/${row.slug}`); }} className="cursor-pointer transition hover:bg-slate-900/75 focus:bg-slate-900/75 focus:outline-none">
            <td className="px-4 py-4">
              <div className="font-semibold text-slate-100">{row.name}</div>
              <div className="mt-1 font-mono text-xs text-slate-500">{row.ticker} · {row.exchange}</div>
            </td>
            <td className="px-4 py-4">
              <div className="font-semibold text-slate-100"><PriceDisplay value={row.price} {...priceProps}/></div>
              <div className={`mt-1 text-[11px] ${freshnessState.stale ? 'text-amber-300' : 'text-emerald-300'}`}>{freshnessState.stale ? '● Ancien' : '● À jour'} · {fmtPriceTime(row.price_as_of)}</div>
            </td>
            <td className="px-4 py-4"><ScoreBadge score={row.business_quality_score}/><div className="mt-1 text-[10px] text-slate-600">fixe</div></td>
            <td className="px-4 py-4"><ScoreBadge score={liveOvs}/><div className="mt-1 text-[10px] text-slate-600">publié {publishedOvs ?? '—'}</div></td>
            <td className="px-4 py-4"><div className="font-mono font-semibold text-slate-100">{fmtPct(liveReturn(row))}</div><div className="mt-1 text-[10px] text-slate-600">annualisé</div></td>
            <td className="px-4 py-4"><ScoreBadge score={liveInvestmentScore}/><div className="mt-1 text-[10px] text-slate-600">publié {publishedInvestment ?? '—'}</div></td>
            <td className="px-4 py-4 text-slate-200">{row.price_o90 === null ? <span className="text-slate-500">Non calibré</span> : <PriceDisplay value={row.price_o90} {...priceProps}/>}</td>
            <td className="px-4 py-4"><OroTitanDistance value={row.distance_o90_pct} compact/></td>
            <td className="px-4 py-4"><div className="space-y-2"><EntryZoneBadge zone={row.entry_zone}/><div><CompanyStatusBadge status={row.status}/></div></div></td>
            <td className="px-4 py-4 font-mono text-xs text-slate-400">{row.analysis_date ?? '—'}</td>
          </tr>;
        })}</tbody>
      </table>
      {filtered.length === 0 ? <div className="p-10 text-center text-sm text-slate-500">Aucune société ne correspond aux filtres.</div> : null}
    </div>

    <p className="text-xs leading-5 text-slate-500">L’OQS reste celui de l’analyse certifiée. L’OVS et le score d’investissement « live » sont recalculés au dernier cours disponible en conservant les hypothèses économiques du snapshot. Ils n’écrasent jamais le score publié.</p>
  </div>;
}

function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: Array<[string,string]> }) {
  return <label className="text-sm text-slate-400">{label}<select value={value} onChange={(event) => onChange(event.target.value)} className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100 outline-none transition focus:border-cyan-700">{options.map(([optionValue, optionLabel]) => <option key={optionValue} value={optionValue}>{optionLabel}</option>)}</select></label>;
}

function NumericFilter({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (value: string) => void; placeholder: string }) {
  return <label className="text-sm text-slate-400">{label}<input value={value} onChange={(event) => onChange(event.target.value)} type="number" step="0.1" placeholder={placeholder} className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100 outline-none transition focus:border-cyan-700"/></label>;
}
