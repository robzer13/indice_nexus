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

type SortKey = 'distance' | 'score' | 'fairValue' | 'analysisDate';
type SortDirection = 'asc' | 'desc';
type Row = CompanyState & { distance_o90_pct: number | null; entry_zone: EntryZone; stale: boolean };

function compareNullableNumber(a: number | null, b: number | null, direction: SortDirection): number {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return direction === 'asc' ? a - b : b - a;
}

function compareRow(a: Row, b: Row, key: SortKey, direction: SortDirection): number {
  if (key === 'distance') return compareNullableNumber(a.distance_o90_pct, b.distance_o90_pct, direction);
  if (key === 'score') return compareNullableNumber(a.orotitan_score, b.orotitan_score, direction);
  if (key === 'fairValue') return compareNullableNumber(a.fair_value_base, b.fair_value_base, direction);
  const left = a.analysis_date ?? '';
  const right = b.analysis_date ?? '';
  return direction === 'asc' ? left.localeCompare(right) : right.localeCompare(left);
}

export function ScreenerTable({ companies }: { companies: CompanyState[] }) {
  const router = useRouter();
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState<'ALL' | CompanyStatus>('ALL');
  const [quality, setQuality] = useState<'ALL' | 'TRUE' | 'FALSE' | 'NULL'>('ALL');
  const [country, setCountry] = useState('ALL');
  const [sector, setSector] = useState('ALL');
  const [entryZone, setEntryZone] = useState<'ALL' | EntryZone>('ALL');
  const [calibration, setCalibration] = useState<'ALL' | 'CALIBRATED' | 'UNCALIBRATED'>('ALL');
  const [freshness, setFreshness] = useState<'ALL' | 'FRESH' | 'STALE'>('ALL');
  const [scoreMin, setScoreMin] = useState('');
  const [distanceMin, setDistanceMin] = useState('');
  const [distanceMax, setDistanceMax] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('distance');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [secondarySort, setSecondarySort] = useState<'NONE' | SortKey>('score');

  const rows = useMemo<Row[]>(() => companies.map((company) => {
    const distance = getDistanceO90(company.price, company.price_o90);
    return {
      ...company,
      distance_o90_pct: distance,
      entry_zone: getEntryZone(distance),
      stale: getFreshness(company.price_as_of).stale,
    };
  }), [companies]);

  const countries = useMemo(() => [...new Set(rows.map((row) => row.country).filter((value): value is string => Boolean(value)))].sort(), [rows]);
  const sectors = useMemo(() => [...new Set(rows.map((row) => row.sector).filter((value): value is string => Boolean(value)))].sort(), [rows]);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const minScore = scoreMin.trim() ? Number(scoreMin) : null;
    const minDistance = distanceMin.trim() ? Number(distanceMin) : null;
    const maxDistance = distanceMax.trim() ? Number(distanceMax) : null;

    return rows
      .filter((row) => !needle || row.name.toLowerCase().includes(needle) || row.ticker.toLowerCase().includes(needle))
      .filter((row) => status === 'ALL' || row.status === status)
      .filter((row) => quality === 'ALL' || (quality === 'TRUE' && row.quality_orotitan === true) || (quality === 'FALSE' && row.quality_orotitan === false) || (quality === 'NULL' && row.quality_orotitan === null))
      .filter((row) => country === 'ALL' || row.country === country)
      .filter((row) => sector === 'ALL' || row.sector === sector)
      .filter((row) => entryZone === 'ALL' || row.entry_zone === entryZone)
      .filter((row) => calibration === 'ALL' || (calibration === 'CALIBRATED' && row.distance_o90_pct !== null) || (calibration === 'UNCALIBRATED' && row.distance_o90_pct === null))
      .filter((row) => freshness === 'ALL' || (freshness === 'FRESH' && !row.stale) || (freshness === 'STALE' && row.stale))
      .filter((row) => minScore === null || (row.orotitan_score !== null && row.orotitan_score >= minScore))
      .filter((row) => minDistance === null || (row.distance_o90_pct !== null && row.distance_o90_pct >= minDistance))
      .filter((row) => maxDistance === null || (row.distance_o90_pct !== null && row.distance_o90_pct <= maxDistance))
      .sort((a, b) => {
        const primary = compareRow(a, b, sortKey, sortDirection);
        if (primary !== 0 || secondarySort === 'NONE' || secondarySort === sortKey) return primary;
        return compareRow(a, b, secondarySort, 'desc');
      });
  }, [rows, search, status, quality, country, sector, entryZone, calibration, freshness, scoreMin, distanceMin, distanceMax, sortKey, sortDirection, secondarySort]);

  function setSort(next: SortKey) {
    if (next === sortKey) setSortDirection((current) => current === 'asc' ? 'desc' : 'asc');
    else {
      setSortKey(next);
      setSortDirection('desc');
    }
  }

  const sortMark = (key: SortKey) => key === sortKey ? (sortDirection === 'asc' ? ' ↑' : ' ↓') : '';

  return <div className="space-y-4">
    <div className="space-y-3 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <label className="text-sm text-slate-400">Recherche<input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Société ou ticker" className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"/></label>
        <Select label="Statut" value={status} onChange={(value) => setStatus(value as 'ALL' | CompanyStatus)} options={[['ALL','Tous'],['OROTITAN','OroTitan'],['FINALIST','Finalist'],['PRICE_WAIT','Price wait'],['TIER_1','Tier 1'],['WATCHLIST','Watchlist'],['REJECTED','Rejected']]}/>
        <Select label="Qualité OroTitan" value={quality} onChange={(value) => setQuality(value as typeof quality)} options={[['ALL','Toutes'],['TRUE','Structurellement OroTitan'],['FALSE','Non OroTitan'],['NULL','Non renseigné']]}/>
        <Select label="Zone d’entrée" value={entryZone} onChange={(value) => setEntryZone(value as typeof entryZone)} options={[['ALL','Toutes'],['AT_OR_BELOW_O90','O90 atteint'],['WITHIN_5','À moins de 5 %'],['WITHIN_10','À 5–10 %'],['WITHIN_20','À 10–20 %'],['ABOVE_20','À plus de 20 %'],['UNCALIBRATED','Non calibré']]}/>
        <Select label="Pays" value={country} onChange={setCountry} options={[['ALL','Tous'],...countries.map((value) => [value,value] as [string,string])]}/>
        <Select label="Secteur" value={sector} onChange={setSector} options={[['ALL','Tous'],...sectors.map((value) => [value,value] as [string,string])]}/>
        <Select label="Calibration O90" value={calibration} onChange={(value) => setCalibration(value as typeof calibration)} options={[['ALL','Toutes'],['CALIBRATED','Calibrées'],['UNCALIBRATED','Non calibrées']]}/>
        <Select label="Fraîcheur cours" value={freshness} onChange={(value) => setFreshness(value as typeof freshness)} options={[['ALL','Toutes'],['FRESH','Récentes'],['STALE','Périmées']]}/>
      </div>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
        <NumericFilter label="Score min." value={scoreMin} onChange={setScoreMin} placeholder="ex. 80"/>
        <NumericFilter label="Distance min. %" value={distanceMin} onChange={setDistanceMin} placeholder="ex. -20"/>
        <NumericFilter label="Distance max. %" value={distanceMax} onChange={setDistanceMax} placeholder="ex. 5"/>
        <Select label="Tri secondaire" value={secondarySort} onChange={(value) => setSecondarySort(value as typeof secondarySort)} options={[['NONE','Aucun'],['score','Score'],['distance','Distance O90'],['fairValue','Fair value'],['analysisDate','Date analyse']]}/>
        <button onClick={() => { setSearch(''); setStatus('ALL'); setQuality('ALL'); setCountry('ALL'); setSector('ALL'); setEntryZone('ALL'); setCalibration('ALL'); setFreshness('ALL'); setScoreMin(''); setDistanceMin(''); setDistanceMax(''); }} className="self-end rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-400 hover:bg-slate-800">Réinitialiser</button>
      </div>
    </div>

    <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/60">
      <table className="min-w-[1200px] w-full text-left text-sm">
        <thead className="border-b border-slate-800 bg-slate-900/80 text-xs uppercase tracking-wide text-slate-500"><tr>
          <th className="px-4 py-3">Société</th><th className="px-4 py-3">Ticker</th><th className="px-4 py-3">Statut</th><th className="px-4 py-3">Cours</th><th className="px-4 py-3"><button onClick={() => setSort('fairValue')}>Fair value{sortMark('fairValue')}</button></th><th className="px-4 py-3"><button onClick={() => setSort('score')}>Score{sortMark('score')}</button></th><th className="px-4 py-3">O90</th><th className="px-4 py-3"><button onClick={() => setSort('distance')}>Distance{sortMark('distance')}</button></th><th className="px-4 py-3">Zone</th><th className="px-4 py-3"><button onClick={() => setSort('analysisDate')}>Analyse{sortMark('analysisDate')}</button></th>
        </tr></thead>
        <tbody className="divide-y divide-slate-800">{filtered.map((row) => {
          const priceProps = { currency: row.currency, quoteUnit: row.quote_unit, priceDecimals: row.price_decimals };
          return <tr key={row.id} tabIndex={0} role="link" onClick={() => router.push(`/company/${row.slug}`)} onKeyDown={(event) => { if (event.key === 'Enter') router.push(`/company/${row.slug}`); }} className="cursor-pointer transition hover:bg-slate-900/70 focus:bg-slate-900/70 focus:outline-none">
            <td className="px-4 py-4 font-medium text-slate-100">{row.name}</td><td className="px-4 py-4 font-mono text-slate-400">{row.ticker}</td><td className="px-4 py-4"><CompanyStatusBadge status={row.status}/></td><td className="px-4 py-4 text-slate-100"><PriceDisplay value={row.price} {...priceProps}/></td><td className="px-4 py-4 text-slate-200"><PriceDisplay value={row.fair_value_base} {...priceProps}/></td><td className="px-4 py-4"><ScoreBadge score={row.orotitan_score}/></td><td className="px-4 py-4 text-slate-200">{row.price_o90 === null ? <span className="text-slate-500">Non calibré</span> : <PriceDisplay value={row.price_o90} {...priceProps}/>}</td><td className="px-4 py-4"><OroTitanDistance value={row.distance_o90_pct} compact/></td><td className="px-4 py-4"><EntryZoneBadge zone={row.entry_zone}/></td><td className="px-4 py-4 font-mono text-xs text-slate-400">{row.analysis_date ?? '—'}</td>
          </tr>;
        })}</tbody>
      </table>
      {filtered.length === 0 ? <div className="p-8 text-center text-sm text-slate-500">Aucune société ne correspond aux filtres.</div> : null}
    </div>
    <div className="text-xs text-slate-500">{filtered.length} société{filtered.length > 1 ? 's' : ''} affichée{filtered.length > 1 ? 's' : ''}. Les NULL restent hors tri numérique.</div>
  </div>;
}

function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: Array<[string,string]> }) {
  return <label className="text-sm text-slate-400">{label}<select value={value} onChange={(event) => onChange(event.target.value)} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100">{options.map(([optionValue, optionLabel]) => <option key={optionValue} value={optionValue}>{optionLabel}</option>)}</select></label>;
}

function NumericFilter({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (value: string) => void; placeholder: string }) {
  return <label className="text-sm text-slate-400">{label}<input value={value} onChange={(event) => onChange(event.target.value)} type="number" step="0.1" placeholder={placeholder} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"/></label>;
}
