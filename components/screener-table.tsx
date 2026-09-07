'use client';

import { useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { CompanyStatusBadge } from '@/components/company-status-badge';
import { EntryZoneBadge } from '@/components/entry-zone-badge';
import { OroTitanDistance } from '@/components/orotitan-distance';
import { PriceDisplay } from '@/components/price-display';
import { ScoreBadge } from '@/components/score-badge';
import { Panel } from '@/components/ui/panel';
import { getDistanceO90 } from '@/lib/domain/distance';
import { getEntryZone, type EntryZone } from '@/lib/domain/entry-zone';
import { getFreshness } from '@/lib/domain/freshness';
import { getFairValueUpsidePct } from '@/lib/domain/valuation-upside';
import type { CompanyState, CompanyStatus } from '@/lib/domain/types';

type SortKey = 'distance' | 'score' | 'fairValueUpside' | 'analysisDate';
type SortDirection = 'asc' | 'desc';
type Row = CompanyState & {
  distance_o90_pct: number | null;
  fair_value_upside_pct: number | null;
  entry_zone: EntryZone;
  stale: boolean;
};

function compareNullableNumber(a: number | null, b: number | null, direction: SortDirection): number {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return direction === 'asc' ? a - b : b - a;
}

function compareRow(a: Row, b: Row, key: SortKey, direction: SortDirection): number {
  if (key === 'distance') return compareNullableNumber(a.distance_o90_pct, b.distance_o90_pct, direction);
  if (key === 'score') return compareNullableNumber(a.orotitan_score, b.orotitan_score, direction);
  if (key === 'fairValueUpside') return compareNullableNumber(a.fair_value_upside_pct, b.fair_value_upside_pct, direction);
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
      fair_value_upside_pct: getFairValueUpsidePct(company.price, company.fair_value_base),
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

  return <div className="space-y-5">
    <Panel as="section" className="space-y-4 p-4 sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-700/50 pb-3"><div><div className="text-xs font-semibold uppercase tracking-[0.16em] text-ink-muted">Contrôle du screener</div><p className="mt-1 text-sm text-ink-secondary">Affinez la base active sans perdre la lecture des seuils OroTitan.</p></div><span className="font-mono text-xs text-ink-muted">{filtered.length} résultat{filtered.length > 1 ? 's' : ''}</span></div>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <label className="text-sm font-medium text-ink-secondary">Recherche<input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Société ou ticker" className="mt-1 w-full rounded-control border border-slate-600/70 bg-cockpit-bg px-3 py-2.5 text-sm text-ink-primary placeholder:text-ink-muted hover:border-slate-500 focus:border-state-accent focus:bg-cockpit-panel"/></label>
        <Select label="Statut" value={status} onChange={(value) => setStatus(value as 'ALL' | CompanyStatus)} options={[['ALL','Tous'],['OROTITAN','OroTitan'],['FINALIST','Finalist'],['PRICE_WAIT','Price wait'],['TIER_1','Tier 1'],['WATCHLIST','Watchlist'],['REJECTED','Rejected']]}/>
        <Select label="Qualité OroTitan" value={quality} onChange={(value) => setQuality(value as typeof quality)} options={[['ALL','Toutes'],['TRUE','Structurellement OroTitan'],['FALSE','Non OroTitan'],['NULL','Non renseigné']]}/>
        <Select label="Zone d’entrée" value={entryZone} onChange={(value) => setEntryZone(value as typeof entryZone)} options={[['ALL','Toutes'],['AT_OR_BELOW_O90','O90 atteint'],['WITHIN_5','À moins de 5 %'],['WITHIN_10','À 5–10 %'],['WITHIN_20','À 10–20 %'],['ABOVE_20','À plus de 20 %'],['UNCALIBRATED','Non calibré']]}/>
        <Select label="Pays" value={country} onChange={setCountry} options={[['ALL','Tous'],...countries.map((value) => [value,value] as [string,string])]}/>
        <Select label="Secteur" value={sector} onChange={setSector} options={[['ALL','Tous'],...sectors.map((value) => [value,value] as [string,string])]}/>
        <Select label="Calibration O90" value={calibration} onChange={(value) => setCalibration(value as typeof calibration)} options={[['ALL','Toutes'],['CALIBRATED','Calibrées'],['UNCALIBRATED','Non calibrées']]}/>
        <Select label="Fraîcheur cours" value={freshness} onChange={(value) => setFreshness(value as typeof freshness)} options={[['ALL','Toutes'],['FRESH','Récentes'],['STALE','Périmées']]}/>
      </div>
      <div className="grid gap-3 border-t border-slate-700/50 pt-4 md:grid-cols-2 xl:grid-cols-5">
        <NumericFilter label="Score min." value={scoreMin} onChange={setScoreMin} placeholder="ex. 80"/>
        <NumericFilter label="Distance min. %" value={distanceMin} onChange={setDistanceMin} placeholder="ex. -20"/>
        <NumericFilter label="Distance max. %" value={distanceMax} onChange={setDistanceMax} placeholder="ex. 5"/>
        <Select label="Tri secondaire" value={secondarySort} onChange={(value) => setSecondarySort(value as typeof secondarySort)} options={[['NONE','Aucun'],['score','Score'],['distance','Distance O90'],['fairValueUpside','Upside FV'],['analysisDate','Date analyse']]}/>
        <button type="button" onClick={() => {
          setSearch('');
          setStatus('ALL');
          setQuality('ALL');
          setCountry('ALL');
          setSector('ALL');
          setEntryZone('ALL');
          setCalibration('ALL');
          setFreshness('ALL');
          setScoreMin('');
          setDistanceMin('');
          setDistanceMax('');
          setSortKey('distance');
          setSortDirection('desc');
          setSecondarySort('score');
        }} className="self-end rounded-control border border-state-accent/40 bg-cockpit-active px-3 py-2.5 text-sm font-semibold text-state-accent transition-colors hover:border-state-accent hover:bg-cockpit-hover">Réinitialiser</button>
      </div>
    </Panel>

    <div className="overflow-x-auto rounded-panel border border-slate-700/60 bg-cockpit-bg shadow-panel-soft">
      <table className="min-w-[1360px] w-full text-left text-[13px]">
        <thead className="border-b border-slate-700/70 bg-cockpit-panel text-[11px] uppercase tracking-[0.14em] text-ink-muted"><tr>
          <th className="w-[19%] px-4 py-3.5">Société</th><th className="px-4 py-3.5">Ticker</th><th className="px-4 py-3.5">Statut</th><th className="px-4 py-3.5">Cours</th><th className="px-4 py-3.5">Fair value</th><th className="px-4 py-3.5"><button type="button" className="rounded px-1 py-1 text-left hover:text-ink-primary" onClick={() => setSort('fairValueUpside')}>Upside FV{sortMark('fairValueUpside')}</button></th><th className="px-4 py-3.5"><button type="button" className="rounded px-1 py-1 text-left hover:text-ink-primary" onClick={() => setSort('score')}>Score{sortMark('score')}</button></th><th className="px-4 py-3.5">O90</th><th className="px-4 py-3.5"><button type="button" className="rounded px-1 py-1 text-left text-state-accent hover:text-ink-primary" onClick={() => setSort('distance')}>Distance{sortMark('distance')}</button></th><th className="px-4 py-3.5">Zone</th><th className="px-4 py-3.5"><button type="button" className="rounded px-1 py-1 text-left hover:text-ink-primary" onClick={() => setSort('analysisDate')}>Analyse{sortMark('analysisDate')}</button></th>
        </tr></thead>
        <tbody className="divide-y divide-slate-800/70">{filtered.map((row) => {
          const priceProps = { currency: row.currency, quoteUnit: row.quote_unit, priceDecimals: row.price_decimals };
          return <tr key={row.id} tabIndex={0} role="link" onClick={() => router.push(`/company/${row.slug}`)} onKeyDown={(event) => { if (event.key === 'Enter') router.push(`/company/${row.slug}`); }} className="cursor-pointer transition-colors duration-150 hover:bg-cockpit-hover/70 focus:bg-cockpit-active/80 focus:outline-none focus-visible:relative focus-visible:z-[1]">
            <td className="px-4 py-3.5 font-semibold text-ink-primary">{row.name}</td><td className="px-4 py-3.5 font-mono text-xs text-ink-muted">{row.ticker}</td><td className="px-4 py-3.5"><CompanyStatusBadge status={row.status}/></td><td className="px-4 py-3.5 text-base font-semibold text-ink-primary"><PriceDisplay value={row.price} {...priceProps}/></td><td className="px-4 py-3.5 text-ink-secondary"><PriceDisplay value={row.fair_value_base} {...priceProps}/></td><td className="px-4 py-3.5 font-mono text-sm font-semibold tabular-nums"><span className={row.fair_value_upside_pct === null ? 'text-ink-muted' : row.fair_value_upside_pct >= 0 ? 'text-state-success' : 'text-state-danger'}>{row.fair_value_upside_pct === null ? '—' : `${row.fair_value_upside_pct >= 0 ? '+' : ''}${row.fair_value_upside_pct.toFixed(1)}%`}</span></td><td className="px-4 py-3.5"><ScoreBadge score={row.orotitan_score}/></td><td className="px-4 py-3.5 text-base font-semibold text-ink-primary">{row.price_o90 === null ? <span className="text-ink-muted">Non calibré</span> : <PriceDisplay value={row.price_o90} {...priceProps}/>}</td><td className="px-4 py-3.5"><OroTitanDistance value={row.distance_o90_pct} compact/></td><td className="px-4 py-3.5"><EntryZoneBadge zone={row.entry_zone}/></td><td className="px-4 py-3.5 font-mono text-xs text-ink-muted">{row.analysis_date ?? '—'}</td>
          </tr>;
        })}</tbody>
      </table>
      {filtered.length === 0 ? <div className="p-10 text-center text-sm text-ink-muted">Aucune société ne correspond aux filtres.</div> : null}
    </div>
    <div className="text-xs text-ink-muted">{filtered.length} société{filtered.length > 1 ? 's' : ''} affichée{filtered.length > 1 ? 's' : ''}. Les NULL restent hors tri numérique.</div>
  </div>;
}

function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: Array<[string,string]> }) {
  return <label className="text-sm font-medium text-ink-secondary">{label}<select value={value} onChange={(event) => onChange(event.target.value)} className="mt-1 w-full rounded-control border border-slate-600/70 bg-cockpit-bg px-3 py-2.5 text-sm text-ink-primary hover:border-slate-500 focus:border-state-accent focus:bg-cockpit-panel">{options.map(([optionValue, optionLabel]) => <option key={optionValue} value={optionValue}>{optionLabel}</option>)}</select></label>;
}

function NumericFilter({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (value: string) => void; placeholder: string }) {
  return <label className="text-sm font-medium text-ink-secondary">{label}<input value={value} onChange={(event) => onChange(event.target.value)} type="number" step="0.1" placeholder={placeholder} className="mt-1 w-full rounded-control border border-slate-600/70 bg-cockpit-bg px-3 py-2.5 text-sm text-ink-primary placeholder:text-ink-muted hover:border-slate-500 focus:border-state-accent focus:bg-cockpit-panel"/></label>;
}
