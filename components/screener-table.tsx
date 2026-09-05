'use client';

import { useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { CompanyStatusBadge } from '@/components/company-status-badge';
import { OroTitanDistance } from '@/components/orotitan-distance';
import { PriceDisplay } from '@/components/price-display';
import { ScoreBadge } from '@/components/score-badge';
import { getDistanceO90 } from '@/lib/domain/distance';
import type { CompanyState, CompanyStatus } from '@/lib/domain/types';

type SortKey = 'distance' | 'score' | 'fairValue' | 'analysisDate';
type SortDirection = 'asc' | 'desc';

type Row = CompanyState & { distance_o90_pct: number | null };

function compareNullableNumber(a: number | null, b: number | null, direction: SortDirection): number {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return direction === 'asc' ? a - b : b - a;
}

export function ScreenerTable({ companies }: { companies: CompanyState[] }) {
  const router = useRouter();
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState<'ALL' | CompanyStatus>('ALL');
  const [quality, setQuality] = useState<'ALL' | 'TRUE' | 'FALSE' | 'NULL'>('ALL');
  const [sortKey, setSortKey] = useState<SortKey>('distance');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const rows = useMemo<Row[]>(() => companies.map((company) => ({ ...company, distance_o90_pct: getDistanceO90(company.price, company.price_o90) })), [companies]);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return rows
      .filter((row) => !needle || row.name.toLowerCase().includes(needle) || row.ticker.toLowerCase().includes(needle))
      .filter((row) => status === 'ALL' || row.status === status)
      .filter((row) => quality === 'ALL' || (quality === 'TRUE' && row.quality_orotitan === true) || (quality === 'FALSE' && row.quality_orotitan === false) || (quality === 'NULL' && row.quality_orotitan === null))
      .sort((a, b) => {
        if (sortKey === 'distance') return compareNullableNumber(a.distance_o90_pct, b.distance_o90_pct, sortDirection);
        if (sortKey === 'score') return compareNullableNumber(a.orotitan_score, b.orotitan_score, sortDirection);
        if (sortKey === 'fairValue') return compareNullableNumber(a.fair_value_base, b.fair_value_base, sortDirection);
        const left = a.analysis_date ?? '';
        const right = b.analysis_date ?? '';
        return sortDirection === 'asc' ? left.localeCompare(right) : right.localeCompare(left);
      });
  }, [rows, search, status, quality, sortKey, sortDirection]);

  function setSort(next: SortKey) {
    if (next === sortKey) setSortDirection((current) => current === 'asc' ? 'desc' : 'asc');
    else {
      setSortKey(next);
      setSortDirection('desc');
    }
  }

  const sortMark = (key: SortKey) => key === sortKey ? (sortDirection === 'asc' ? ' ↑' : ' ↓') : '';

  return <div className="space-y-4">
    <div className="grid gap-3 rounded-xl border border-slate-800 bg-slate-900/60 p-4 md:grid-cols-3">
      <label className="text-sm text-slate-400">Recherche<input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Société ou ticker" className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:border-cyan-500" /></label>
      <label className="text-sm text-slate-400">Statut<select value={status} onChange={(event) => setStatus(event.target.value as 'ALL' | CompanyStatus)} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"><option value="ALL">Tous</option><option value="OROTITAN">OroTitan</option><option value="FINALIST">Finalist</option><option value="PRICE_WAIT">Price wait</option><option value="TIER_1">Tier 1</option><option value="WATCHLIST">Watchlist</option><option value="REJECTED">Rejected</option></select></label>
      <label className="text-sm text-slate-400">Qualité OroTitan<select value={quality} onChange={(event) => setQuality(event.target.value as typeof quality)} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"><option value="ALL">Toutes</option><option value="TRUE">Structurellement OroTitan</option><option value="FALSE">Non OroTitan</option><option value="NULL">Non renseigné</option></select></label>
    </div>

    <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/60">
      <table className="min-w-[1050px] w-full text-left text-sm">
        <thead className="border-b border-slate-800 bg-slate-900/80 text-xs uppercase tracking-wide text-slate-500"><tr>
          <th className="px-4 py-3">Société</th><th className="px-4 py-3">Ticker</th><th className="px-4 py-3">Statut</th><th className="px-4 py-3">Cours</th><th className="px-4 py-3"><button onClick={() => setSort('fairValue')}>Fair value{sortMark('fairValue')}</button></th><th className="px-4 py-3"><button onClick={() => setSort('score')}>OroTitan Score{sortMark('score')}</button></th><th className="px-4 py-3">O90</th><th className="px-4 py-3"><button onClick={() => setSort('distance')}>Distance O90{sortMark('distance')}</button></th><th className="px-4 py-3"><button onClick={() => setSort('analysisDate')}>Analyse{sortMark('analysisDate')}</button></th>
        </tr></thead>
        <tbody className="divide-y divide-slate-800">{filtered.map((row) => {
          const priceProps = { currency: row.currency, quoteUnit: row.quote_unit, priceDecimals: row.price_decimals };
          return <tr key={row.id} tabIndex={0} role="link" onClick={() => router.push(`/company/${row.slug}`)} onKeyDown={(event) => { if (event.key === 'Enter') router.push(`/company/${row.slug}`); }} className="cursor-pointer transition hover:bg-slate-900/70 focus:bg-slate-900/70 focus:outline-none">
            <td className="px-4 py-4 font-medium text-slate-100">{row.name}</td><td className="px-4 py-4 font-mono text-slate-400">{row.ticker}</td><td className="px-4 py-4"><CompanyStatusBadge status={row.status} /></td><td className="px-4 py-4 text-slate-100"><PriceDisplay value={row.price} {...priceProps} /></td><td className="px-4 py-4 text-slate-200"><PriceDisplay value={row.fair_value_base} {...priceProps} /></td><td className="px-4 py-4"><ScoreBadge score={row.orotitan_score} /></td><td className="px-4 py-4 text-slate-200">{row.price_o90 === null ? <span className="text-slate-500">Non calibré</span> : <PriceDisplay value={row.price_o90} {...priceProps} />}</td><td className="px-4 py-4"><OroTitanDistance value={row.distance_o90_pct} compact /></td><td className="px-4 py-4 font-mono text-xs text-slate-400">{row.analysis_date ?? '—'}</td>
          </tr>;
        })}</tbody>
      </table>
      {filtered.length === 0 ? <div className="p-8 text-center text-sm text-slate-500">Aucune société ne correspond aux filtres.</div> : null}
    </div>
    <div className="text-xs text-slate-500">{filtered.length} société{filtered.length > 1 ? 's' : ''} affichée{filtered.length > 1 ? 's' : ''}. Les valeurs non calibrées restent hors tri numérique et sont toujours placées après les valeurs disponibles.</div>
  </div>;
}
