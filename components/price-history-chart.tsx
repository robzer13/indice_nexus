import { formatPrice } from '@/lib/domain/format-price';
import type { MarketPriceRow, QuoteUnit } from '@/lib/domain/types';

interface Threshold {
  label: string;
  value: number | null;
}

export function PriceHistoryChart({ points, thresholds, currency, quoteUnit, priceDecimals }: { points: MarketPriceRow[]; thresholds: Threshold[]; currency: string; quoteUnit: QuoteUnit; priceDecimals: number }) {
  if (points.length < 2) {
    return <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 text-sm text-slate-500">Historique de prix encore insuffisant. Le graphique apparaîtra après plusieurs synchronisations de marché.</div>;
  }

  const usableThresholds = thresholds.filter((item): item is { label: string; value: number } => item.value !== null && Number.isFinite(item.value));
  const values = [...points.map((point) => point.price), ...usableThresholds.map((item) => item.value)];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const padding = Math.max((max - min) * 0.08, max * 0.01);
  const low = min - padding;
  const high = max + padding;
  const width = 900;
  const height = 260;
  const left = 48;
  const right = 16;
  const top = 18;
  const bottom = 32;
  const chartW = width - left - right;
  const chartH = height - top - bottom;

  const x = (index: number) => left + (index / Math.max(1, points.length - 1)) * chartW;
  const y = (value: number) => top + ((high - value) / Math.max(0.000001, high - low)) * chartH;
  const polyline = points.map((point, index) => `${x(index)},${y(point.price)}`).join(' ');
  const fmt = (value: number) => formatPrice({ value, currency, quoteUnit, priceDecimals });

  return <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Historique du cours et seuils OroTitan" className="h-auto w-full">
      <line x1={left} y1={top + chartH} x2={left + chartW} y2={top + chartH} stroke="currentColor" className="text-slate-800"/>
      <text x={4} y={top + 4} className="fill-slate-500 text-[11px]">{fmt(high)}</text>
      <text x={4} y={top + chartH} className="fill-slate-500 text-[11px]">{fmt(low)}</text>
      {usableThresholds.map((threshold, index) => <g key={threshold.label}><line x1={left} y1={y(threshold.value)} x2={left + chartW} y2={y(threshold.value)} stroke="currentColor" strokeDasharray="5 5" className={index === 0 ? 'text-cyan-700' : 'text-slate-700'}/><text x={left + 6} y={y(threshold.value) - 5} className="fill-slate-500 text-[10px]">{threshold.label} · {fmt(threshold.value)}</text></g>)}
      <polyline points={polyline} fill="none" stroke="currentColor" strokeWidth="2.5" className="text-cyan-300"/>
      {points.map((point, index) => <circle key={point.id} cx={x(index)} cy={y(point.price)} r="3" fill="currentColor" className="text-cyan-300"/>)}
      <text x={left} y={height - 7} className="fill-slate-500 text-[10px]">{new Date(points[0].as_of).toLocaleDateString('fr-FR')}</text>
      <text x={left + chartW} y={height - 7} textAnchor="end" className="fill-slate-500 text-[10px]">{new Date(points.at(-1)!.as_of).toLocaleDateString('fr-FR')}</text>
    </svg>
  </div>;
}
