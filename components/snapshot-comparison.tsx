import { PriceDisplay } from '@/components/price-display';
import type { QuoteUnit, SnapshotHistoryRow } from '@/lib/domain/types';

function delta(current: number | null, previous: number | null): number | null {
  if (current === null || previous === null || !Number.isFinite(current) || !Number.isFinite(previous)) return null;
  return current - previous;
}

function Delta({ value, suffix = '' }: { value: number | null; suffix?: string }) {
  if (value === null) return <span className="text-slate-600">—</span>;
  const sign = value > 0 ? '+' : '';
  return <span className={value > 0 ? 'text-emerald-300' : value < 0 ? 'text-rose-300' : 'text-slate-400'}>{sign}{value.toFixed(1)}{suffix}</span>;
}

export function SnapshotComparison({ snapshots, currency, quoteUnit, priceDecimals }: { snapshots: SnapshotHistoryRow[]; currency: string; quoteUnit: QuoteUnit; priceDecimals: number }) {
  if (snapshots.length < 2) {
    return <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 text-sm text-slate-500">Aucun snapshot précédent comparable pour l’instant.</div>;
  }
  const current = snapshots[0];
  const previous = snapshots[1];
  const priceProps = { currency, quoteUnit, priceDecimals };

  return <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
    <div className="flex flex-wrap items-center justify-between gap-2"><div><div className="text-sm font-semibold text-white">Snapshot actuel vs précédent</div><div className="mt-1 text-xs text-slate-500">{current.analysis_date} / {current.model_version} vs {previous.analysis_date} / {previous.model_version}</div></div></div>
    <div className="mt-4 grid gap-4 sm:grid-cols-3">
      <Compare label="OroTitan Score" current={current.orotitan_score ?? '—'} previous={previous.orotitan_score ?? '—'} change={<Delta value={delta(current.orotitan_score, previous.orotitan_score)}/>}/>
      <Compare label="Fair value centrale" current={<PriceDisplay value={current.fair_value_base} {...priceProps}/>} previous={<PriceDisplay value={previous.fair_value_base} {...priceProps}/>} change={<Delta value={delta(current.fair_value_base, previous.fair_value_base)}/>}/>
      <Compare label="O90" current={current.price_o90 === null ? 'Non calibré' : <PriceDisplay value={current.price_o90} {...priceProps}/>} previous={previous.price_o90 === null ? 'Non calibré' : <PriceDisplay value={previous.price_o90} {...priceProps}/>} change={<Delta value={delta(current.price_o90, previous.price_o90)}/>}/>
    </div>
  </div>;
}

function Compare({ label, current, previous, change }: { label: string; current: React.ReactNode; previous: React.ReactNode; change: React.ReactNode }) {
  return <div className="rounded-lg border border-slate-800 bg-slate-950/55 p-4"><div className="text-xs uppercase tracking-wide text-slate-600">{label}</div><div className="mt-2 text-base font-semibold text-slate-100">{current}</div><div className="mt-1 text-xs text-slate-500">Précédent : {previous}</div><div className="mt-2 text-xs">Δ {change}</div></div>;
}
