import type { ReactNode } from 'react';

export function MetricCard({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-panel"><div className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{label}</div><div className="mt-2 text-2xl font-semibold text-slate-100">{value}</div>{hint ? <div className="mt-1 text-xs text-slate-500">{hint}</div> : null}</div>;
}
