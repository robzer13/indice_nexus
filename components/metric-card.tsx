import type { ReactNode } from 'react';
import { Panel } from '@/components/ui/panel';

export function MetricCard({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return <Panel className="p-4 shadow-panel"><div className="text-xs font-semibold uppercase tracking-[0.14em] text-ink-muted">{label}</div><div className="mt-2 text-2xl font-semibold tabular-nums text-ink-primary">{value}</div>{hint ? <div className="mt-1 text-xs text-ink-muted">{hint}</div> : null}</Panel>;
}
