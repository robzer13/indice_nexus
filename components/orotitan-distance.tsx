export function OroTitanDistance({ value, compact = false }: { value: number | null; compact?: boolean }) {
  if (value === null) {
    return <span className="inline-flex rounded-control border border-state-neutral/30 bg-cockpit-bg px-2.5 py-1 text-xs font-semibold text-ink-muted">Non calibré</span>;
  }
  const reached = value >= 0;
  const near = value < 0 && value > -15;
  const label = reached ? 'O90 atteint' : near ? 'Proche du seuil' : 'Encore éloignée';
  const classes = reached
    ? 'border-state-success/50 bg-state-success/10 text-state-success'
    : near
      ? 'border-state-warning/50 bg-state-warning/10 text-state-warning'
      : 'border-state-neutral/35 bg-state-neutral/10 text-ink-secondary';
  return (
    <span className={`inline-flex items-center gap-2 rounded-control border px-2.5 py-1 ${classes}`}>
      <span className="font-mono text-sm font-bold tabular-nums">{value >= 0 ? '+' : ''}{value.toFixed(1)}%</span>
      {compact ? <span className="sr-only">{label}</span> : <span className="text-xs font-medium">{label}</span>}
    </span>
  );
}
