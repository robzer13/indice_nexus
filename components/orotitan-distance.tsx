export function OroTitanDistance({ value, compact = false }: { value: number | null; compact?: boolean }) {
  if (value === null) {
    return <span className="inline-flex rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-xs font-medium text-slate-400">Non calibré</span>;
  }
  const reached = value >= 0;
  const near = value < 0 && value > -15;
  const label = reached ? 'O90 atteint' : near ? 'Proche du seuil' : 'Encore éloignée';
  const classes = reached
    ? 'border-emerald-400/40 bg-emerald-400/10 text-emerald-200'
    : near
      ? 'border-amber-400/40 bg-amber-400/10 text-amber-200'
      : 'border-slate-600 bg-slate-800/70 text-slate-300';
  return (
    <span className={`inline-flex items-center gap-2 rounded-md border px-2 py-1 ${classes}`}>
      <span className="font-mono text-sm font-semibold tabular-nums">{value >= 0 ? '+' : ''}{value.toFixed(1)}%</span>
      {!compact && <span className="text-xs">{label}</span>}
    </span>
  );
}
