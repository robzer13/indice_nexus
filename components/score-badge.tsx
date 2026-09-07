export function ScoreBadge({ score }: { score: number | null }) {
  if (score === null) return <span className="text-xs text-ink-muted">Non disponible</span>;
  const tone = score >= 90 ? 'text-state-success border-state-success/40 bg-state-success/10' : score >= 80 ? 'text-state-accent border-state-accent/40 bg-state-accent/10' : score >= 70 ? 'text-state-warning border-state-warning/40 bg-state-warning/10' : 'text-ink-secondary border-state-neutral/35 bg-state-neutral/10';
  return <span className={`inline-flex min-w-12 justify-center rounded-control border px-2 py-1 font-mono text-sm font-bold tabular-nums ${tone}`} aria-label={`Score OroTitan ${score.toFixed(0)} sur 100`}>{score.toFixed(0)}</span>;
}
