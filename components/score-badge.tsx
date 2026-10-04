export function ScoreBadge({ score, compact = false }: { score: number | null; compact?: boolean }) {
  if (score === null) return <span className="text-slate-500">Non disponible</span>;
  const tone = score >= 90
    ? 'text-emerald-200 border-emerald-400/30 bg-emerald-400/10'
    : score >= 80
      ? 'text-cyan-200 border-cyan-400/30 bg-cyan-400/10'
      : score >= 70
        ? 'text-amber-100 border-amber-400/30 bg-amber-400/10'
        : score >= 50
          ? 'text-orange-200 border-orange-400/30 bg-orange-400/10'
          : 'text-rose-200 border-rose-400/30 bg-rose-400/10';
  return <span className={`inline-flex min-w-12 justify-center rounded-md border px-2 py-1 font-mono font-semibold ${compact ? 'text-xs' : 'text-sm'} ${tone}`}>{score.toFixed(0)}</span>;
}
