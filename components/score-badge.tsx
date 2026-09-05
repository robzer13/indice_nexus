export function ScoreBadge({ score }: { score: number | null }) {
  if (score === null) return <span className="text-slate-500">Non disponible</span>;
  const tone = score >= 90 ? 'text-emerald-200 border-emerald-400/30 bg-emerald-400/10' : score >= 80 ? 'text-cyan-200 border-cyan-400/30 bg-cyan-400/10' : score >= 70 ? 'text-amber-200 border-amber-400/30 bg-amber-400/10' : 'text-slate-300 border-slate-600 bg-slate-800/60';
  return <span className={`inline-flex min-w-12 justify-center rounded-md border px-2 py-1 font-mono text-sm font-semibold ${tone}`}>{score.toFixed(0)}</span>;
}
