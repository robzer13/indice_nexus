export default function OroTitanLoading() {
  return (
    <div className="space-y-4" aria-busy="true">
      <div className="h-5 w-40 animate-pulse rounded bg-slate-800" />
      <div className="h-10 max-w-xl animate-pulse rounded bg-slate-800" />
      <div className="grid gap-3 md:grid-cols-3">
        {[0, 1, 2].map((value) => <div key={value} className="h-28 animate-pulse rounded-xl border border-slate-800 bg-slate-900/60" />)}
      </div>
      <p className="text-sm text-slate-500">Chargement du dossier OroTitan…</p>
    </div>
  );
}
