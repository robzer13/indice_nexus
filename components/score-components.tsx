import type { Json } from '@/lib/domain/types';

function renderValue(value: Json | undefined): string {
  if (value === undefined || value === null) return 'NULL';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value);
  return JSON.stringify(value, null, 2);
}

export function ScoreComponents({ value }: { value: Json }) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return <pre className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950 p-4 text-xs text-slate-300">{renderValue(value)}</pre>;
  }
  const entries = Object.entries(value);
  if (entries.length === 0) return <p className="text-sm text-slate-500">Aucune composante disponible.</p>;
  return <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">{entries.map(([key, component]) => <div key={key} className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"><div className="break-all text-xs font-medium text-slate-500">{key}</div><pre className="mt-2 whitespace-pre-wrap break-words font-mono text-sm text-slate-200">{renderValue(component)}</pre></div>)}</div>;
}
