import type { Json } from '@/lib/domain/types';

const LABELS: Record<string, string> = {
  oqs: 'OQS — qualité globale',
  ovs: 'OVS publié',
  investmentScore: 'Score investissement publié',
  moat: 'Avantage concurrentiel',
  runway: 'Potentiel de croissance',
  returnQuality: 'Qualité des rendements',
  cashEconomics: 'Économie du cash',
  capitalAllocation: 'Allocation du capital',
  managementGovernance: 'Management & gouvernance',
  resilienceRisk: 'Résilience & risques',
};

function renderValue(value: Json | undefined): string {
  if (value === undefined || value === null) return 'Non disponible';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value);
  return JSON.stringify(value, null, 2);
}

export function ScoreComponents({ value }: { value: Json }) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return <pre className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950 p-4 text-xs text-slate-300">{renderValue(value)}</pre>;
  }
  const entries = Object.entries(value);
  if (entries.length === 0) return <p className="text-sm text-slate-500">Aucune composante disponible.</p>;
  return <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{entries.map(([key, component]) => <div key={key} className="rounded-xl border border-slate-800 bg-slate-950/55 p-4"><div className="text-xs font-medium text-slate-500">{LABELS[key] ?? key}</div><pre className="mt-2 whitespace-pre-wrap break-words font-mono text-base font-semibold text-slate-100">{renderValue(component)}</pre></div>)}</div>;
}
