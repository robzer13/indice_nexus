import type { CompanyStatus } from '@/lib/domain/types';

const labels: Record<CompanyStatus, string> = {
  OROTITAN: 'OroTitan',
  FINALIST: 'Finalist',
  PRICE_WAIT: 'Price wait',
  TIER_1: 'Tier 1',
  WATCHLIST: 'Watchlist',
  REJECTED: 'Rejected',
};

const classes: Record<CompanyStatus, string> = {
  OROTITAN: 'border-emerald-400/40 bg-emerald-400/10 text-emerald-200',
  FINALIST: 'border-cyan-400/40 bg-cyan-400/10 text-cyan-200',
  PRICE_WAIT: 'border-amber-400/40 bg-amber-400/10 text-amber-200',
  TIER_1: 'border-indigo-400/40 bg-indigo-400/10 text-indigo-200',
  WATCHLIST: 'border-slate-400/30 bg-slate-400/10 text-slate-300',
  REJECTED: 'border-rose-400/40 bg-rose-400/10 text-rose-200',
};

export function CompanyStatusBadge({ status }: { status: CompanyStatus | null }) {
  if (!status) return <span className="text-sm text-slate-500">Non renseigné</span>;
  return <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold tracking-wide ${classes[status]}`}>{labels[status]}</span>;
}
