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
  OROTITAN: 'border-state-success/40 bg-state-success/10 text-state-success',
  FINALIST: 'border-state-accent/40 bg-state-accent/10 text-state-accent',
  PRICE_WAIT: 'border-state-warning/40 bg-state-warning/10 text-state-warning',
  TIER_1: 'border-state-info/40 bg-state-info/10 text-state-info',
  WATCHLIST: 'border-state-neutral/35 bg-state-neutral/10 text-ink-secondary',
  REJECTED: 'border-state-danger/40 bg-state-danger/10 text-state-danger',
};

export function CompanyStatusBadge({ status }: { status: CompanyStatus | null }) {
  if (!status) return <span className="text-xs text-ink-muted">Non renseigné</span>;
  return <span className={`inline-flex rounded-full border px-2.5 py-1 text-[11px] font-semibold tracking-wide ${classes[status]}`}>{labels[status]}</span>;
}
