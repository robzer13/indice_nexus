import { ENTRY_ZONE_LABELS, type EntryZone } from '@/lib/domain/entry-zone';

const classes: Record<EntryZone, string> = {
  AT_OR_BELOW_O90: 'border-state-success/45 bg-state-success/10 text-state-success',
  WITHIN_5: 'border-state-accent/45 bg-state-accent/10 text-state-accent',
  WITHIN_10: 'border-state-warning/45 bg-state-warning/10 text-state-warning',
  WITHIN_20: 'border-amber-500/50 bg-amber-500/10 text-amber-300',
  ABOVE_20: 'border-state-neutral/30 bg-state-neutral/10 text-ink-muted',
  UNCALIBRATED: 'border-state-neutral/25 bg-cockpit-bg text-ink-muted',
};

export function EntryZoneBadge({ zone }: { zone: EntryZone }) {
  return <span className={`inline-flex rounded-full border px-2.5 py-1 text-[11px] font-semibold tracking-wide ${classes[zone]}`}>{ENTRY_ZONE_LABELS[zone]}</span>;
}
