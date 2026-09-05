import { ENTRY_ZONE_LABELS, type EntryZone } from '@/lib/domain/entry-zone';

const classes: Record<EntryZone, string> = {
  AT_OR_BELOW_O90: 'border-emerald-800 bg-emerald-950/35 text-emerald-200',
  WITHIN_5: 'border-cyan-800 bg-cyan-950/30 text-cyan-200',
  WITHIN_10: 'border-sky-900 bg-sky-950/25 text-sky-200',
  WITHIN_20: 'border-amber-900 bg-amber-950/25 text-amber-200',
  ABOVE_20: 'border-slate-700 bg-slate-900 text-slate-400',
  UNCALIBRATED: 'border-slate-800 bg-slate-950 text-slate-600',
};

export function EntryZoneBadge({ zone }: { zone: EntryZone }) {
  return <span className={`inline-flex rounded-full border px-2 py-1 text-xs font-medium ${classes[zone]}`}>{ENTRY_ZONE_LABELS[zone]}</span>;
}
