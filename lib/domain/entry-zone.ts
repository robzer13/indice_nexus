export type EntryZone =
  | 'AT_OR_BELOW_O90'
  | 'WITHIN_5'
  | 'WITHIN_10'
  | 'WITHIN_20'
  | 'ABOVE_20'
  | 'UNCALIBRATED';

export const ENTRY_ZONE_LABELS: Record<EntryZone, string> = {
  AT_OR_BELOW_O90: 'O90 atteint',
  WITHIN_5: 'À moins de 5 %',
  WITHIN_10: 'À 5–10 %',
  WITHIN_20: 'À 10–20 %',
  ABOVE_20: 'À plus de 20 %',
  UNCALIBRATED: 'Non calibré',
};

export function getEntryZone(distanceO90Pct: number | null): EntryZone {
  if (distanceO90Pct === null || !Number.isFinite(distanceO90Pct)) return 'UNCALIBRATED';
  if (distanceO90Pct >= 0) return 'AT_OR_BELOW_O90';
  if (distanceO90Pct >= -5) return 'WITHIN_5';
  if (distanceO90Pct >= -10) return 'WITHIN_10';
  if (distanceO90Pct >= -20) return 'WITHIN_20';
  return 'ABOVE_20';
}
