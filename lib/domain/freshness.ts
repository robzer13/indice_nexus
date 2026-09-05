export const MARKET_PRICE_STALE_HOURS = 96;

export interface FreshnessState {
  label: string;
  stale: boolean;
  ageHours: number | null;
}

export function getFreshness(
  asOf: string | null,
  now = new Date(),
): FreshnessState {
  if (!asOf) {
    return { label: 'Cours indisponible', stale: true, ageHours: null };
  }

  const parsed = new Date(asOf);
  if (Number.isNaN(parsed.getTime())) {
    return { label: 'Date de cours invalide', stale: true, ageHours: null };
  }

  const ageHours = Math.max(0, (now.getTime() - parsed.getTime()) / 3_600_000);
  const stale = ageHours > MARKET_PRICE_STALE_HOURS;

  return {
    label: stale ? 'Données anciennes' : 'Données récentes',
    stale,
    ageHours,
  };
}
