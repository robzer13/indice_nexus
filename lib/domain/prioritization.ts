import { getDistanceO90 } from '@/lib/domain/distance';
import type { CompanyState } from '@/lib/domain/types';

export interface PrioritizedCompany extends CompanyState {
  distance_o90_pct: number;
}

export function prioritizeCompanies(
  companies: readonly CompanyState[],
  limit = 6,
): PrioritizedCompany[] {
  return companies
    .map((company) => {
      const distance = getDistanceO90(company.price, company.price_o90);
      return distance === null
        ? null
        : { ...company, distance_o90_pct: distance };
    })
    .filter((company): company is PrioritizedCompany => company !== null)
    .sort((a, b) => {
      const aReached = a.distance_o90_pct >= 0;
      const bReached = b.distance_o90_pct >= 0;

      if (aReached !== bReached) {
        return aReached ? -1 : 1;
      }

      if (aReached && bReached) {
        return b.distance_o90_pct - a.distance_o90_pct;
      }

      if (a.distance_o90_pct !== b.distance_o90_pct) {
        return b.distance_o90_pct - a.distance_o90_pct;
      }

      return (b.orotitan_score ?? -1) - (a.orotitan_score ?? -1);
    })
    .slice(0, Math.max(0, limit));
}
