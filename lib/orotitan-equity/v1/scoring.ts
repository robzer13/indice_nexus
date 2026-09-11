import type { InvestmentConclusionStatus, MosStatus, ScorePermission, ValuationReliability } from "./certification";
import type { CanonicalScore } from "./semantic-states";

export const MOS_CAPS = { ROBUST: 100, ADEQUATE: 90, THIN: 75, NONE: 55 } as const;
export const VALUATION_RELIABILITY_CAPS = { HIGH: 100, MEDIUM: 95, LOW: 80 } as const;

export function computeReturnComponent(currentFiveYearScore: number, normalizedTenYearScore: number): number {
  return Math.min(0.6 * currentFiveYearScore + 0.4 * normalizedTenYearScore, normalizedTenYearScore + 15);
}

export interface OvsInput {
  returnComponent: number;
  mosStatus: MosStatus;
  valuationReliability: ValuationReliability;
  scorePermission: ScorePermission;
  investmentConclusionStatus: InvestmentConclusionStatus;
}

export function computeOvs(input: OvsInput): CanonicalScore {
  if (input.scorePermission === "SUSPENDED") {
    return input.valuationReliability === "NOT_ASSESSABLE" ? "NOT_ASSESSABLE" : "NOT_AVAILABLE";
  }
  if (input.valuationReliability === "NOT_ASSESSABLE" || input.mosStatus === "NOT_ASSESSABLE") return "NOT_ASSESSABLE";
  if (input.investmentConclusionStatus === "NOT_CERTIFIED" || input.investmentConclusionStatus === "INSUFFICIENT_DATA") {
    return "NOT_ASSESSABLE";
  }
  return Math.min(input.returnComponent, MOS_CAPS[input.mosStatus], VALUATION_RELIABILITY_CAPS[input.valuationReliability]);
}

export function computeInvestmentScore(oqs: CanonicalScore, ovs: CanonicalScore, permission: ScorePermission) {
  if (permission === "SUSPENDED") return { investmentRaw: "NOT_AVAILABLE" as const, investmentScore: "NOT_AVAILABLE" as const };
  if (typeof ovs !== "number" || typeof oqs !== "number") return { investmentRaw: "NOT_AVAILABLE" as const, investmentScore: "NOT_AVAILABLE" as const };
  const investmentRaw = 0.7 * oqs + 0.3 * ovs;
  return { investmentRaw, investmentScore: Math.min(investmentRaw, oqs, ovs + 15) };
}
