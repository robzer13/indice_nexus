import type { InvestmentConclusionStatus, MosStatus, ScorePermission, ValuationReliability } from "./certification";
import type { CanonicalScore, ScoreRange } from "./semantic-states";
export type NumericScore = number | ScoreRange;

export const MOS_CAPS = { ROBUST: 100, ADEQUATE: 90, THIN: 75, NONE: 55 } as const;
export const VALUATION_RELIABILITY_CAPS = { HIGH: 100, MEDIUM: 95, LOW: 80 } as const;

const bounds = (value: NumericScore): ScoreRange => typeof value === "number" ? { min: value, max: value } : value;
const scalarOrRange = (min: number, max: number): NumericScore => min === max ? min : { min, max };

export function computeReturnComponent(primaryExpectedReturnScore: NumericScore, normalizedExpectedReturnScore: NumericScore): NumericScore {
  const primary = bounds(primaryExpectedReturnScore);
  const normalized = bounds(normalizedExpectedReturnScore);
  const lower = Math.min(0.6 * primary.min + 0.4 * normalized.min, normalized.min + 15);
  const upper = Math.min(0.6 * primary.max + 0.4 * normalized.max, normalized.max + 15);
  return scalarOrRange(lower, upper);
}

export interface OvsInput {
  returnComponent: CanonicalScore;
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
  if (typeof input.returnComponent === "string") return "NOT_ASSESSABLE";
  const returnComponent = bounds(input.returnComponent);
  const cap = Math.min(MOS_CAPS[input.mosStatus], VALUATION_RELIABILITY_CAPS[input.valuationReliability]);
  return scalarOrRange(Math.min(returnComponent.min, cap), Math.min(returnComponent.max, cap));
}

export function computeInvestmentScore(oqs: CanonicalScore, ovs: CanonicalScore, permission: ScorePermission) {
  if (permission === "SUSPENDED") return { investmentRaw: "NOT_AVAILABLE" as const, investmentScore: "NOT_AVAILABLE" as const };
  if ((typeof ovs !== "number" && typeof ovs !== "object") || (typeof oqs !== "number" && typeof oqs !== "object")) {
    return { investmentRaw: "NOT_AVAILABLE" as const, investmentScore: "NOT_AVAILABLE" as const };
  }
  const oqsBounds = bounds(oqs);
  const ovsBounds = bounds(ovs);
  const lowerRaw = 0.7 * oqsBounds.min + 0.3 * ovsBounds.min;
  const upperRaw = 0.7 * oqsBounds.max + 0.3 * ovsBounds.max;
  const lowerScore = Math.min(lowerRaw, oqsBounds.min, ovsBounds.min + 15);
  const upperScore = Math.min(upperRaw, oqsBounds.max, ovsBounds.max + 15);
  return { investmentRaw: scalarOrRange(lowerRaw, upperRaw), investmentScore: scalarOrRange(lowerScore, upperScore) };
}
