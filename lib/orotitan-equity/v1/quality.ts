import { z } from "zod";
import { scoreRangeSchema, semanticStateSchema, type ScoreRange } from "./semantic-states";

export const DIMENSION_KEYS = [
  "MOAT", "RUNWAY", "RETURN_QUALITY", "CASH_ECONOMICS", "CAPITAL_ALLOCATION",
  "MANAGEMENT_GOVERNANCE", "RESILIENCE_RISK",
] as const;
export type DimensionKey = (typeof DIMENSION_KEYS)[number];

export const DIMENSION_WEIGHTS: Readonly<Record<DimensionKey, number>> = {
  MOAT: 0.2, RUNWAY: 0.15, RETURN_QUALITY: 0.2, CASH_ECONOMICS: 0.1,
  CAPITAL_ALLOCATION: 0.15, MANAGEMENT_GOVERNANCE: 0.1, RESILIENCE_RISK: 0.1,
};

const numericDimensionScoreSchema = z.number().finite().min(0).max(100).refine((score) => score % 5 === 0, {
  message: "Dimension scores must use 5-point increments",
});
const dimensionRangeSchema = scoreRangeSchema.refine((range) => range.min >= 0 && range.max <= 100
  && range.min % 5 === 0 && range.max % 5 === 0, {
  message: "Dimension range endpoints must use 5-point increments within [0,100]",
});
export const dimensionScoreSchema = z.union([numericDimensionScoreSchema, dimensionRangeSchema, semanticStateSchema]);
export const qualityDimensionsSchema = z.object(
  Object.fromEntries(DIMENSION_KEYS.map((key) => [key, dimensionScoreSchema])) as Record<DimensionKey, typeof dimensionScoreSchema>,
).strict();
export type QualityDimensions = z.infer<typeof qualityDimensionsSchema>;

export const moatEvidenceStateSchema = z.enum(["UNKNOWN", "PLAUSIBLE", "SUPPORTED", "STRONGLY_SUPPORTED", "FALSIFIED"]);
export const runwayEvidenceStateSchema = z.enum(["UNKNOWN", "PLAUSIBLE", "SUPPORTED", "STRONGLY_SUPPORTED"]);
export type MoatEvidenceState = z.infer<typeof moatEvidenceStateSchema>;
export type RunwayEvidenceState = z.infer<typeof runwayEvidenceStateSchema>;

const isRange = (value: number | ScoreRange): value is ScoreRange => typeof value !== "number";
const bounds = (value: number | ScoreRange): ScoreRange => isRange(value) ? value : { min: value, max: value };
const scalarOrRange = (min: number, max: number): number | ScoreRange => min === max ? min : { min, max };

function computeOqsScalar(scores: Record<DimensionKey, number>) {
  const oqsRaw = DIMENSION_KEYS.reduce((sum, key) => sum + scores[key] * DIMENSION_WEIGHTS[key], 0);
  const weakLinkCap = Math.min(100, Math.min(...DIMENSION_KEYS.map((key) => scores[key])) + 25);
  return { oqsRaw, weakLinkCap, oqs: Math.min(oqsRaw, weakLinkCap) };
}

export function computeOqs(dimensions: QualityDimensions) {
  const normalized = DIMENSION_KEYS.map((key) => dimensions[key]);
  if (!normalized.every((score): score is number | ScoreRange => typeof score === "number" || typeof score === "object")) {
    throw new Error("All seven dimension scores must be numeric; canonical weights cannot be renormalized");
  }
  if (normalized.some(isRange)) {
    const lower = Object.fromEntries(DIMENSION_KEYS.map((key, index) => [key, bounds(normalized[index]).min])) as Record<DimensionKey, number>;
    const upper = Object.fromEntries(DIMENSION_KEYS.map((key, index) => [key, bounds(normalized[index]).max])) as Record<DimensionKey, number>;
    const lowerResult = computeOqsScalar(lower);
    const upperResult = computeOqsScalar(upper);
    return {
      oqsRaw: scalarOrRange(lowerResult.oqsRaw, upperResult.oqsRaw),
      weakLinkCap: scalarOrRange(lowerResult.weakLinkCap, upperResult.weakLinkCap),
      oqs: scalarOrRange(lowerResult.oqs, upperResult.oqs),
    };
  }
  return computeOqsScalar(Object.fromEntries(DIMENSION_KEYS.map((key, index) => [key, normalized[index]])) as Record<DimensionKey, number>);
}
