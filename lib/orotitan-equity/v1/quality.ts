import { z } from "zod";
import { semanticStateSchema } from "./semantic-states";

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
export const dimensionScoreSchema = z.union([numericDimensionScoreSchema, semanticStateSchema]);
export const qualityDimensionsSchema = z.object(
  Object.fromEntries(DIMENSION_KEYS.map((key) => [key, dimensionScoreSchema])) as Record<DimensionKey, typeof dimensionScoreSchema>,
).strict();
export type QualityDimensions = z.infer<typeof qualityDimensionsSchema>;

export const evidenceStateSchema = z.enum(["UNKNOWN", "PLAUSIBLE", "SUPPORTED", "STRONGLY_SUPPORTED", "FALSIFIED"]);
export type EvidenceState = z.infer<typeof evidenceStateSchema>;

export function computeOqs(dimensions: QualityDimensions) {
  const scores = DIMENSION_KEYS.map((key) => dimensions[key]);
  if (!scores.every((score): score is number => typeof score === "number")) {
    throw new Error("All seven dimension scores must be numeric; canonical weights cannot be renormalized");
  }
  const oqsRaw = DIMENSION_KEYS.reduce((sum, key, index) => sum + scores[index] * DIMENSION_WEIGHTS[key], 0);
  const weakLinkCap = Math.min(100, Math.min(...scores) + 25);
  return { oqsRaw, weakLinkCap, oqs: Math.min(oqsRaw, weakLinkCap) };
}
