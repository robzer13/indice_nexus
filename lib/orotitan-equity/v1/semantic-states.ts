import { z } from "zod";

export const SEMANTIC_STATES = [
  "UNKNOWN",
  "NOT_APPLICABLE",
  "NOT_ASSESSABLE",
  "MISSING",
  "NOT_AVAILABLE",
] as const;

export const semanticStateSchema = z.enum(SEMANTIC_STATES);
export type SemanticState = z.infer<typeof semanticStateSchema>;

export const scoreRangeSchema = z.object({
  min: z.number().finite(),
  max: z.number().finite(),
}).strict().refine((range) => range.min <= range.max, { message: "Score range min must be <= max" });
export type ScoreRange = z.infer<typeof scoreRangeSchema>;

export const canonicalScoreSchema = z.union([
  z.number().finite().min(0).max(100),
  scoreRangeSchema.refine((range) => range.min >= 0 && range.max <= 100, { message: "Score range must be within [0,100]" }),
  semanticStateSchema,
]);
export type CanonicalScore = number | ScoreRange | SemanticState;
