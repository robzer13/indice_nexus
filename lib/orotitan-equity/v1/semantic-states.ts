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

export const canonicalScoreSchema = z.union([
  z.number().finite().min(0).max(100),
  semanticStateSchema,
]);
export type CanonicalScore = z.infer<typeof canonicalScoreSchema>;
