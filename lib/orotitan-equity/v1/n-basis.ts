import { semanticStateSchema, type ScoreRange, type SemanticState } from "./semantic-states";

export type NormalizationReturnValue = number | ScoreRange | SemanticState;

function parseReturnValue(value: unknown, name: string): NormalizationReturnValue {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`${name} must be finite`);
    return value;
  }
  if (value && typeof value === "object" && !Array.isArray(value) && "min" in value && "max" in value) {
    const min = (value as { min?: unknown }).min;
    const max = (value as { max?: unknown }).max;
    if (typeof min !== "number" || typeof max !== "number" || !Number.isFinite(min) || !Number.isFinite(max) || min > max) {
      throw new Error(`${name} range is invalid`);
    }
    return { min, max };
  }
  if (typeof value === "string" && semanticStateSchema.safeParse(value).success) return value as SemanticState;
  throw new Error(`${name} is not a valid return value`);
}

function isNumericReturn(value: NormalizationReturnValue): value is number | ScoreRange {
  return typeof value === "number" || typeof value === "object";
}

export function selectNormalizationReturn(input: {
  matureNormalizationReturn: unknown;
  noMultipleExpansionReturn: unknown;
}): NormalizationReturnValue {
  const mature = parseReturnValue(input.matureNormalizationReturn, "mature_normalization_return");

  if (isNumericReturn(mature)) return mature;
  if (mature !== "NOT_ASSESSABLE" && mature !== "NOT_AVAILABLE") return mature;

  const sameMultiple = parseReturnValue(input.noMultipleExpansionReturn, "no_multiple_expansion_return");
  if (isNumericReturn(sameMultiple)) return sameMultiple;
  return mature;
}
