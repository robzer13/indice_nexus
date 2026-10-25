/**
 * V2.0.12 REVIEW CANDIDATE — field-local exact-literal compatibility only.
 *
 * Do not replace the pinned V2.0.10 schema or validator.
 * The frozen 2026-09-19 FICO Deep Dive contains exactly two valid,
 * already-certified source vocabularies that differ from the canonical
 * projection enums. This overlay preserves the literal source values;
 * it is NOT an analyst interpretation and adds no numeric outputs.
 *
 * Execution requires separate emergency-compatibility admission.
 */
export const EXACT_SOURCE_SEMANTIC_COMPATIBILITY = {
  patch: "OROTITAN_V2_EXACT_SOURCE_SEMANTIC_COMPATIBILITY_PATCH_V2.0.12",
  version: "2.0.12",
  sourceReturnInterpretability: "MEDIUM",
  sourceRefreshClass: "ACTIVATION_CHECK_OR_TARGETED_DELTA_IF_PRICE_ZONE_REACHED",
  compatibilitySchemaId: "urn:orotitan:equity-research:screener-contract:v1-v2-compat-literal-review-2.0.12",
} as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function arrayEquals(value: unknown, expected: readonly string[]): boolean {
  return Array.isArray(value) && value.length === expected.length &&
    value.every((v, index) => v === expected[index]);
}

export function buildV1FieldLocalLiteralCompatibleSchema(
  frozenV1CompatSchema: unknown,
): Record<string, unknown> {
  if (!isRecord(frozenV1CompatSchema)) throw new Error("V1_COMPAT_SCHEMA_NOT_OBJECT");
  const schema = JSON.parse(JSON.stringify(frozenV1CompatSchema)) as Record<string, unknown>;
  if (!isRecord(schema.$defs)) throw new Error("V1_COMPAT_DEFS_MISSING");
  const defs = schema.$defs;
  const metrics = (defs.analyticalMetrics as Record<string, unknown>).properties as Record<string, unknown>;
  const operational = (defs.operationalState as Record<string, unknown>).properties as Record<string, unknown>;
  const interp = metrics.return_interpretability as Record<string, unknown>;
  const refresh = operational.refresh_class as Record<string, unknown>;
  if (!arrayEquals(interp.enum, [
    "INTERPRETABLE", "INTERPRETABLE_WITH_LIMITATIONS", "NOT_INTERPRETABLE", "UNKNOWN",
  ])) throw new Error("FROZEN_RETURN_INTERPRETABILITY_ENUM_CHANGED");
  if (!arrayEquals(refresh.enum, [
    "PRICE_ONLY_DELTA", "ROUTINE_FUNDAMENTAL_DELTA", "FULL_REFRESH_REQUIRED", "NOT_APPLICABLE",
  ])) throw new Error("FROZEN_REFRESH_CLASS_ENUM_CHANGED");

  interp.enum = [...(interp.enum as string[]), EXACT_SOURCE_SEMANTIC_COMPATIBILITY.sourceReturnInterpretability];
  refresh.enum = [...(refresh.enum as string[]), EXACT_SOURCE_SEMANTIC_COMPATIBILITY.sourceRefreshClass];
  schema.$id = EXACT_SOURCE_SEMANTIC_COMPATIBILITY.compatibilitySchemaId;
  schema.title = "OroTitan V1 Analytical Core Exact-Certified-Literal Compatibility v2.0.12";
  schema.description = "Review-only additive field-local enum extension preserving certified source literals; no change to pinned V1.0/V2.0.10, I2, I3-B, other fields or frozen history.";
  return schema;
}
