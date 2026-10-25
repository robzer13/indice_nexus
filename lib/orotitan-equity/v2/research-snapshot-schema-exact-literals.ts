/** Opt-in strict review path: immutable original V1 compat + exactly two field-local certified literals + V2.0.11 lossless thesis. NOT production-wired. */
/* Opt-in V2.0.11 thesis-length-only schema compatibility. Frozen V2.0.10 validator stays byte-identical. */
import Ajv2020, { type ErrorObject } from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { readFileSync } from "node:fs";
import {
  validateResearchSnapshotForPersistenceAgainstSchema,
  type ResearchSnapshot,
  type ResearchSnapshotSchemaValidator,
  type ValidationFailure,
} from "../v1/research-snapshot-schema";
import type { V2Product } from "./product";
import { validateV2ProductWithLongCertifiedThesis } from "./thesis-length-compatibility";
import { buildV1FieldLocalLiteralCompatibleSchema } from "./exact-source-semantic-compatibility";

type JsonObject = Record<string, unknown>;

export type V2ResearchSnapshot = ResearchSnapshot & { v2_product: V2Product };
export type ValidatedV2ResearchSnapshot = {
  ok: true;
  snapshot: V2ResearchSnapshot;
  canonicalContract: Record<string, unknown>;
};

const schema = JSON.parse(
  readFileSync(new URL("../../../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V2_THESIS_COMPAT_V2.0.11.json", import.meta.url), "utf8"),
) as JsonObject;
const ajv = new Ajv2020({ allErrors: true, strict: false });
addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);
const validateOverlay = ajv.compile(schema);

const v1CompatSchema = buildV1FieldLocalLiteralCompatibleSchema(JSON.parse(
  readFileSync(new URL("../../../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.10.json", import.meta.url), "utf8"),
)) as JsonObject;
const v1CompatAjv = new Ajv2020({ allErrors: true, strict: true, strictRequired: false, strictTypes: false });
addFormats(v1CompatAjv as unknown as Parameters<typeof addFormats>[0]);
for (const keyword of [
  "x-orotitan-authority",
  "x-orotitan-layer-model",
  "x-orotitan-null-semantics",
  "x-orotitan-computed-vs-stored",
  "x-orotitan-deterministic-rules",
  "x-orotitan-server-recomputation-policy",
  "x-storage-precision",
  "x-unit",
]) v1CompatAjv.addKeyword({ keyword });
v1CompatAjv.addSchema(v1CompatSchema);
const validateV1CompatCore = v1CompatAjv.compile({
  $ref: "urn:orotitan:equity-research:screener-contract:v1-v2-compat-literal-review-2.0.12#/$defs/researchSnapshot",
}) as ResearchSnapshotSchemaValidator;

function formatAjvErrors(errors: ErrorObject[] | null | undefined): string[] {
  return (errors ?? []).map((error) => `${error.instancePath || "/"} ${error.message ?? "is invalid"}`);
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function validateV2ResearchSnapshotWithExactCertifiedLiterals(
  input: unknown,
  dossierId: string,
): ValidationFailure | ValidatedV2ResearchSnapshot {
  if (!validateOverlay(input)) return { ok: false, stage: "schema", errors: formatAjvErrors(validateOverlay.errors) };
  if (!isObject(input)) return { ok: false, stage: "schema", errors: ["canonical payload must be an object"] };

  const productErrors = validateV2ProductWithLongCertifiedThesis(input.v2_product);
  if (productErrors.length > 0) return { ok: false, stage: "boundary", errors: productErrors };

  const core = { ...input };
  delete core.v2_product;
  const base = validateResearchSnapshotForPersistenceAgainstSchema(core, dossierId, validateV1CompatCore);
  if (!base.ok) return base;

  return {
    ok: true,
    snapshot: input as V2ResearchSnapshot,
    canonicalContract: base.canonicalContract as unknown as Record<string, unknown>,
  };
}

export { validateOverlay as validateV2ThesisCompatibleOverlaySchema };
