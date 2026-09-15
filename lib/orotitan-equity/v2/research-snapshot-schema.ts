import Ajv2020, { type ErrorObject } from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { readFileSync } from "node:fs";
import {
  validateResearchSnapshotForPersistence as validateV1Snapshot,
  type ResearchSnapshot,
  type ValidationFailure,
} from "../v1/research-snapshot-schema";
import { validateV2ProductSemantics, type V2Product } from "./product";

type JsonObject = Record<string, unknown>;

export type V2ResearchSnapshot = ResearchSnapshot & { v2_product: V2Product };
export type ValidatedV2ResearchSnapshot = {
  ok: true;
  snapshot: V2ResearchSnapshot;
  canonicalContract: Record<string, unknown>;
};

const schema = JSON.parse(
  readFileSync(new URL("../../../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V2.json", import.meta.url), "utf8"),
) as JsonObject;
const ajv = new Ajv2020({ allErrors: true, strict: false });
addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);
const validateOverlay = ajv.compile(schema);

function formatAjvErrors(errors: ErrorObject[] | null | undefined): string[] {
  return (errors ?? []).map((error) => `${error.instancePath || "/"} ${error.message ?? "is invalid"}`);
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function validateV2ResearchSnapshotForPersistence(
  input: unknown,
  dossierId: string,
): ValidationFailure | ValidatedV2ResearchSnapshot {
  if (!validateOverlay(input)) return { ok: false, stage: "schema", errors: formatAjvErrors(validateOverlay.errors) };
  if (!isObject(input)) return { ok: false, stage: "schema", errors: ["canonical payload must be an object"] };

  const productErrors = validateV2ProductSemantics(input.v2_product);
  if (productErrors.length > 0) return { ok: false, stage: "boundary", errors: productErrors };

  const core = { ...input };
  delete core.v2_product;
  const base = validateV1Snapshot(core, dossierId);
  if (!base.ok) return base;

  return {
    ok: true,
    snapshot: input as V2ResearchSnapshot,
    canonicalContract: base.canonicalContract as unknown as Record<string, unknown>,
  };
}

export { validateOverlay as validateV2OverlaySchema };
