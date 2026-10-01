import Ajv2020, { type ErrorObject } from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { readFileSync } from "node:fs";

type JsonObject = Record<string, unknown>;

export type AnalyticalReferenceIndex = {
  sourceIds?: ReadonlySet<string>;
  evidenceIds?: ReadonlySet<string>;
  conflictIds?: ReadonlySet<string>;
  calculationIds?: ReadonlySet<string>;
  assumptionIds?: ReadonlySet<string>;
};

export type AnalyticalDataValidationResult =
  | { ok: true; artifact: JsonObject }
  | { ok: false; stage: "schema" | "semantic"; errors: string[] };

const schema = JSON.parse(
  readFileSync(
    new URL(
      "../../../schemas/vnext/orotitan-analytical-data-contracts-v2.schema.v0.1.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as JsonObject;

const ajv = new Ajv2020({ allErrors: true, strict: false });
addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);
const validateShape = ajv.compile(schema);

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatAjvErrors(errors: ErrorObject[] | null | undefined): string[] {
  return (errors ?? []).map(
    (error) => `${error.instancePath || "/"} ${error.message ?? "is invalid"}`,
  );
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function recordsOf(artifact: JsonObject): JsonObject[] {
  const body = isObject(artifact.body) ? artifact.body : null;
  if (!body || !Array.isArray(body.records)) return [];
  return body.records.filter(isObject);
}

function pushDuplicateIdErrors(
  records: JsonObject[],
  key: string,
  label: string,
  errors: string[],
): void {
  const seen = new Set<string>();
  for (const record of records) {
    const id = record[key];
    if (typeof id !== "string") continue;
    if (seen.has(id)) errors.push(`${label} duplicate id: ${id}`);
    seen.add(id);
  }
}

function requireKnown(
  ids: string[],
  known: ReadonlySet<string> | undefined,
  label: string,
  errors: string[],
): void {
  if (!known) return;
  for (const id of ids) {
    if (!known.has(id)) errors.push(`${label} unresolved reference: ${id}`);
  }
}

function validateEvidenceLedger(
  artifact: JsonObject,
  refs: AnalyticalReferenceIndex,
  errors: string[],
): void {
  const cutoff = artifact.data_cutoff;
  const records = recordsOf(artifact);
  pushDuplicateIdErrors(records, "evidence_id", "EVIDENCE", errors);

  for (const record of records) {
    const evidenceId = typeof record.evidence_id === "string" ? record.evidence_id : "<unknown>";
    const sourceDate = record.source_date;
    const recordCutoff = record.data_cutoff;

    if (typeof cutoff === "string" && typeof sourceDate === "string" && sourceDate > cutoff) {
      errors.push(`EVIDENCE ${evidenceId} source_date exceeds artifact data_cutoff`);
    }
    if (typeof cutoff === "string" && typeof recordCutoff === "string" && recordCutoff !== cutoff) {
      errors.push(`EVIDENCE ${evidenceId} data_cutoff mismatches artifact data_cutoff`);
    }

    const sourceIds: string[] = [];
    if (typeof record.source_id === "string") sourceIds.push(record.source_id);
    if (typeof record.root_source_id === "string") sourceIds.push(record.root_source_id);
    requireKnown(sourceIds, refs.sourceIds, "SOURCE", errors);
  }
}

function validateConflictLedger(
  artifact: JsonObject,
  refs: AnalyticalReferenceIndex,
  errors: string[],
): void {
  const records = recordsOf(artifact);
  pushDuplicateIdErrors(records, "conflict_id", "CONFLICT", errors);

  for (const record of records) {
    const evidenceIds: string[] = [];
    for (const sideName of ["source_a", "source_b"]) {
      const side = isObject(record[sideName]) ? (record[sideName] as JsonObject) : null;
      if (side && typeof side.evidence_id === "string") evidenceIds.push(side.evidence_id);
    }
    if (evidenceIds.length === 2 && evidenceIds[0] === evidenceIds[1]) {
      errors.push(`CONFLICT ${String(record.conflict_id)} cannot compare the same evidence id twice`);
    }
    requireKnown(evidenceIds, refs.evidenceIds, "EVIDENCE", errors);
  }
}

function validateCalculationLedger(
  artifact: JsonObject,
  refs: AnalyticalReferenceIndex,
  errors: string[],
): void {
  const records = recordsOf(artifact);
  pushDuplicateIdErrors(records, "calculation_id", "CALCULATION", errors);
  for (const record of records) {
    requireKnown(
      stringArray(record.input_evidence_ids),
      refs.evidenceIds,
      "EVIDENCE",
      errors,
    );
  }
}

function validateAssumptionRegister(
  artifact: JsonObject,
  errors: string[],
): void {
  pushDuplicateIdErrors(recordsOf(artifact), "assumption_id", "ASSUMPTION", errors);
}

function validateDdInputSufficiency(artifact: JsonObject, errors: string[]): void {
  const body = isObject(artifact.body) ? artifact.body : null;
  if (!body || !Array.isArray(body.blocks)) return;

  const ready = body.ready_for_deep_dive === true;
  const criticalBlockers = stringArray(body.critical_blockers);
  if (ready && criticalBlockers.length > 0) {
    errors.push("READY_FOR_DEEP_DIVE cannot be true while critical_blockers is non-empty");
  }

  for (const block of body.blocks.filter(isObject)) {
    const applicability = block.applicability;
    const status = block.dd_input_status;
    const blockId = String(block.block_id ?? "<unknown>");

    if (applicability === "NOT_APPLICABLE" && status !== "NOT_APPLICABLE") {
      errors.push(`${blockId}: NOT_APPLICABLE applicability requires NOT_APPLICABLE status`);
    }
    if (applicability === "APPLICABLE" && status === "NOT_APPLICABLE") {
      errors.push(`${blockId}: APPLICABLE block cannot use NOT_APPLICABLE status`);
    }
    if (ready && status === "INSUFFICIENT") {
      errors.push(`${blockId}: READY_FOR_DEEP_DIVE cannot include INSUFFICIENT block`);
    }
  }
}

function validateAnalyticalBlock(
  artifact: JsonObject,
  refs: AnalyticalReferenceIndex,
  errors: string[],
): void {
  const body = isObject(artifact.body) ? artifact.body : null;
  if (!body) return;

  const supporting = stringArray(body.supporting_evidence_ids);
  const contradicting = stringArray(body.contradicting_evidence_ids);
  const conflictIds = stringArray(body.conflict_ids);
  const overlap = supporting.filter((id) => contradicting.includes(id));

  if (overlap.length > 0 && conflictIds.length === 0) {
    errors.push(
      `supporting/contradicting evidence overlap requires explicit conflict semantics: ${overlap.join(", ")}`,
    );
  }

  requireKnown([...supporting, ...contradicting], refs.evidenceIds, "EVIDENCE", errors);
  requireKnown(conflictIds, refs.conflictIds, "CONFLICT", errors);
  requireKnown(stringArray(body.calculation_ids), refs.calculationIds, "CALCULATION", errors);
  requireKnown(stringArray(body.material_assumption_ids), refs.assumptionIds, "ASSUMPTION", errors);
}

function validateMaterialChangeRevalidation(
  artifact: JsonObject,
  refs: AnalyticalReferenceIndex,
  errors: string[],
): void {
  const body = isObject(artifact.body) ? artifact.body : null;
  if (!body || !Array.isArray(body.checks)) return;

  const requiredChecks = new Set([
    "REOPEN_MATERIAL_EVIDENCE",
    "VERIFY_ROOT_PRIMARY_SOURCES",
    "SEARCH_DISCONFIRMING_EVIDENCE",
    "TEST_BEST_ALTERNATIVE_EXPLANATION",
    "RECONCILE_DOWNSTREAM_BLOCKS",
    "RECORD_PRIOR_STATE_CHANGE_REASON",
  ]);

  const observed = new Set<string>();
  let hasFail = false;
  for (const check of body.checks.filter(isObject)) {
    if (typeof check.check === "string") {
      if (observed.has(check.check)) errors.push(`duplicate material-change check: ${check.check}`);
      observed.add(check.check);
    }
    if (check.status === "FAIL") hasFail = true;
    requireKnown(stringArray(check.evidence_ids), refs.evidenceIds, "EVIDENCE", errors);
  }

  for (const check of requiredChecks) {
    if (!observed.has(check)) errors.push(`missing material-change check: ${check}`);
  }

  if (body.outcome === "REVALIDATED" && hasFail) {
    errors.push("material change cannot be REVALIDATED while a required check is FAIL");
  }

  requireKnown(
    stringArray(body.triggering_evidence_ids),
    refs.evidenceIds,
    "EVIDENCE",
    errors,
  );
}

export function validateAnalyticalDataArtifact(
  input: unknown,
  refs: AnalyticalReferenceIndex = {},
): AnalyticalDataValidationResult {
  if (!validateShape(input)) {
    return { ok: false, stage: "schema", errors: formatAjvErrors(validateShape.errors) };
  }
  if (!isObject(input)) {
    return { ok: false, stage: "schema", errors: ["artifact must be an object"] };
  }

  const errors: string[] = [];
  switch (input.artifact_type) {
    case "EVIDENCE_LEDGER":
      validateEvidenceLedger(input, refs, errors);
      break;
    case "CONFLICT_LEDGER":
      validateConflictLedger(input, refs, errors);
      break;
    case "CALCULATION_LEDGER":
      validateCalculationLedger(input, refs, errors);
      break;
    case "MATERIAL_ASSUMPTION_REGISTER":
      validateAssumptionRegister(input, errors);
      break;
    case "DD_INPUT_SUFFICIENCY_RECORD":
      validateDdInputSufficiency(input, errors);
      break;
    case "ANALYTICAL_BLOCK_OUTPUT":
      validateAnalyticalBlock(input, refs, errors);
      break;
    case "MATERIAL_CHANGE_REVALIDATION_RECORD":
      validateMaterialChangeRevalidation(input, refs, errors);
      break;
    default:
      break;
  }

  return errors.length > 0
    ? { ok: false, stage: "semantic", errors }
    : { ok: true, artifact: input };
}

export { validateShape as validateAnalyticalDataArtifactShape };
