import Ajv2020, { type ErrorObject, type ValidateFunction } from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { readFileSync } from "node:fs";

type JsonObject = Record<string, unknown>;

export type AnalyticalDataBundleV2 = {
  evidenceLedger: JsonObject;
  conflictLedger: JsonObject;
  researchGapRegister: JsonObject;
  companyEconomicDna: JsonObject;
  calculationLedger: JsonObject;
  assumptionRegister: JsonObject;
  analyticalBlockOutputs: JsonObject;
};

export type AnalyticalValidationResult = {
  valid: boolean;
  errors: string[];
};

const schemaPaths = {
  common:
    "../../../contracts/orotitan-equity/vnext/analytical-engine-v2/OROTITAN_ANALYTICAL_COMMON_V2.0_DRAFT.schema.json",
  evidence:
    "../../../contracts/orotitan-equity/vnext/analytical-engine-v2/OROTITAN_EVIDENCE_LEDGER_V2.0_DRAFT.schema.json",
  conflict:
    "../../../contracts/orotitan-equity/vnext/analytical-engine-v2/OROTITAN_CONFLICT_LEDGER_V2.0_DRAFT.schema.json",
  gap:
    "../../../contracts/orotitan-equity/vnext/analytical-engine-v2/OROTITAN_RESEARCH_GAP_REGISTER_V2.0_DRAFT.schema.json",
  dna:
    "../../../contracts/orotitan-equity/vnext/analytical-engine-v2/OROTITAN_COMPANY_ECONOMIC_DNA_V2.0_DRAFT.schema.json",
  calculations:
    "../../../contracts/orotitan-equity/vnext/analytical-engine-v2/OROTITAN_CALCULATION_LEDGER_V2.0_DRAFT.schema.json",
  assumptions:
    "../../../contracts/orotitan-equity/vnext/analytical-engine-v2/OROTITAN_MATERIAL_ASSUMPTION_REGISTER_V2.0_DRAFT.schema.json",
  blocks:
    "../../../contracts/orotitan-equity/vnext/analytical-engine-v2/OROTITAN_ANALYTICAL_BLOCK_OUTPUTS_V2.0_DRAFT.schema.json",
} as const;

function loadSchema(path: string): JsonObject {
  return JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8")) as JsonObject;
}

const schemas = {
  common: loadSchema(schemaPaths.common),
  evidence: loadSchema(schemaPaths.evidence),
  conflict: loadSchema(schemaPaths.conflict),
  gap: loadSchema(schemaPaths.gap),
  dna: loadSchema(schemaPaths.dna),
  calculations: loadSchema(schemaPaths.calculations),
  assumptions: loadSchema(schemaPaths.assumptions),
  blocks: loadSchema(schemaPaths.blocks),
};

const ajv = new Ajv2020({
  allErrors: true,
  strict: false,
  strictRequired: false,
  strictTypes: false,
});
addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);

for (const schema of Object.values(schemas)) ajv.addSchema(schema);

const validators = {
  evidence: ajv.getSchema(String(schemas.evidence.$id)),
  conflict: ajv.getSchema(String(schemas.conflict.$id)),
  gap: ajv.getSchema(String(schemas.gap.$id)),
  dna: ajv.getSchema(String(schemas.dna.$id)),
  calculations: ajv.getSchema(String(schemas.calculations.$id)),
  assumptions: ajv.getSchema(String(schemas.assumptions.$id)),
  blocks: ajv.getSchema(String(schemas.blocks.$id)),
};

function requireValidator(
  value: ValidateFunction<unknown> | undefined,
  name: string,
): ValidateFunction<unknown> {
  if (!value) throw new Error(`missing AJV validator for ${name}`);
  return value;
}

function formatAjvErrors(
  label: string,
  errors: ErrorObject[] | null | undefined,
): string[] {
  return (errors ?? []).map(
    (error) =>
      `${label}${error.instancePath || "/"} ${error.message ?? "schema error"}`,
  );
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function objectAt(parent: JsonObject, key: string): JsonObject {
  const value = parent[key];
  return isObject(value) ? value : {};
}

function arrayAt(parent: JsonObject, key: string): unknown[] {
  const value = parent[key];
  return Array.isArray(value) ? value : [];
}

function stringAt(parent: JsonObject, key: string): string {
  const value = parent[key];
  return typeof value === "string" ? value : "";
}

function objectsAt(parent: JsonObject, key: string): JsonObject[] {
  return arrayAt(parent, key).filter(isObject);
}

function stringArrayAt(parent: JsonObject, key: string): string[] {
  return arrayAt(parent, key).filter(
    (value): value is string => typeof value === "string",
  );
}

function addUniqueIndex(
  rows: JsonObject[],
  idKey: string,
  label: string,
  errors: string[],
): Map<string, JsonObject> {
  const index = new Map<string, JsonObject>();
  for (const row of rows) {
    const id = stringAt(row, idKey);
    if (!id) continue;
    if (index.has(id)) {
      errors.push(`${label}: duplicate ${idKey} ${id}`);
      continue;
    }
    index.set(id, row);
  }
  return index;
}

function isIsoDate(value: string): boolean {
  return /^\d{4}-\d{2}-\d{2}$/.test(value);
}

function validateSameRun(
  artifacts: Array<[string, JsonObject]>,
  errors: string[],
): void {
  const keys = [
    "run_id",
    "issuer_id",
    "security_id",
    "dossier_id",
    "run_type",
    "canonical_mode",
    "data_cutoff",
    "contract_set_sha256",
  ] as const;

  const [baseLabel, baseArtifact] = artifacts[0];
  const baseLock = objectAt(baseArtifact, "run_lock");

  for (const [label, artifact] of artifacts.slice(1)) {
    const lock = objectAt(artifact, "run_lock");
    for (const key of keys) {
      if (lock[key] !== baseLock[key]) {
        errors.push(
          `run lock mismatch: ${label}.${key} differs from ${baseLabel}.${key}`,
        );
      }
    }
  }
}

function validateEvidenceReferences(
  ids: string[],
  label: string,
  evidenceIndex: Map<string, JsonObject>,
  errors: string[],
  requireAdmitted = true,
): void {
  for (const id of ids) {
    const evidence = evidenceIndex.get(id);
    if (!evidence) {
      errors.push(`${label}: unresolved evidence_id ${id}`);
      continue;
    }
    if (requireAdmitted && stringAt(evidence, "admission_status") !== "ADMITTED") {
      errors.push(`${label}: evidence_id ${id} is not ADMITTED`);
    }
  }
}

export function validateAnalyticalDataBundleV2(
  bundle: AnalyticalDataBundleV2,
): AnalyticalValidationResult {
  const errors: string[] = [];

  for (const [name, data, validator] of [
    ["evidenceLedger", bundle.evidenceLedger, validators.evidence],
    ["conflictLedger", bundle.conflictLedger, validators.conflict],
    ["researchGapRegister", bundle.researchGapRegister, validators.gap],
    ["companyEconomicDna", bundle.companyEconomicDna, validators.dna],
    ["calculationLedger", bundle.calculationLedger, validators.calculations],
    ["assumptionRegister", bundle.assumptionRegister, validators.assumptions],
    ["analyticalBlockOutputs", bundle.analyticalBlockOutputs, validators.blocks],
  ] as const) {
    const validate = requireValidator(validator, name);
    if (!validate(data)) errors.push(...formatAjvErrors(name, validate.errors));
  }

  if (errors.length > 0) return { valid: false, errors };

  validateSameRun(
    [
      ["evidenceLedger", bundle.evidenceLedger],
      ["conflictLedger", bundle.conflictLedger],
      ["researchGapRegister", bundle.researchGapRegister],
      ["companyEconomicDna", bundle.companyEconomicDna],
      ["calculationLedger", bundle.calculationLedger],
      ["assumptionRegister", bundle.assumptionRegister],
      ["analyticalBlockOutputs", bundle.analyticalBlockOutputs],
    ],
    errors,
  );

  const cutoff = stringAt(objectAt(bundle.evidenceLedger, "run_lock"), "data_cutoff");

  const sources = objectsAt(bundle.evidenceLedger, "sources");
  const sourceIndex = addUniqueIndex(sources, "source_id", "sources", errors);

  const evidenceItems = objectsAt(bundle.evidenceLedger, "evidence_items");
  const evidenceIndex = addUniqueIndex(
    evidenceItems,
    "evidence_id",
    "evidence_items",
    errors,
  );

  for (const evidence of evidenceItems) {
    const evidenceId = stringAt(evidence, "evidence_id");
    const sourceId = stringAt(evidence, "source_id");
    const source = sourceIndex.get(sourceId);
    if (!source) {
      errors.push(`evidence ${evidenceId}: unresolved source_id ${sourceId}`);
    }

    if (stringAt(evidence, "admission_status") === "ADMITTED") {
      const evidenceDate = stringAt(evidence, "source_date");
      if (!isIsoDate(evidenceDate)) {
        errors.push(
          `evidence ${evidenceId}: ADMITTED evidence requires verifiable source_date`,
        );
      } else if (cutoff && evidenceDate > cutoff) {
        errors.push(
          `evidence ${evidenceId}: source_date ${evidenceDate} exceeds DATA_CUTOFF ${cutoff}`,
        );
      }

      if (source) {
        const sourceDate = stringAt(source, "source_date");
        if (isIsoDate(sourceDate) && cutoff && sourceDate > cutoff) {
          errors.push(
            `source ${sourceId}: source_date ${sourceDate} exceeds DATA_CUTOFF ${cutoff}`,
          );
        }
      }
    }
  }

  const conflicts = objectsAt(bundle.conflictLedger, "conflicts");
  const conflictIndex = addUniqueIndex(
    conflicts,
    "conflict_id",
    "conflicts",
    errors,
  );
  for (const conflict of conflicts) {
    const id = stringAt(conflict, "conflict_id");
    validateEvidenceReferences(
      [
        ...stringArrayAt(conflict, "side_a_evidence_ids"),
        ...stringArrayAt(conflict, "side_b_evidence_ids"),
        ...stringArrayAt(conflict, "resolution_evidence_ids"),
      ],
      `conflict ${id}`,
      evidenceIndex,
      errors,
      false,
    );
  }

  const questions = objectsAt(bundle.researchGapRegister, "questions");
  const questionIndex = addUniqueIndex(
    questions,
    "question_id",
    "questions",
    errors,
  );
  for (const question of questions) {
    validateEvidenceReferences(
      stringArrayAt(question, "current_evidence_ids"),
      `question ${stringAt(question, "question_id")}`,
      evidenceIndex,
      errors,
      false,
    );
  }

  for (const provenance of objectsAt(bundle.companyEconomicDna, "field_provenance")) {
    validateEvidenceReferences(
      [
        ...stringArrayAt(provenance, "supporting_evidence_ids"),
        ...stringArrayAt(provenance, "contradicting_evidence_ids"),
      ],
      `DNA provenance ${stringAt(provenance, "field_path")}`,
      evidenceIndex,
      errors,
    );
  }

  const assumptions = objectsAt(bundle.assumptionRegister, "assumptions");
  const assumptionIndex = addUniqueIndex(
    assumptions,
    "assumption_id",
    "assumptions",
    errors,
  );
  for (const assumption of assumptions) {
    validateEvidenceReferences(
      stringArrayAt(assumption, "evidence_ids"),
      `assumption ${stringAt(assumption, "assumption_id")}`,
      evidenceIndex,
      errors,
    );
  }

  const calculations = objectsAt(bundle.calculationLedger, "calculations");
  const calculationIndex = addUniqueIndex(
    calculations,
    "calculation_id",
    "calculations",
    errors,
  );
  for (const calculation of calculations) {
    const calculationId = stringAt(calculation, "calculation_id");
    validateEvidenceReferences(
      stringArrayAt(calculation, "evidence_ids"),
      `calculation ${calculationId}`,
      evidenceIndex,
      errors,
    );
    for (const assumptionId of stringArrayAt(calculation, "assumption_ids")) {
      if (!assumptionIndex.has(assumptionId)) {
        errors.push(
          `calculation ${calculationId}: unresolved assumption_id ${assumptionId}`,
        );
      }
    }
    for (const input of objectsAt(calculation, "inputs")) {
      const sourceKind = stringAt(input, "source_kind");
      const sourceRefId = stringAt(input, "source_ref_id");
      if (sourceKind === "EVIDENCE" && !evidenceIndex.has(sourceRefId)) {
        errors.push(
          `calculation ${calculationId}: unresolved evidence input ${sourceRefId}`,
        );
      }
      if (sourceKind === "CALCULATION" && !calculationIndex.has(sourceRefId)) {
        errors.push(
          `calculation ${calculationId}: unresolved calculation input ${sourceRefId}`,
        );
      }
      if (sourceKind === "ASSUMPTION" && !assumptionIndex.has(sourceRefId)) {
        errors.push(
          `calculation ${calculationId}: unresolved assumption input ${sourceRefId}`,
        );
      }
      if (
        sourceKind === "DETERMINISTIC_CONSTANT" &&
        sourceRefId !== "NOT_APPLICABLE"
      ) {
        errors.push(
          `calculation ${calculationId}: deterministic constant must use NOT_APPLICABLE source_ref_id`,
        );
      }
    }
  }

  const claims = objectsAt(bundle.analyticalBlockOutputs, "claims");
  const claimIndex = addUniqueIndex(claims, "claim_id", "claims", errors);

  for (const claim of claims) {
    const claimId = stringAt(claim, "claim_id");
    const supportingIds = stringArrayAt(claim, "supporting_evidence_ids");
    const contradictingIds = stringArrayAt(claim, "contradicting_evidence_ids");

    validateEvidenceReferences(
      supportingIds,
      `claim ${claimId} supporting`,
      evidenceIndex,
      errors,
    );
    validateEvidenceReferences(
      contradictingIds,
      `claim ${claimId} contradicting`,
      evidenceIndex,
      errors,
    );

    for (const evidenceId of supportingIds) {
      const evidence = evidenceIndex.get(evidenceId);
      if (!evidence) continue;
      if (
        ["ASSUMPTION", "INFERENCE", "CALCULATION"].includes(
          stringAt(evidence, "evidence_role"),
        )
      ) {
        errors.push(
          `claim ${claimId}: ${stringAt(evidence, "evidence_role")} ${evidenceId} cannot masquerade as supporting evidence`,
        );
      }
    }

    if (
      stringAt(claim, "materiality") === "MATERIAL" &&
      stringAt(claim, "status") === "SUPPORTED" &&
      supportingIds.length === 0 &&
      stringArrayAt(claim, "calculation_ids").length === 0
    ) {
      errors.push(
        `claim ${claimId}: material SUPPORTED claim requires evidence or calculation lineage`,
      );
    }

    for (const calculationId of stringArrayAt(claim, "calculation_ids")) {
      const calculation = calculationIndex.get(calculationId);
      if (!calculation) {
        errors.push(`claim ${claimId}: unresolved calculation_id ${calculationId}`);
      } else if (stringAt(calculation, "block") !== stringAt(claim, "block")) {
        errors.push(
          `claim ${claimId}: calculation ${calculationId} belongs to another block`,
        );
      }
    }

    for (const assumptionId of stringArrayAt(claim, "assumption_ids")) {
      const assumption = assumptionIndex.get(assumptionId);
      if (!assumption) {
        errors.push(`claim ${claimId}: unresolved assumption_id ${assumptionId}`);
      }
    }

    for (const conflictId of stringArrayAt(claim, "conflict_ids")) {
      const conflict = conflictIndex.get(conflictId);
      if (!conflict) {
        errors.push(`claim ${claimId}: unresolved conflict_id ${conflictId}`);
        continue;
      }
      if (
        stringAt(claim, "status") === "SUPPORTED" &&
        stringAt(conflict, "materiality") === "MATERIAL" &&
        stringAt(conflict, "status") === "OPEN"
      ) {
        errors.push(
          `claim ${claimId}: cannot be SUPPORTED while material conflict ${conflictId} is OPEN`,
        );
      }
    }
  }

  const triggers = objectsAt(bundle.analyticalBlockOutputs, "invalidation_triggers");
  const triggerIndex = addUniqueIndex(
    triggers,
    "trigger_id",
    "invalidation_triggers",
    errors,
  );
  for (const trigger of triggers) {
    for (const claimId of stringArrayAt(trigger, "affected_claim_ids")) {
      if (!claimIndex.has(claimId)) {
        errors.push(
          `trigger ${stringAt(trigger, "trigger_id")}: unresolved claim_id ${claimId}`,
        );
      }
    }
  }

  const nodes = objectsAt(bundle.analyticalBlockOutputs, "causal_nodes");
  const nodeIndex = addUniqueIndex(nodes, "node_id", "causal_nodes", errors);
  for (const node of nodes) {
    const claimId = stringAt(node, "claim_id");
    if (claimId !== "NOT_APPLICABLE" && !claimIndex.has(claimId)) {
      errors.push(
        `causal node ${stringAt(node, "node_id")}: unresolved claim_id ${claimId}`,
      );
    }
    validateEvidenceReferences(
      stringArrayAt(node, "evidence_ids"),
      `causal node ${stringAt(node, "node_id")}`,
      evidenceIndex,
      errors,
    );
  }

  const edges = objectsAt(bundle.analyticalBlockOutputs, "causal_edges");
  const edgeIndex = addUniqueIndex(edges, "edge_id", "causal_edges", errors);
  for (const edge of edges) {
    for (const key of ["from_node_id", "to_node_id"] as const) {
      const nodeId = stringAt(edge, key);
      if (!nodeIndex.has(nodeId)) {
        errors.push(
          `causal edge ${stringAt(edge, "edge_id")}: unresolved ${key} ${nodeId}`,
        );
      }
    }
    validateEvidenceReferences(
      [
        ...stringArrayAt(edge, "supporting_evidence_ids"),
        ...stringArrayAt(edge, "counterevidence_ids"),
      ],
      `causal edge ${stringAt(edge, "edge_id")}`,
      evidenceIndex,
      errors,
    );
  }

  const blockOutputs = objectsAt(bundle.analyticalBlockOutputs, "block_outputs");
  addUniqueIndex(blockOutputs, "block_output_id", "block_outputs", errors);

  for (const output of blockOutputs) {
    const outputId = stringAt(output, "block_output_id");
    if (stringAt(output, "block") !== stringAt(output, "module_type")) {
      errors.push(`block output ${outputId}: block must equal module_type`);
    }

    for (const claimId of stringArrayAt(output, "material_claim_ids")) {
      const claim = claimIndex.get(claimId);
      if (!claim) {
        errors.push(`block output ${outputId}: unresolved claim_id ${claimId}`);
      } else if (stringAt(claim, "block") !== stringAt(output, "block")) {
        errors.push(
          `block output ${outputId}: claim ${claimId} belongs to another block`,
        );
      }
    }

    validateEvidenceReferences(
      stringArrayAt(output, "evidence_ids"),
      `block output ${outputId}`,
      evidenceIndex,
      errors,
    );

    for (const id of stringArrayAt(output, "calculation_ids")) {
      if (!calculationIndex.has(id)) {
        errors.push(`block output ${outputId}: unresolved calculation_id ${id}`);
      }
    }
    for (const id of stringArrayAt(output, "assumption_ids")) {
      if (!assumptionIndex.has(id)) {
        errors.push(`block output ${outputId}: unresolved assumption_id ${id}`);
      }
    }

    for (const id of stringArrayAt(output, "conflict_ids")) {
      if (!conflictIndex.has(id)) {
        errors.push(`block output ${outputId}: unresolved conflict_id ${id}`);
      }
    }
    for (const id of stringArrayAt(output, "open_question_ids")) {
      const question = questionIndex.get(id);
      if (!question) {
        errors.push(`block output ${outputId}: unresolved question_id ${id}`);
      } else if (
        stringAt(output, "status") === "COMPLETE" &&
        stringAt(question, "status") === "OPEN"
      ) {
        errors.push(
          `block output ${outputId}: COMPLETE block cannot retain OPEN question ${id}`,
        );
      }
    }
    for (const id of stringArrayAt(output, "causal_node_ids")) {
      if (!nodeIndex.has(id)) {
        errors.push(`block output ${outputId}: unresolved causal_node_id ${id}`);
      }
    }
    for (const id of stringArrayAt(output, "causal_edge_ids")) {
      if (!edgeIndex.has(id)) {
        errors.push(`block output ${outputId}: unresolved causal_edge_id ${id}`);
      }
    }
    for (const id of stringArrayAt(output, "invalidation_trigger_ids")) {
      if (!triggerIndex.has(id)) {
        errors.push(
          `block output ${outputId}: unresolved invalidation_trigger_id ${id}`,
        );
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

export function validateMaterialNumericValueV2(
  value: unknown,
): AnalyticalValidationResult {
  const validate = ajv.compile({
    $ref:
      "urn:orotitan:equity-research:analytical-common:v2.0-draft#/$defs/materialNumericValue",
  });
  const valid = validate(value);
  return {
    valid: Boolean(valid),
    errors: valid ? [] : formatAjvErrors("materialNumericValue", validate.errors),
  };
}
