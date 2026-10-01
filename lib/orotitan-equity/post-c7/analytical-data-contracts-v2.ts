import Ajv2020, { type ErrorObject } from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { readFileSync } from "node:fs";

type JsonObject = Record<string, unknown>;

export type AnalyticalDataContext = {
  runId: string;
  issuerId: string;
  securityId: string | null;
  dossierId: string | null;
  dataCutoff: string;
};

export type AnalyticalDataValidationResult =
  | { ok: true; value: JsonObject }
  | { ok: false; stage: "schema" | "semantic"; errors: string[] };

const schema = JSON.parse(
  readFileSync(
    new URL(
      "../../../contracts/orotitan-equity/post-c7/OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2_SCHEMA_V0.1.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as JsonObject;

const ajv = new Ajv2020({
  allErrors: true,
  strict: true,
  strictRequired: false,
  strictTypes: false,
});
addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);
const validateSchema = ajv.compile(schema);

function formatAjvErrors(errors: ErrorObject[] | null | undefined): string[] {
  return (errors ?? []).map(
    (error) => `${error.instancePath || "/"} ${error.message ?? "is invalid"}`,
  );
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asObjects(value: unknown): JsonObject[] {
  return Array.isArray(value) ? value.filter(isObject) : [];
}

function stringSet(items: JsonObject[], key: string): Set<string> {
  return new Set(
    items
      .map((item) => item[key])
      .filter((value): value is string => typeof value === "string"),
  );
}

function duplicateIds(items: JsonObject[], key: string): string[] {
  const seen = new Set<string>();
  const dup = new Set<string>();
  for (const item of items) {
    const value = item[key];
    if (typeof value !== "string") continue;
    if (seen.has(value)) dup.add(value);
    seen.add(value);
  }
  return [...dup].sort();
}

function refsFromArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function checkEvidenceRefs(
  errors: string[],
  label: string,
  refs: unknown,
  evidenceIds: Set<string>,
): void {
  for (const id of refsFromArray(refs)) {
    if (!evidenceIds.has(id)) errors.push(`${label} references unknown evidence_id ${id}`);
  }
}

function compareIsoDate(left: string, right: string): number {
  return left.localeCompare(right);
}

function blockCode(item: JsonObject): string | null {
  return typeof item.block === "string" ? item.block : null;
}

export function validateAnalyticalDataPackage(
  input: unknown,
  context: AnalyticalDataContext,
): AnalyticalDataValidationResult {
  if (!validateSchema(input)) {
    return { ok: false, stage: "schema", errors: formatAjvErrors(validateSchema.errors) };
  }

  const value = input as JsonObject;
  const errors: string[] = [];

  const exact: Array<[string, unknown, unknown]> = [
    ["run_id", value.run_id, context.runId],
    ["issuer_id", value.issuer_id, context.issuerId],
    ["security_id", value.security_id ?? null, context.securityId],
    ["dossier_id", value.dossier_id ?? null, context.dossierId],
    ["data_cutoff", value.data_cutoff, context.dataCutoff],
  ];
  for (const [name, actual, expected] of exact) {
    if (actual !== expected) errors.push(`${name} does not match authoritative run context`);
  }

  const sources = asObjects(value.sources);
  const evidence = asObjects(value.evidence);
  const conflicts = asObjects(value.conflicts);
  const gaps = asObjects(value.gaps);
  const assumptions = asObjects(value.assumptions);
  const blocks = asObjects(value.analytical_blocks);
  const changes = asObjects(value.material_changes);

  const idSpecs: Array<[string, JsonObject[], string]> = [
    ["source", sources, "source_id"],
    ["evidence", evidence, "evidence_id"],
    ["conflict", conflicts, "conflict_id"],
    ["gap", gaps, "gap_id"],
    ["assumption", assumptions, "assumption_id"],
    ["material_change", changes, "change_id"],
  ];
  for (const [label, items, key] of idSpecs) {
    for (const id of duplicateIds(items, key)) errors.push(`duplicate ${label} id ${id}`);
  }
  for (const code of duplicateIds(blocks, "block")) {
    errors.push(`duplicate analytical block ${code}`);
  }

  const sourceIds = stringSet(sources, "source_id");
  const evidenceIds = stringSet(evidence, "evidence_id");
  const evidenceById = new Map<string, JsonObject>();
  for (const item of evidence) {
    if (typeof item.evidence_id === "string") evidenceById.set(item.evidence_id, item);
  }
  const conflictIds = stringSet(conflicts, "conflict_id");
  const gapIds = stringSet(gaps, "gap_id");
  const assumptionIds = stringSet(assumptions, "assumption_id");
  const blockCodes = new Set(blocks.map(blockCode).filter((v): v is string => v !== null));

  for (const source of sources) {
    const id = source.source_id;
    const sourceDate = source.source_date;
    if (typeof sourceDate === "string" && compareIsoDate(sourceDate, context.dataCutoff) > 0) {
      errors.push(`source ${String(id)} is post-cutoff (${sourceDate} > ${context.dataCutoff})`);
    }
    const root = source.root_source_id;
    if (typeof root === "string" && !sourceIds.has(root)) {
      errors.push(`source ${String(id)} references unknown root_source_id ${root}`);
    }
    if (typeof root === "string" && root === id) {
      errors.push(`source ${String(id)} cannot reference itself as root_source_id`);
    }
  }

  const rootBySource = new Map<string, string>();
  for (const source of sources) {
    if (typeof source.source_id === "string" && typeof source.root_source_id === "string") {
      rootBySource.set(source.source_id, source.root_source_id);
    }
  }
  for (const source of sources) {
    if (typeof source.source_id !== "string") continue;
    const origin = source.source_id;
    const seen = new Set<string>([origin]);
    let current = origin;
    while (rootBySource.has(current)) {
      const next = rootBySource.get(current)!;
      if (!sourceIds.has(next)) break;
      if (seen.has(next)) {
        errors.push(`source root_source_id cycle detected from ${origin}`);
        break;
      }
      seen.add(next);
      current = next;
    }
  }

  for (const item of evidence) {
    const id = item.evidence_id;
    if (typeof item.source_id === "string" && !sourceIds.has(item.source_id)) {
      errors.push(`evidence ${String(id)} references unknown source_id ${item.source_id}`);
    }
    for (const datum of asObjects(item.numeric_data)) {
      const kind = datum.value_kind;
      if (kind === "RANGE" && typeof datum.min === "number" && typeof datum.max === "number" && datum.min > datum.max) {
        errors.push(`evidence ${String(id)} numeric range has min > max for ${String(datum.metric)}`);
      }
      const periodStart = datum.period_start;
      const periodEnd = datum.period_end;
      if (typeof periodStart === "string" && typeof periodEnd === "string" && periodStart > periodEnd) {
        errors.push(`evidence ${String(id)} numeric period has start > end for ${String(datum.metric)}`);
      }
      const hasTemporalAnchor =
        typeof datum.period_end === "string" || typeof datum.as_of_date === "string";
      if (!hasTemporalAnchor) {
        errors.push(`evidence ${String(id)} numeric datum ${String(datum.metric)} lacks period_end/as_of_date`);
      }
    }
  }

  for (const conflict of conflicts) {
    checkEvidenceRefs(errors, `conflict ${String(conflict.conflict_id)}`, conflict.evidence_ids, evidenceIds);
    if (
      conflict.status === "RESOLVED" &&
      (typeof conflict.resolution !== "string" || conflict.resolution.trim() === "")
    ) {
      errors.push(`resolved conflict ${String(conflict.conflict_id)} requires resolution text`);
    }
  }

  for (const gap of gaps) {
    checkEvidenceRefs(errors, `gap ${String(gap.gap_id)}`, gap.evidence_ids, evidenceIds);
    if (
      (gap.status === "EXHAUSTED_NOT_ASSESSABLE" || gap.status === "BLOCKED_INPUT") &&
      (typeof gap.why_unresolved !== "string" || gap.why_unresolved.trim() === "")
    ) {
      errors.push(`${String(gap.status)} gap ${String(gap.gap_id)} requires why_unresolved`);
    }
  }

  for (const assumption of assumptions) {
    checkEvidenceRefs(
      errors,
      `assumption ${String(assumption.assumption_id)}`,
      assumption.basis_evidence_ids,
      evidenceIds,
    );
  }

  const dna = isObject(value.company_economic_dna) ? value.company_economic_dna : {};
  for (const [dimension, raw] of Object.entries(dna)) {
    if (!isObject(raw)) continue;
    checkEvidenceRefs(errors, `company_economic_dna.${dimension}`, raw.evidence_ids, evidenceIds);
  }

  for (const block of blocks) {
    const code = String(block.block);
    checkEvidenceRefs(errors, `block ${code} supporting`, block.supporting_evidence_ids, evidenceIds);
    checkEvidenceRefs(errors, `block ${code} counterevidence`, block.counterevidence_ids, evidenceIds);
    for (const evidenceId of [
      ...refsFromArray(block.supporting_evidence_ids),
      ...refsFromArray(block.counterevidence_ids),
    ]) {
      const evidenceItem = evidenceById.get(evidenceId);
      if (evidenceItem && !refsFromArray(evidenceItem.block_relevance).includes(code)) {
        errors.push(`block ${code} uses evidence_id ${evidenceId} without matching block_relevance`);
      }
    }
    if (
      block.status === "COMPLETE" &&
      refsFromArray(block.supporting_evidence_ids).length === 0 &&
      refsFromArray(block.counterevidence_ids).length === 0
    ) {
      errors.push(`block ${code} cannot be COMPLETE without traceable evidence`);
    }

    for (const id of refsFromArray(block.conflict_ids)) {
      if (!conflictIds.has(id)) errors.push(`block ${code} references unknown conflict_id ${id}`);
    }
    const blockingConflict = conflicts.some((conflict) =>
      refsFromArray(conflict.affected_blocks).includes(code) &&
      (conflict.status === "OPEN" || conflict.status === "UNRESOLVED_BLOCKING")
    );
    if (blockingConflict && block.status === "COMPLETE") {
      errors.push(`block ${code} cannot be COMPLETE with an open/blocking conflict`);
    }
    for (const id of refsFromArray(block.gap_ids)) {
      if (!gapIds.has(id)) errors.push(`block ${code} references unknown gap_id ${id}`);
    }
    for (const id of refsFromArray(block.assumption_ids)) {
      if (!assumptionIds.has(id)) errors.push(`block ${code} references unknown assumption_id ${id}`);
    }
    for (const upstream of refsFromArray(block.upstream_block_refs)) {
      if (!blockCodes.has(upstream)) errors.push(`block ${code} references absent upstream block ${upstream}`);
      if (upstream === code) errors.push(`block ${code} cannot depend on itself`);
    }

    for (const linkId of duplicateIds(asObjects(block.causal_links), "link_id")) {
      errors.push(`block ${code} has duplicate causal link_id ${linkId}`);
    }
    for (const link of asObjects(block.causal_links)) {
      checkEvidenceRefs(errors, `block ${code} causal link ${String(link.link_id)}`, link.evidence_ids, evidenceIds);
      checkEvidenceRefs(errors, `block ${code} causal counterevidence ${String(link.link_id)}`, link.counterevidence_ids, evidenceIds);
      for (const evidenceId of [
        ...refsFromArray(link.evidence_ids),
        ...refsFromArray(link.counterevidence_ids),
      ]) {
        const evidenceItem = evidenceById.get(evidenceId);
        if (evidenceItem && !refsFromArray(evidenceItem.block_relevance).includes(code)) {
          errors.push(`block ${code} causal link ${String(link.link_id)} uses evidence_id ${evidenceId} without matching block_relevance`);
        }
      }
      if (link.status === "SUPPORTED" && refsFromArray(link.evidence_ids).length === 0) {
        errors.push(`block ${code} causal link ${String(link.link_id)} cannot be SUPPORTED without evidence`);
      }
      if (
        link.status === "MIXED" &&
        (refsFromArray(link.evidence_ids).length === 0 ||
          refsFromArray(link.counterevidence_ids).length === 0)
      ) {
        errors.push(`block ${code} causal link ${String(link.link_id)} MIXED requires evidence and counterevidence`);
      }
    }
    for (const overlayName of duplicateIds(asObjects(block.sector_overlays), "overlay")) {
      errors.push(`block ${code} has duplicate sector overlay ${overlayName}`);
    }
    for (const overlay of asObjects(block.sector_overlays)) {
      checkEvidenceRefs(errors, `block ${code} sector overlay ${String(overlay.overlay)}`, overlay.evidence_ids, evidenceIds);
      for (const evidenceId of refsFromArray(overlay.evidence_ids)) {
        const evidenceItem = evidenceById.get(evidenceId);
        if (evidenceItem && !refsFromArray(evidenceItem.block_relevance).includes(code)) {
          errors.push(`block ${code} sector overlay ${String(overlay.overlay)} uses evidence_id ${evidenceId} without matching block_relevance`);
        }
      }
      if (overlay.status === "REQUIRED_MISSING" && block.status === "COMPLETE") {
        errors.push(`block ${code} cannot be COMPLETE with REQUIRED_MISSING sector overlay`);
      }
    }

    const criticalOpenGap = gaps.some((gap) =>
      gap.affected_block === code &&
      gap.materiality === "CRITICAL" &&
      (
        gap.status === "OPEN" ||
        gap.status === "BLOCKED_INPUT" ||
        gap.status === "EXHAUSTED_NOT_ASSESSABLE"
      )
    );
    if (criticalOpenGap && block.status === "COMPLETE") {
      errors.push(`block ${code} cannot be COMPLETE with a critical unresolved gap`);
    }
    if (block.status === "NOT_ASSESSABLE") {
      const hasTraceableNotAssessableGap = gaps.some((gap) =>
        gap.affected_block === code &&
        (gap.status === "EXHAUSTED_NOT_ASSESSABLE" || gap.status === "BLOCKED_INPUT")
      );
      if (!hasTraceableNotAssessableGap) {
        errors.push(`block ${code} NOT_ASSESSABLE requires a traceable exhausted/blocked gap`);
      }
    }
  }

  for (const change of changes) {
    checkEvidenceRefs(
      errors,
      `material change ${String(change.change_id)}`,
      change.trigger_evidence_ids,
      evidenceIds,
    );
    const checklist = isObject(change.revalidation) ? change.revalidation : {};
    if (checklist.status === "PASS") {
      for (const key of [
        "reopened_material_evidence",
        "verified_primary_sources",
        "searched_disconfirming_evidence",
        "tested_best_alternative_explanation",
        "reconciled_downstream_blocks",
        "recorded_prior_state_change_reason",
      ]) {
        if (checklist[key] !== true) {
          errors.push(`material change ${String(change.change_id)} cannot PASS revalidation with ${key} != true`);
        }
      }
    }
  }

  const serial = value.serial_acquirer_profile;
  if (isObject(serial)) {
    checkEvidenceRefs(errors, "serial_acquirer_profile", serial.evidence_ids, evidenceIds);
    if (serial.applicability === "APPLICABLE") {
      const requiredNarrative = [
        "organic_vs_acquired_growth",
        "purchase_price_discipline",
        "integration_model",
        "goodwill_intangibles_economics",
        "dilution_funding",
        "acquisition_roi",
        "deployment_runway",
      ];
      for (const key of requiredNarrative) {
        if (typeof serial[key] !== "string" || (serial[key] as string).trim() === "") {
          errors.push(`serial_acquirer_profile APPLICABLE requires ${key}`);
        }
      }
      if (refsFromArray(serial.evidence_ids).length === 0) {
        errors.push("serial_acquirer_profile APPLICABLE requires evidence_ids");
      }
    }
  }

  if (errors.length > 0) return { ok: false, stage: "semantic", errors };
  return { ok: true, value };
}

export { validateSchema };
