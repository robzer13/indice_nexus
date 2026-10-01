export const INITIAL_DD_INPUT_BLOCKS = [
  "IDENTITY_PIT_INPUTS",
  "BUSINESS_MODEL_INPUTS",
  "MOAT_INPUTS",
  "RUNWAY_INPUTS",
  "RETURN_QUALITY_INPUTS",
  "FCF_FORENSIC_INPUTS",
  "CAPITAL_ALLOCATION_INPUTS",
  "MANAGEMENT_GOVERNANCE_INPUTS",
  "OUTSIDE_VIEW_INPUTS",
  "RISK_RESILIENCE_INPUTS",
  "VALUATION_INPUTS",
] as const;

export type ValidationIssue = {
  code: string;
  path: string;
  message: string;
};

export type CrossArtifactValidationResult = {
  ok: boolean;
  issues: ValidationIssue[];
};

type PayloadContext = {
  run_id: string;
  stage_code: string;
  stage_revision: number;
  data_cutoff: string;
  contract_set_sha256: string;
};

type SourceRow = {
  source_id: string;
  source_date: string;
  root_source_id: string | null;
  included_in_evidence_ledger: boolean;
};

type SourceManifest = {
  context: PayloadContext;
  manifest_version: string;
  sources: SourceRow[];
};

type EvidenceRow = {
  evidence_id: string;
  source_id: string;
  root_source_id: string | null;
  source_date: string;
  data_cutoff: string;
};

type EvidenceLedger = {
  context: PayloadContext;
  ledger_version: string;
  evidence: EvidenceRow[];
};

type ConflictRow = {
  conflict_id: string;
  evidence_id_a: string;
  evidence_id_b: string;
  resolution: "RESOLVED" | "UNRESOLVED";
};

type ConflictLedger = {
  context: PayloadContext;
  ledger_version: string;
  conflicts: ConflictRow[];
};

type CalculationRow = {
  calculation_id: string;
  input_evidence_ids: string[];
};

type CalculationLedger = {
  context: PayloadContext;
  ledger_version: string;
  calculations: CalculationRow[];
};

type AssumptionRow = {
  assumption_id: string;
  epistemic_type: string;
};

type AssumptionRegister = {
  context: PayloadContext;
  register_version: string;
  assumptions: AssumptionRow[];
};

type HypothesisRow = {
  hypothesis_id: string;
  supporting_evidence_ids: string[];
  contradicting_evidence_ids: string[];
};

type HypothesisRegister = {
  context: PayloadContext;
  register_version: string;
  hypotheses: HypothesisRow[];
};

type GapRow = {
  gap_id: string;
};

type GapRegister = {
  context: PayloadContext;
  register_version: string;
  gaps: GapRow[];
};

type SufficiencyBlock = {
  block_id: string;
  applicability: "REQUIRED" | "CONDITIONAL" | "NOT_APPLICABLE";
  dd_input_status: "SUFFICIENT" | "INSUFFICIENT" | "NOT_APPLICABLE";
  mandatory_coverage: boolean | null;
  evidence_adequacy: boolean | null;
  material_blocking_gaps: string[];
  key_evidence_ids: string[];
  key_conflict_ids: string[];
};

type SufficiencyRecord = {
  context: PayloadContext;
  record_version: string;
  scope_type: "INITIAL" | "FULL_REFRESH_REQUIRED" | "TARGETED_REFRESH";
  blocks: SufficiencyBlock[];
  critical_blockers: string[];
  ready_for_deep_dive: "YES" | "NO";
};

type InputLock = {
  context: PayloadContext;
  evidence_ledger_version: string;
  source_manifest_version: string;
  dd_input_sufficiency_version: string;
  open_noncritical_gaps: string[];
  open_material_conflicts: string[];
  critical_blockers: string[];
  ready_for_deep_dive: "YES" | "NO";
};

type VerdictField = {
  evidence_ids: string[];
  calculation_ids: string[];
  assumption_ids: string[];
  conflict_ids: string[];
};

type AnalyticalBlockOutput = {
  context: PayloadContext;
  block_id: string;
  data_cutoff: string;
  supporting_evidence_ids: string[];
  contradicting_evidence_ids: string[];
  calculation_ids: string[];
  material_assumption_ids: string[];
  conflict_ids: string[];
  canonical_verdict_fields: VerdictField[];
};

export type AnalyticalCoreBundle = {
  run: {
    run_id: string;
    data_cutoff: string;
    contract_set_sha256: string;
  };
  stage: {
    stage_code: string;
    stage_revision: number;
  };
  sourceManifest: SourceManifest;
  evidenceLedger: EvidenceLedger;
  conflictLedger: ConflictLedger;
  calculationLedger?: CalculationLedger;
  assumptionRegister?: AssumptionRegister;
  hypothesisRegister?: HypothesisRegister;
  gapRegister?: GapRegister;
  sufficiencyRecord?: SufficiencyRecord;
  inputLock?: InputLock;
  analyticalBlocks?: AnalyticalBlockOutput[];
};

function issue(
  issues: ValidationIssue[],
  code: string,
  path: string,
  message: string,
) {
  issues.push({ code, path, message });
}

function uniqueMap<T>(
  rows: T[],
  idOf: (row: T) => string,
  label: string,
  issues: ValidationIssue[],
): Map<string, T> {
  const map = new Map<string, T>();
  rows.forEach((row, index) => {
    const id = idOf(row);
    if (map.has(id)) {
      issue(
        issues,
        "DUPLICATE_ID",
        `${label}[${index}]`,
        `Duplicate ID ${id}`,
      );
    } else {
      map.set(id, row);
    }
  });
  return map;
}

function isAfterCutoff(date: string, cutoff: string): boolean {
  if (date === "UNKNOWN_DATE") return false;
  return date > cutoff;
}

function validateContext(
  context: PayloadContext,
  bundle: AnalyticalCoreBundle,
  path: string,
  issues: ValidationIssue[],
) {
  if (context.run_id !== bundle.run.run_id) {
    issue(issues, "RUN_ID_MISMATCH", `${path}.run_id`, "Payload run_id does not match registry run.");
  }
  if (context.stage_code !== bundle.stage.stage_code) {
    issue(issues, "STAGE_CODE_MISMATCH", `${path}.stage_code`, "Payload stage_code does not match owning stage.");
  }
  if (context.stage_revision !== bundle.stage.stage_revision) {
    issue(issues, "STAGE_REVISION_MISMATCH", `${path}.stage_revision`, "Payload stage_revision does not match registry stage revision.");
  }
  if (context.data_cutoff !== bundle.run.data_cutoff) {
    issue(issues, "DATA_CUTOFF_MISMATCH", `${path}.data_cutoff`, "Payload data_cutoff does not match run cutoff.");
  }
  if (context.contract_set_sha256 !== bundle.run.contract_set_sha256) {
    issue(issues, "CONTRACT_SET_MISMATCH", `${path}.contract_set_sha256`, "Payload contract set hash does not match run pins.");
  }
}

function requireIds(
  ids: string[],
  known: Map<string, unknown>,
  path: string,
  kind: string,
  issues: ValidationIssue[],
) {
  ids.forEach((id, index) => {
    if (!known.has(id)) {
      issue(
        issues,
        `UNKNOWN_${kind}_ID`,
        `${path}[${index}]`,
        `Unknown ${kind} ID: ${id}`,
      );
    }
  });
}

export function validateAnalyticalCoreBundle(
  bundle: AnalyticalCoreBundle,
): CrossArtifactValidationResult {
  const issues: ValidationIssue[] = [];

  validateContext(bundle.sourceManifest.context, bundle, "sourceManifest.context", issues);
  validateContext(bundle.evidenceLedger.context, bundle, "evidenceLedger.context", issues);
  validateContext(bundle.conflictLedger.context, bundle, "conflictLedger.context", issues);
  if (bundle.calculationLedger) validateContext(bundle.calculationLedger.context, bundle, "calculationLedger.context", issues);
  if (bundle.assumptionRegister) validateContext(bundle.assumptionRegister.context, bundle, "assumptionRegister.context", issues);
  if (bundle.hypothesisRegister) validateContext(bundle.hypothesisRegister.context, bundle, "hypothesisRegister.context", issues);
  if (bundle.gapRegister) validateContext(bundle.gapRegister.context, bundle, "gapRegister.context", issues);
  if (bundle.sufficiencyRecord) validateContext(bundle.sufficiencyRecord.context, bundle, "sufficiencyRecord.context", issues);
  if (bundle.inputLock) validateContext(bundle.inputLock.context, bundle, "inputLock.context", issues);

  const sources = uniqueMap(
    bundle.sourceManifest.sources,
    (row) => row.source_id,
    "sourceManifest.sources",
    issues,
  );
  const evidence = uniqueMap(
    bundle.evidenceLedger.evidence,
    (row) => row.evidence_id,
    "evidenceLedger.evidence",
    issues,
  );
  const conflicts = uniqueMap(
    bundle.conflictLedger.conflicts,
    (row) => row.conflict_id,
    "conflictLedger.conflicts",
    issues,
  );
  const calculations = uniqueMap(
    bundle.calculationLedger?.calculations ?? [],
    (row) => row.calculation_id,
    "calculationLedger.calculations",
    issues,
  );
  const assumptions = uniqueMap(
    bundle.assumptionRegister?.assumptions ?? [],
    (row) => row.assumption_id,
    "assumptionRegister.assumptions",
    issues,
  );
  const gaps = uniqueMap(
    bundle.gapRegister?.gaps ?? [],
    (row) => row.gap_id,
    "gapRegister.gaps",
    issues,
  );

  bundle.sourceManifest.sources.forEach((row, index) => {
    if (row.root_source_id && !sources.has(row.root_source_id)) {
      issue(
        issues,
        "UNKNOWN_ROOT_SOURCE_ID",
        `sourceManifest.sources[${index}].root_source_id`,
        `Unknown root source ID: ${row.root_source_id}`,
      );
    }
  });

  bundle.evidenceLedger.evidence.forEach((row, index) => {
    const source = sources.get(row.source_id);
    if (!source) {
      issue(
        issues,
        "UNKNOWN_SOURCE_ID",
        `evidenceLedger.evidence[${index}].source_id`,
        `Unknown source ID: ${row.source_id}`,
      );
    } else if (!source.included_in_evidence_ledger) {
      issue(
        issues,
        "SOURCE_MANIFEST_EVIDENCE_FLAG_MISMATCH",
        `evidenceLedger.evidence[${index}].source_id`,
        "Evidence references a Source Manifest row not marked as included in the Evidence Ledger.",
      );
    }

    if (row.root_source_id && !sources.has(row.root_source_id)) {
      issue(
        issues,
        "UNKNOWN_ROOT_SOURCE_ID",
        `evidenceLedger.evidence[${index}].root_source_id`,
        `Unknown root source ID: ${row.root_source_id}`,
      );
    }

    if (row.data_cutoff !== bundle.run.data_cutoff) {
      issue(
        issues,
        "EVIDENCE_CUTOFF_MISMATCH",
        `evidenceLedger.evidence[${index}].data_cutoff`,
        "Evidence cutoff does not match run cutoff.",
      );
    }

    if (isAfterCutoff(row.source_date, bundle.run.data_cutoff)) {
      issue(
        issues,
        "POST_CUTOFF_EVIDENCE",
        `evidenceLedger.evidence[${index}].source_date`,
        "Post-cutoff source cannot be admitted as current-run evidence.",
      );
    }

    if (source && isAfterCutoff(source.source_date, bundle.run.data_cutoff)) {
      issue(
        issues,
        "POST_CUTOFF_SOURCE_ADMITTED",
        `sourceManifest.sources[${row.source_id}].source_date`,
        "Source Manifest source used as evidence is post-cutoff.",
      );
    }
  });

  bundle.conflictLedger.conflicts.forEach((row, index) => {
    requireIds(
      [row.evidence_id_a, row.evidence_id_b],
      evidence,
      `conflictLedger.conflicts[${index}]`,
      "EVIDENCE",
      issues,
    );
  });

  bundle.calculationLedger?.calculations.forEach((row, index) => {
    requireIds(
      row.input_evidence_ids,
      evidence,
      `calculationLedger.calculations[${index}].input_evidence_ids`,
      "EVIDENCE",
      issues,
    );
  });

  bundle.assumptionRegister?.assumptions.forEach((row, index) => {
    if (row.epistemic_type !== "ASSUMPTION") {
      issue(
        issues,
        "ASSUMPTION_EPISTEMIC_LEAKAGE",
        `assumptionRegister.assumptions[${index}].epistemic_type`,
        "Material Assumption Register rows must remain ASSUMPTION.",
      );
    }
  });

  bundle.hypothesisRegister?.hypotheses.forEach((row, index) => {
    requireIds(
      row.supporting_evidence_ids,
      evidence,
      `hypothesisRegister.hypotheses[${index}].supporting_evidence_ids`,
      "EVIDENCE",
      issues,
    );
    requireIds(
      row.contradicting_evidence_ids,
      evidence,
      `hypothesisRegister.hypotheses[${index}].contradicting_evidence_ids`,
      "EVIDENCE",
      issues,
    );
  });

  if (bundle.sufficiencyRecord) {
    const s = bundle.sufficiencyRecord;
    const blockIds = new Set<string>();
    s.blocks.forEach((block, index) => {
      if (blockIds.has(block.block_id)) {
        issue(
          issues,
          "DUPLICATE_SUFFICIENCY_BLOCK",
          `sufficiencyRecord.blocks[${index}].block_id`,
          `Duplicate DD input block: ${block.block_id}`,
        );
      }
      blockIds.add(block.block_id);

      requireIds(
        block.key_evidence_ids,
        evidence,
        `sufficiencyRecord.blocks[${index}].key_evidence_ids`,
        "EVIDENCE",
        issues,
      );
      requireIds(
        block.key_conflict_ids,
        conflicts,
        `sufficiencyRecord.blocks[${index}].key_conflict_ids`,
        "CONFLICT",
        issues,
      );
      requireIds(
        block.material_blocking_gaps,
        gaps,
        `sufficiencyRecord.blocks[${index}].material_blocking_gaps`,
        "GAP",
        issues,
      );

      if (block.dd_input_status === "SUFFICIENT") {
        if (block.mandatory_coverage !== true) {
          issue(
            issues,
            "SUFFICIENT_WITHOUT_MANDATORY_COVERAGE",
            `sufficiencyRecord.blocks[${index}].mandatory_coverage`,
            "SUFFICIENT requires mandatory coverage.",
          );
        }
        if (block.evidence_adequacy !== true) {
          issue(
            issues,
            "SUFFICIENT_WITHOUT_EVIDENCE_ADEQUACY",
            `sufficiencyRecord.blocks[${index}].evidence_adequacy`,
            "SUFFICIENT requires adequate evidence.",
          );
        }
        if (block.material_blocking_gaps.length > 0) {
          issue(
            issues,
            "SUFFICIENT_WITH_BLOCKING_GAP",
            `sufficiencyRecord.blocks[${index}].material_blocking_gaps`,
            "SUFFICIENT cannot retain a material blocking gap.",
          );
        }
      }

      if (
        block.dd_input_status === "NOT_APPLICABLE" &&
        block.applicability !== "NOT_APPLICABLE"
      ) {
        issue(
          issues,
          "INVALID_NOT_APPLICABLE_SUFFICIENCY",
          `sufficiencyRecord.blocks[${index}]`,
          "NOT_APPLICABLE DD input status requires NOT_APPLICABLE applicability.",
        );
      }
    });

    if (s.scope_type !== "TARGETED_REFRESH") {
      INITIAL_DD_INPUT_BLOCKS.forEach((requiredBlock) => {
        if (!blockIds.has(requiredBlock)) {
          issue(
            issues,
            "MISSING_REQUIRED_DD_INPUT_BLOCK",
            "sufficiencyRecord.blocks",
            `Missing required DD input block: ${requiredBlock}`,
          );
        }
      });
      if (blockIds.size !== INITIAL_DD_INPUT_BLOCKS.length) {
        issue(
          issues,
          "INVALID_INITIAL_DD_INPUT_BLOCK_SET",
          "sufficiencyRecord.blocks",
          "Initial/full refresh sufficiency must contain exactly the 11 frozen DD input blocks.",
        );
      }
    }

    const allAdmissible = s.blocks.every(
      (block) =>
        block.dd_input_status === "SUFFICIENT" ||
        block.dd_input_status === "NOT_APPLICABLE",
    );
    if (
      s.ready_for_deep_dive === "YES" &&
      (!allAdmissible || s.critical_blockers.length > 0)
    ) {
      issue(
        issues,
        "UNSUPPORTED_READY_FOR_DEEP_DIVE",
        "sufficiencyRecord.ready_for_deep_dive",
        "READY_FOR_DEEP_DIVE cannot be YES with insufficient blocks or critical blockers.",
      );
    }
  }

  if (bundle.inputLock) {
    const lock = bundle.inputLock;

    if (lock.evidence_ledger_version !== bundle.evidenceLedger.ledger_version) {
      issue(
        issues,
        "INPUT_LOCK_EVIDENCE_VERSION_MISMATCH",
        "inputLock.evidence_ledger_version",
        "Analysis Input Lock does not pin the loaded Evidence Ledger version.",
      );
    }
    if (lock.source_manifest_version !== bundle.sourceManifest.manifest_version) {
      issue(
        issues,
        "INPUT_LOCK_SOURCE_VERSION_MISMATCH",
        "inputLock.source_manifest_version",
        "Analysis Input Lock does not pin the loaded Source Manifest version.",
      );
    }
    if (
      bundle.sufficiencyRecord &&
      lock.dd_input_sufficiency_version !== bundle.sufficiencyRecord.record_version
    ) {
      issue(
        issues,
        "INPUT_LOCK_SUFFICIENCY_VERSION_MISMATCH",
        "inputLock.dd_input_sufficiency_version",
        "Analysis Input Lock does not pin the loaded sufficiency record version.",
      );
    }

    requireIds(
      lock.open_noncritical_gaps,
      gaps,
      "inputLock.open_noncritical_gaps",
      "GAP",
      issues,
    );
    requireIds(
      lock.open_material_conflicts,
      conflicts,
      "inputLock.open_material_conflicts",
      "CONFLICT",
      issues,
    );

    if (
      bundle.sufficiencyRecord &&
      lock.ready_for_deep_dive !== bundle.sufficiencyRecord.ready_for_deep_dive
    ) {
      issue(
        issues,
        "INPUT_LOCK_READY_STATE_MISMATCH",
        "inputLock.ready_for_deep_dive",
        "Input Lock and sufficiency record disagree on READY_FOR_DEEP_DIVE.",
      );
    }

    if (
      lock.ready_for_deep_dive === "YES" &&
      (lock.critical_blockers.length > 0 || lock.open_material_conflicts.length > 0)
    ) {
      issue(
        issues,
        "INPUT_LOCK_UNSUPPORTED_READY_STATE",
        "inputLock.ready_for_deep_dive",
        "Input Lock cannot be ready with critical blockers or open material conflicts.",
      );
    }
  }

  bundle.analyticalBlocks?.forEach((block, index) => {
    validateContext(
      block.context,
      bundle,
      `analyticalBlocks[${index}].context`,
      issues,
    );
    if (block.data_cutoff !== bundle.run.data_cutoff) {
      issue(
        issues,
        "BLOCK_CUTOFF_MISMATCH",
        `analyticalBlocks[${index}].data_cutoff`,
        "Analytical block cutoff does not match run cutoff.",
      );
    }

    requireIds(
      [...block.supporting_evidence_ids, ...block.contradicting_evidence_ids],
      evidence,
      `analyticalBlocks[${index}].evidence_ids`,
      "EVIDENCE",
      issues,
    );
    requireIds(
      block.calculation_ids,
      calculations,
      `analyticalBlocks[${index}].calculation_ids`,
      "CALCULATION",
      issues,
    );
    requireIds(
      block.material_assumption_ids,
      assumptions,
      `analyticalBlocks[${index}].material_assumption_ids`,
      "ASSUMPTION",
      issues,
    );
    requireIds(
      block.conflict_ids,
      conflicts,
      `analyticalBlocks[${index}].conflict_ids`,
      "CONFLICT",
      issues,
    );

    block.canonical_verdict_fields.forEach((field, fieldIndex) => {
      requireIds(
        field.evidence_ids,
        evidence,
        `analyticalBlocks[${index}].canonical_verdict_fields[${fieldIndex}].evidence_ids`,
        "EVIDENCE",
        issues,
      );
      requireIds(
        field.calculation_ids,
        calculations,
        `analyticalBlocks[${index}].canonical_verdict_fields[${fieldIndex}].calculation_ids`,
        "CALCULATION",
        issues,
      );
      requireIds(
        field.assumption_ids,
        assumptions,
        `analyticalBlocks[${index}].canonical_verdict_fields[${fieldIndex}].assumption_ids`,
        "ASSUMPTION",
        issues,
      );
      requireIds(
        field.conflict_ids,
        conflicts,
        `analyticalBlocks[${index}].canonical_verdict_fields[${fieldIndex}].conflict_ids`,
        "CONFLICT",
        issues,
      );
    });
  });

  return {
    ok: issues.length === 0,
    issues,
  };
}
