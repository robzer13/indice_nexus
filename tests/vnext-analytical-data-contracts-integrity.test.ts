import assert from "node:assert/strict";
import test from "node:test";

import { validateAnalyticalCoreBundle } from "../runtime/vnext/analytical-data-contracts";

const sha = "a".repeat(64);

function ctx() {
  return {
    run_id: "RUN-001",
    stage_code: "RESEARCH",
    stage_revision: 0,
    data_cutoff: "2026-10-01",
    contract_set_sha256: sha,
  };
}

function bundle() {
  return {
    run: {
      run_id: "RUN-001",
      data_cutoff: "2026-10-01",
      contract_set_sha256: sha,
    },
    stage: { stage_code: "RESEARCH", stage_revision: 0 },
    sourceManifest: {
      context: ctx(),
      manifest_version: "SM-1",
      sources: [
        {
          source_id: "S-001",
          source_date: "2026-09-30",
          root_source_id: null,
          included_in_evidence_ledger: true,
        },
      ],
    },
    evidenceLedger: {
      context: ctx(),
      ledger_version: "EL-1",
      evidence: [
        {
          evidence_id: "E-001",
          source_id: "S-001",
          root_source_id: null,
          source_date: "2026-09-30",
          data_cutoff: "2026-10-01",
        },
      ],
    },
    conflictLedger: {
      context: ctx(),
      ledger_version: "CL-1",
      conflicts: [],
    },
    calculationLedger: {
      context: ctx(),
      ledger_version: "CALC-1",
      calculations: [],
    },
    assumptionRegister: {
      context: ctx(),
      register_version: "AR-1",
      assumptions: [],
    },
    hypothesisRegister: {
      context: ctx(),
      register_version: "HR-1",
      hypotheses: [],
    },
    gapRegister: {
      context: ctx(),
      register_version: "GR-1",
      gaps: [],
    },
    sufficiencyRecord: {
      context: ctx(),
      record_version: "DD-1",
      scope_type: "TARGETED_REFRESH",
      blocks: [
        {
          block_id: "MOAT_INPUTS",
          applicability: "REQUIRED",
          dd_input_status: "SUFFICIENT",
          mandatory_coverage: true,
          evidence_adequacy: true,
          material_blocking_gaps: [],
          key_evidence_ids: ["E-001"],
          key_conflict_ids: [],
        },
      ],
      critical_blockers: [],
      ready_for_deep_dive: "YES",
    },
    inputLock: {
      context: ctx(),
      evidence_ledger_version: "EL-1",
      source_manifest_version: "SM-1",
      dd_input_sufficiency_version: "DD-1",
      open_noncritical_gaps: [],
      open_material_conflicts: [],
      critical_blockers: [],
      ready_for_deep_dive: "YES",
    },
  };
}

test("clean targeted-refresh core bundle passes integrity validation", () => {
  const result = validateAnalyticalCoreBundle(bundle() as never);
  assert.equal(result.ok, true, JSON.stringify(result.issues, null, 2));
});

test("post-cutoff evidence fails closed", () => {
  const value = bundle();
  value.sourceManifest.sources[0].source_date = "2026-10-02";
  value.evidenceLedger.evidence[0].source_date = "2026-10-02";

  const result = validateAnalyticalCoreBundle(value as never);
  assert.equal(result.ok, false);
  assert.ok(
    result.issues.some((item) => item.code === "POST_CUTOFF_EVIDENCE"),
  );
});

test("unknown Evidence ID fails closed", () => {
  const value = bundle();
  value.sufficiencyRecord.blocks[0].key_evidence_ids = ["E-MISSING"];

  const result = validateAnalyticalCoreBundle(value as never);
  assert.equal(result.ok, false);
  assert.ok(
    result.issues.some((item) => item.code === "UNKNOWN_EVIDENCE_ID"),
  );
});

test("SUFFICIENT cannot coexist with a material blocking gap", () => {
  const value = bundle();
  value.gapRegister.gaps = [{ gap_id: "G-001" }];
  value.sufficiencyRecord.blocks[0].material_blocking_gaps = ["G-001"];

  const result = validateAnalyticalCoreBundle(value as never);
  assert.equal(result.ok, false);
  assert.ok(
    result.issues.some((item) => item.code === "SUFFICIENT_WITH_BLOCKING_GAP"),
  );
});

test("READY_FOR_DEEP_DIVE cannot be YES with an insufficient block", () => {
  const value = bundle();
  value.sufficiencyRecord.blocks[0].dd_input_status = "INSUFFICIENT";

  const result = validateAnalyticalCoreBundle(value as never);
  assert.equal(result.ok, false);
  assert.ok(
    result.issues.some(
      (item) => item.code === "UNSUPPORTED_READY_FOR_DEEP_DIVE",
    ),
  );
});

test("initial analysis fails when any frozen DD input block is missing", () => {
  const value = bundle();
  value.sufficiencyRecord.scope_type = "INITIAL";

  const result = validateAnalyticalCoreBundle(value as never);
  assert.equal(result.ok, false);
  assert.ok(
    result.issues.some(
      (item) => item.code === "MISSING_REQUIRED_DD_INPUT_BLOCK",
    ),
  );
});

test("Analysis Input Lock must pin the loaded Evidence Ledger version", () => {
  const value = bundle();
  value.inputLock.evidence_ledger_version = "EL-OLD";

  const result = validateAnalyticalCoreBundle(value as never);
  assert.equal(result.ok, false);
  assert.ok(
    result.issues.some(
      (item) => item.code === "INPUT_LOCK_EVIDENCE_VERSION_MISMATCH",
    ),
  );
});

test("material assumption cannot masquerade as reported evidence", () => {
  const value = bundle();
  value.assumptionRegister.assumptions = [
    { assumption_id: "A-001", epistemic_type: "REPORTED" },
  ];

  const result = validateAnalyticalCoreBundle(value as never);
  assert.equal(result.ok, false);
  assert.ok(
    result.issues.some(
      (item) => item.code === "ASSUMPTION_EPISTEMIC_LEAKAGE",
    ),
  );
});

test("Input Lock cannot be ready with open material conflicts", () => {
  const value = bundle();
  value.conflictLedger.conflicts = [
    {
      conflict_id: "CF-001",
      evidence_id_a: "E-001",
      evidence_id_b: "E-001",
      resolution: "UNRESOLVED",
    },
  ];
  value.inputLock.open_material_conflicts = ["CF-001"];

  const result = validateAnalyticalCoreBundle(value as never);
  assert.equal(result.ok, false);
  assert.ok(
    result.issues.some(
      (item) => item.code === "INPUT_LOCK_UNSUPPORTED_READY_STATE",
    ),
  );
});
