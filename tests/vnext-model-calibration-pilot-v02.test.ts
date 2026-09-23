import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  GATE18_PHASE_B_V02_GENERATION_SCHEMA_ID,
  GATE18_PHASE_B_V02_MODULE_ID,
  GATE18_PHASE_B_V02_SCOPE,
  GATE18_PHASE_B_V02_SYSTEM_PROMPT,
  buildGate18PhaseBV02MaximalOutputFixture,
  buildVerifiedGate18V02MoatPacket,
  gate18PhaseBV02OutputSchema,
} from "../runtime/vnext/model-calibration-pilot-v02";
import type {
  Gate18ArtifactPin,
  Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

type EvidenceArrayKey =
  | "evidence_items"
  | "items"
  | "evidence";

interface Variant {
  name: string;
  evidenceArrayKey: EvidenceArrayKey;
  evidenceIdKey: "evidence_id" | "id";
  moduleKey: "used_in" | "blocks";
  valueKey: "value_statement" | "value";
  conflictIdKey: "conflict_id" | "id";
  conflictScope: "MOAT_INPUTS" | "MOAT";
  omitTopLevelVersion?: boolean;
}

function buildVariant(variant: Variant) {
  const runId = `run-${variant.name}`;
  const dataCutoff = "2026-09-19";

  const scopedEvidence: Record<string, unknown> = {
    [variant.evidenceIdKey]: "E-001",
    claim_id: "CL-001",
    [variant.moduleKey]: ["MOAT_INPUTS"],
    [variant.valueKey]:
      variant.valueKey === "value"
        ? { statement: "Scoped evidence" }
        : "Scoped evidence",
    period: "FY2025",
    sources: ["SRC-001"],
    source_class: "S1",
    claim_fit: "HIGH",
    epistemic_type: "REPORTED",
  };

  const conflictReferencedEvidence: Record<string, unknown> = {
    [variant.evidenceIdKey]: "E-002",
    claim_id: "CL-002",
    [variant.moduleKey]: ["RUNWAY_INPUTS"],
    [variant.valueKey]: "Conflict-referenced evidence",
    period: "FY2025",
    source: "SRC-002",
    source_class: "S1",
    claim_fit: "HIGH",
    epistemic_type: "REPORTED",
  };

  const excludedEvidence: Record<string, unknown> = {
    [variant.evidenceIdKey]: "E-003",
    claim_id: "CL-003",
    [variant.moduleKey]: ["VALUATION_INPUTS"],
    [variant.valueKey]: "Must stay outside MOAT packet",
    period: "FY2025",
    source: "SRC-003",
    source_class: "S1",
    claim_fit: "HIGH",
    epistemic_type: "REPORTED",
  };

  const evidenceLedger = JSON.stringify({
    artifact_schema_version: "2.0.0",
    artifact_type: "EVIDENCE_LEDGER",
    run_id: runId,
    data_cutoff: dataCutoff,
    ...(variant.omitTopLevelVersion ? {} : { version: 1 }),
    [variant.evidenceArrayKey]: [
      scopedEvidence,
      conflictReferencedEvidence,
      excludedEvidence,
    ],
  });

  const conflictLedger = JSON.stringify({
    artifact_schema_version: "2.0.0",
    artifact_type: "CONFLICT_LEDGER",
    run_id: runId,
    data_cutoff: dataCutoff,
    ...(variant.omitTopLevelVersion ? {} : { version: 1 }),
    conflicts: [
      {
        [variant.conflictIdKey]: "C-001",
        metric_claim: "Synthetic moat conflict",
        source_a: "E-001",
        source_b: "E-002",
        value_a: "A",
        value_b: "B",
        conflict_type: "SYNTHETIC",
        reason: "Different evidence states",
        resolution: "UNRESOLVED",
        materiality: "HIGH",
        affected_outputs: [variant.conflictScope],
      },
      {
        [variant.conflictIdKey]: "C-002",
        metric_claim: "Non-moat conflict",
        source_a: "E-003",
        source_b: "E-003",
        value_a: "A",
        value_b: "B",
        conflict_type: "SYNTHETIC",
        reason: "Outside scope",
        resolution: "UNRESOLVED",
        materiality: "LOW",
        affected_outputs: ["VALUATION_INPUTS"],
      },
    ],
  });

  const evidencePin: Gate18ArtifactPin = {
    artifact_id: "evidence-artifact",
    version: 1,
    sha256: sha256(evidenceLedger),
    repository: "robzer13/real-orotitan",
    commit_sha: "a".repeat(40),
    path:
      `artifacts/orotitan-equity/runs/${runId}/research/EVIDENCE_LEDGER__evidence-artifact__v001.json`,
  };

  const conflictPin: Gate18ArtifactPin = {
    artifact_id: "conflict-artifact",
    version: 1,
    sha256: sha256(conflictLedger),
    repository: "robzer13/real-orotitan",
    commit_sha: "b".repeat(40),
    path:
      `artifacts/orotitan-equity/runs/${runId}/research/CONFLICT_LEDGER__conflict-artifact__v001.json`,
  };

  const company: Gate18PilotCompany = {
    role: "SYNTHETIC",
    display_name: variant.name,
    source_run_id: runId,
    data_cutoff: dataCutoff,
    evidence_ledger: evidencePin,
    conflict_ledger: conflictPin,
  };

  const files = new Map<string, Uint8Array>([
    [evidencePin.path, Buffer.from(evidenceLedger, "utf8")],
    [conflictPin.path, Buffer.from(conflictLedger, "utf8")],
  ]);

  return {
    company,
    readArtifact(pin: Gate18ArtifactPin): Uint8Array {
      const value = files.get(pin.path);
      if (!value) {
        throw new Error("missing synthetic artifact");
      }
      return value;
    },
  };
}

const variants: Variant[] = [
  {
    name: "RATIONAL_STYLE",
    evidenceArrayKey: "evidence_items",
    evidenceIdKey: "evidence_id",
    moduleKey: "used_in",
    valueKey: "value_statement",
    conflictIdKey: "conflict_id",
    conflictScope: "MOAT_INPUTS",
  },
  {
    name: "CONSTELLATION_STYLE",
    evidenceArrayKey: "items",
    evidenceIdKey: "evidence_id",
    moduleKey: "blocks",
    valueKey: "value",
    conflictIdKey: "conflict_id",
    conflictScope: "MOAT_INPUTS",
  },
  {
    name: "STM_STYLE",
    evidenceArrayKey: "items",
    evidenceIdKey: "id",
    moduleKey: "blocks",
    valueKey: "value",
    conflictIdKey: "conflict_id",
    conflictScope: "MOAT_INPUTS",
  },
  {
    name: "BROOKFIELD_STYLE",
    evidenceArrayKey: "evidence",
    evidenceIdKey: "evidence_id",
    moduleKey: "used_in",
    valueKey: "value_statement",
    conflictIdKey: "conflict_id",
    conflictScope: "MOAT_INPUTS",
  },
  {
    name: "ADYEN_STYLE",
    evidenceArrayKey: "items",
    evidenceIdKey: "id",
    moduleKey: "blocks",
    valueKey: "value",
    conflictIdKey: "id",
    conflictScope: "MOAT",
    omitTopLevelVersion: true,
  },
];

for (const variant of variants) {
  test(`Gate 18 v0.2 normalizes ${variant.name} without widening the moat scope`, () => {
    const fixture = buildVariant(variant);
    const result = buildVerifiedGate18V02MoatPacket(
      fixture.company,
      fixture.readArtifact,
    );

    assert.equal(result.packet.scope, "MOAT_INPUTS");
    assert.equal(result.sourceEvidenceCount, 3);
    assert.equal(result.sourceConflictCount, 2);
    assert.deepEqual(
      result.packet.evidence_items.map(
        (item) => item.evidence_id,
      ),
      ["E-001", "E-002"],
    );
    assert.deepEqual(
      result.packet.conflicts.map(
        (item) => item.conflict_id,
      ),
      ["C-001"],
    );
    assert.equal(
      result.packet.evidence_items.some(
        (item) => item.evidence_id === "E-003",
      ),
      false,
    );
  });
}

test("Gate 18 v0.2 output contract is deliberately bounded and compact", () => {
  const maximal = buildGate18PhaseBV02MaximalOutputFixture(
    "x".repeat(64),
    "2026-09-19",
  );

  assert.doesNotThrow(() =>
    gate18PhaseBV02OutputSchema.parse(maximal),
  );

  const serialized = JSON.stringify(maximal);
  assert.ok(
    serialized.length <= 3600,
    `maximal v0.2 output fixture unexpectedly large: ${serialized.length}`,
  );
});

test("Gate 18 v0.2 remains ASSIST-only and versioned", () => {
  assert.equal(
    GATE18_PHASE_B_V02_SCOPE,
    "MOAT_INPUTS",
  );
  assert.equal(
    GATE18_PHASE_B_V02_MODULE_ID,
    "MOAT_EVIDENCE_AUDIT_ASSISTED_V0_2",
  );
  assert.equal(
    GATE18_PHASE_B_V02_GENERATION_SCHEMA_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_2",
  );
  assert.match(
    GATE18_PHASE_B_V02_SYSTEM_PROMPT,
    /ASSIST-only evidence-audit task/,
  );
  assert.match(
    GATE18_PHASE_B_V02_SYSTEM_PROMPT,
    /Do not render a final moat mechanism judgment/,
  );
});

test("Gate 18 runner advances to v0.3 while preserving v0.2 packet normalization and paid-call guards", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-b.ts",
    "utf8",
  );

  assert.match(
    source,
    /model-calibration-pilot-v03/,
  );
  assert.match(
    source,
    /GATE18_PHASE_B_V03_SCOPE/,
  );
  assert.match(
    source,
    /--allow-multi-model/,
  );
  assert.match(
    source,
    /VNEXT_GATE18_PHASE_B_EXPLICIT_MODEL_REQUIRED/,
  );
  assert.match(
    source,
    /VNEXT_GATE18_PHASE_B_PRECALL_SPEND_CAP_WOULD_BE_EXCEEDED/,
  );
});


test("Gate 18 v0.2 fails closed when pinned filename version disagrees with manifest version", () => {
  const fixture = buildVariant({
    name: "VERSION_MISMATCH",
    evidenceArrayKey: "items",
    evidenceIdKey: "id",
    moduleKey: "blocks",
    valueKey: "value",
    conflictIdKey: "id",
    conflictScope: "MOAT",
    omitTopLevelVersion: true,
  });

  fixture.company.evidence_ledger.version = 2;

  assert.throws(
    () =>
      buildVerifiedGate18V02MoatPacket(
        fixture.company,
        fixture.readArtifact,
      ),
    /VNEXT_GATE18_V02_PRIVATE_PATH_VERSION_MISMATCH/,
  );
});
