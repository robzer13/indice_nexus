import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";

import rationalAttempt from "../calibration/vnext/OROTITAN_GATE18_RATIONAL_ATTEMPT_001.json";
import smokeAttempt from "../calibration/vnext/OROTITAN_GATE18_SMOKE_ATTEMPT_002.json";
import {
  GATE18_PHASE_B_GENERATION_SCHEMA_SPEC,
  GATE18_PHASE_B_SYSTEM_PROMPT,
  assertGate18PhaseBSemantics,
  buildVerifiedGate18EvidencePacket,
  gate18PhaseBGenerationSchemaSha256,
  gate18PhaseBPromptTemplateSha256,
  type Gate18ArtifactPin,
  type Gate18PhaseBOutput,
  type Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function fixture() {
  const evidence = JSON.stringify({
    artifact_schema_version: "2.0.0",
    artifact_type: "EVIDENCE_LEDGER",
    run_id: "run-1",
    data_cutoff: "2026-09-19",
    version: 1,
    baseline_snapshot_id: "must-not-enter-packet",
    evidence_items: [
      {
        evidence_id: "E-002",
        claim_id: "CL-002",
        claim_metric: "Second",
        value_statement: "Second supported statement",
        period: "FY2025",
        as_of_date: "2025-12-31",
        source: "S-002",
        root_source_id: "ROOT-002",
        independence_group: "GROUP-2",
        source_class: "S1",
        claim_fit: "HIGH",
        source_date: "2026-03-01",
        data_cutoff: "2026-09-19",
        epistemic_type: "REPORTED",
        freshness_state: "CURRENT_FOR_CUTOFF",
        limitations: "",
        conflict_status: "NONE",
        used_in: ["MOAT_INPUTS"],
        investment_score: 99,
      },
      {
        evidence_id: "E-001",
        claim_id: "CL-001",
        claim_metric: "First",
        value_statement: "First supported statement",
        period: "FY2025",
        as_of_date: "2025-12-31",
        source: "S-001",
        root_source_id: "ROOT-001",
        independence_group: "GROUP-1",
        source_class: "S1",
        claim_fit: "HIGH",
        source_date: "2026-03-01",
        epistemic_type: "REPORTED",
        freshness_state: "CURRENT_FOR_CUTOFF",
        limitations: "",
        conflict_status: "C-001",
        used_in: ["MOAT_INPUTS"],
      },
    ],
  });

  const conflicts = JSON.stringify({
    artifact_schema_version: "2.0.0",
    artifact_type: "CONFLICT_LEDGER",
    run_id: "run-1",
    data_cutoff: "2026-09-19",
    version: 1,
    baseline_snapshot_id: "must-not-enter-packet",
    conflicts: [
      {
        conflict_id: "C-001",
        metric_claim: "Denominator",
        source_a: "E-001",
        value_a: "A",
        source_b: "E-002",
        value_b: "B",
        conflict_type: "DEFINITION_DENOMINATOR",
        reason: "Definitions differ",
        resolution: "UNRESOLVED",
        resolution_note: "Carry forward",
        materiality: "HIGH",
        affected_outputs: ["MOAT_INPUTS"],
        next_action: "must-not-enter-packet",
      },
    ],
  });

  const evidencePin: Gate18ArtifactPin = {
    artifact_id: "artifact-e",
    version: 1,
    sha256: sha256(evidence),
    repository: "robzer13/real-orotitan",
    commit_sha: "a".repeat(40),
    path:
      "artifacts/orotitan-equity/runs/run-1/research/EVIDENCE_LEDGER__artifact-e__v001.json",
  };

  const conflictPin: Gate18ArtifactPin = {
    artifact_id: "artifact-c",
    version: 1,
    sha256: sha256(conflicts),
    repository: "robzer13/real-orotitan",
    commit_sha: "b".repeat(40),
    path:
      "artifacts/orotitan-equity/runs/run-1/research/CONFLICT_LEDGER__artifact-c__v001.json",
  };

  const company: Gate18PilotCompany = {
    role: "SIMPLE_CLEAN_COMPOUNDER",
    display_name: "Synthetic Co",
    source_run_id: "run-1",
    data_cutoff: "2026-09-19",
    evidence_ledger: evidencePin,
    conflict_ledger: conflictPin,
  };

  const byPath = new Map([
    [evidencePin.path, Buffer.from(evidence, "utf8")],
    [conflictPin.path, Buffer.from(conflicts, "utf8")],
  ]);

  return {
    evidence,
    conflicts,
    company,
    readArtifact(pin: Gate18ArtifactPin): Uint8Array {
      const bytes = byPath.get(pin.path);
      if (!bytes) {
        throw new Error("missing fixture");
      }
      return bytes;
    },
  };
}

test("Gate 18 Phase B packet verifies pinned bytes and projects only admissible Research evidence", () => {
  const f = fixture();
  const verified = buildVerifiedGate18EvidencePacket(
    f.company,
    f.readArtifact,
  );

  assert.equal(verified.packet.evidence_items.length, 2);
  assert.equal(verified.packet.conflicts.length, 1);
  assert.equal(
    verified.packet.evidence_items[0].evidence_id,
    "E-001",
  );
  assert.equal(
    verified.packet.evidence_items[1].evidence_id,
    "E-002",
  );

  const serialized = JSON.stringify(verified.packet);

  assert.equal(serialized.includes("baseline_snapshot_id"), false);
  assert.equal(serialized.includes("investment_score"), false);
  assert.equal(serialized.includes("next_action"), false);
  assert.equal(
    verified.packet.source_integrity.evidence_ledger_sha256,
    sha256(f.evidence),
  );
  assert.equal(
    verified.packet.source_integrity.conflict_ledger_sha256,
    sha256(f.conflicts),
  );
  assert.match(verified.packetSha256, /^[a-f0-9]{64}$/);
});

test("Gate 18 Phase B packet fails closed on source-byte drift", () => {
  const f = fixture();

  assert.throws(
    () =>
      buildVerifiedGate18EvidencePacket(
        f.company,
        (pin) => {
          const original = f.readArtifact(pin);
          return pin === f.company.evidence_ledger
            ? Buffer.concat([
                Buffer.from(original),
                Buffer.from("\n"),
              ])
            : original;
        },
      ),
    /VNEXT_GATE18_PRIVATE_ARTIFACT_SHA256_MISMATCH/,
  );
});

test("Gate 18 Phase B semantic validator rejects invented evidence references", () => {
  const f = fixture();
  const verified = buildVerifiedGate18EvidencePacket(
    f.company,
    f.readArtifact,
  );

  const output: Gate18PhaseBOutput = {
    case_id: "run-1",
    data_cutoff: "2026-09-19",
    findings: [
      {
        finding_id: "F-001",
        claim: "Synthetic claim",
        support_state: "SUPPORTED",
        evidence_ids: ["E-999"],
        conflict_ids: [],
        causal_link: "Synthetic causal link",
        counterevidence_ids: [],
      },
    ],
    conflicts: [],
    weak_link_candidates: [],
    unresolved_points: [],
  };

  assert.throws(
    () =>
      assertGate18PhaseBSemantics(
        verified.packet,
        output,
      ),
    /VNEXT_GATE18_PHASE_B_UNKNOWN_EVIDENCE_REF/,
  );
});

test("Gate 18 Phase B calibration prompt preserves the human-judgment boundary", () => {
  assert.match(
    GATE18_PHASE_B_SYSTEM_PROMPT,
    /ASSIST-only calibration task/,
  );
  assert.match(
    GATE18_PHASE_B_SYSTEM_PROMPT,
    /Do not render a final moat/,
  );
  assert.doesNotMatch(
    JSON.stringify(
      GATE18_PHASE_B_GENERATION_SCHEMA_SPEC,
    ),
    /investment_score|next_action|valuation_conclusion/i,
  );

  assert.match(
    gate18PhaseBPromptTemplateSha256(),
    /^[a-f0-9]{64}$/,
  );
  assert.match(
    gate18PhaseBGenerationSchemaSha256(),
    /^[a-f0-9]{64}$/,
  );
});

test("Gate 18 Phase B runner is local, explicit-spend and private-output only", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-b.ts",
    "utf8",
  );
  const gitignore = readFileSync(".gitignore", "utf8");

  assert.match(source, /--execute/);
  assert.match(source, /--max-case-spend-usd/);
  assert.match(source, /--model/);
  assert.match(source, /onStepFinish/);
  assert.match(source, /NoObjectGeneratedError/);
  assert.match(source, /generatedTextSha256/);
  assert.match(source, /gatewayCostUsd/);
  assert.match(
    source,
    /VNEXT_GATE18_PHASE_B_EXECUTION_SPEND_CAP_REQUIRED/,
  );
  assert.match(
    source,
    /VERCEL_OIDC_TOKEN/,
  );
  assert.match(
    source,
    /productionMutation: false/,
  );
  assert.match(
    source,
    /publicationAuthority: false/,
  );
  assert.match(
    gitignore,
    /calibration\/vnext\/private-runs\//,
  );

  assert.doesNotMatch(
    source,
    /SUPABASE_SERVICE_ROLE_KEY|OROTITAN_PUBLICATION_ENABLED/,
  );
});

test("Gate 18 successful smoke evidence is exactly 4/4 with no model winner", () => {
  assert.equal(smokeAttempt.gate, 18);
  assert.equal(smokeAttempt.status, "PASS");
  assert.equal(
    smokeAttempt.gate_effect.phase_a_physical_model_smoke,
    "PASS",
  );
  assert.equal(smokeAttempt.receipts.length, 4);
  assert.equal(
    smokeAttempt.receipts.every(
      (receipt) =>
        receipt.schema_valid === true &&
        receipt.finish_reason === "stop" &&
        receipt.provider_attempt_count === 1 &&
        receipt.fallback_used === false,
    ),
    true,
  );
  assert.equal(
    smokeAttempt.aggregate.gateway_cost_usd,
    0.007985,
  );
  assert.equal(smokeAttempt.model_winner_selected, false);
  assert.equal(
    smokeAttempt.gate_effect.gate_18,
    "IN_PROGRESS_NOT_FROZEN",
  );
});


test("Gate 18 RATIONAL attempt 001 is recorded as engineering failure without quality judgment", () => {
  assert.equal(rationalAttempt.gate, 18);
  assert.equal(
    rationalAttempt.phase,
    "B_COMPANY_CALIBRATION",
  );
  assert.equal(rationalAttempt.status, "PARTIAL_OR_FAILED");
  assert.equal(rationalAttempt.models_completed, 0);
  assert.equal(rationalAttempt.quality_observations, 0);
  assert.equal(
    rationalAttempt.interpretation.analytical_quality_failure,
    false,
  );
  assert.equal(
    rationalAttempt.interpretation.engineering_output_contract_failure,
    true,
  );
  assert.equal(
    rationalAttempt.interpretation.root_cause_proven,
    false,
  );
  assert.equal(
    rationalAttempt.interpretation.output_token_limit_hypothesis,
    "UNCONFIRMED",
  );
  assert.equal(
    rationalAttempt.cost_evidence.runner_observed_gateway_cost_usd,
    0.5775745,
  );
  assert.equal(
    rationalAttempt.cost_evidence.vercel_budget_display_after_usd,
    0.68,
  );
  assert.equal(
    rationalAttempt.authority.model_winner_selected,
    false,
  );
  assert.equal(
    rationalAttempt.gate_effect.gate_18,
    "IN_PROGRESS_NOT_FROZEN",
  );
});
