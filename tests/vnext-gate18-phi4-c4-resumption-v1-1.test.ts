import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 C4 resumption selects one bounded Constellation discriminator under v1.1", () => {
  const decision = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHI4_C4_RESUMPTION_DECISION_001.json",
      "utf8",
    ),
  );
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_PHI4_MINI_V1_1_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(
    decision.decision,
    "RESUME_PHI4_C4_UNDER_V1_1_WITH_ONE_CONSTELLATION_CELL",
  );
  assert.equal(decision.selection.company, "Constellation Software");
  assert.equal(decision.selection.archetype, "SERIAL_ACQUIRER");
  assert.equal(decision.selection.selection_is_model_ranking, false);
  assert.equal(decision.selection.selection_is_production_routing, false);

  assert.equal(
    decision.frozen_cell.validation_contract,
    "GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1",
  );
  assert.equal(decision.frozen_cell.context_tokens, 16384);
  assert.equal(decision.frozen_cell.max_output_tokens, 1024);
  assert.equal(decision.frozen_cell.temperature, 0);
  assert.equal(decision.frozen_cell.client_timeout_ms, 480000);

  assert.equal(decision.execution_boundary.inference_authorized, false);
  assert.equal(decision.execution_boundary.authorized_run_count, 0);
  assert.equal(decision.execution_boundary.automatic_retry_authorized, false);

  assert.equal(
    prep.status,
    "SINGLE_RUN_CONSUMED_ENGINEERING_PASS_HUMAN_ADJUDICATION_PENDING",
  );
  assert.equal(prep.constraints.inference_authorized, false);
  assert.equal(prep.constraints.authorized_run_count, 0);
  assert.equal(
    prep.execution_result,
    "G18-PHASEC-C4-CONSTELLATION-PHI4MINI-V1_1-RESULT-001",
  );
  assert.equal(prep.evaluation_contract.raw_output_must_be_preserved, true);
  assert.equal(
    prep.evaluation_contract.validation_contract,
    "GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1",
  );
});


test("Phi-4 Constellation v1.1 single-run authority is consumed after the observed engineering pass", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_PHI4_MINI_V1_1_CONTEXT16384_OUTPUT1024_TIMEOUT480_LOOPBACK_AUTH_001.json",
      "utf8",
    ),
  );
  const runner = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-phi4-mini-v1-1-context16384-output1024-timeout480-loopback-guarded.ts",
    "utf8",
  );

  assert.equal(auth.status, "CONSUMED_SINGLE_RUN_COMPLETE");
  assert.equal(auth.authorization_source.type, "EXPLICIT_USER_AUTHORIZATION_IN_CHAT");
  assert.equal(auth.authorization_source.user_message, "autorisé");
  assert.equal(auth.c4_inference.authorized, false);
  assert.equal(auth.constraints.authorized_run_count, 0);
  assert.equal(auth.constraints.automatic_retry_authorized, false);
  assert.equal(
    auth.execution_result,
    "G18-PHASEC-C4-CONSTELLATION-PHI4MINI-V1_1-RESULT-001",
  );
  assert.equal(auth.execution_observed.schema_valid, true);
  assert.equal(auth.execution_observed.raw_presentation_compliant, false);
  assert.equal(auth.execution_observed.normalized_path_count, 3);
  assert.equal(auth.execution_observed.substantive_status, "PASS");

  assert.match(runner, /evaluateGate18V11Validation/);
  assert.match(
    runner,
    /GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1/,
  );
  assert.match(
    runner,
    /AUTHORIZED_SINGLE_LOCAL_INFERENCE/,
  );
  assert.match(
    runner,
    /phi4-mini:3\.8b-q4_K_M/,
  );
  assert.match(
    runner,
    /Constellation Software/,
  );
  assert.match(
    runner,
    /CLIENT_TIMEOUT_MS = 480_000/,
  );
  assert.match(
    runner,
    /MAX_OUTPUT_TOKENS = 1024/,
  );
});
