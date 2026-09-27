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

  assert.equal(prep.status, "PREPARED_NO_INFERENCE_NOT_AUTHORIZED");
  assert.equal(prep.constraints.inference_authorized, false);
  assert.equal(prep.constraints.authorized_run_count, 0);
  assert.equal(prep.evaluation_contract.raw_output_must_be_preserved, true);
  assert.equal(
    prep.evaluation_contract.validation_contract,
    "GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1",
  );
});
