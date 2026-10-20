import assert from "node:assert/strict";
import test from "node:test";
import { prepareResearchToFundamentalsStart, ResearchStageStartError } from
  "../lib/orotitan-equity/v3/pilotage-research-stage-start-v3-1";
import type { PilotageDecision } from "../lib/orotitan-equity/v3/pilotage-adapter-v3-1";
import pins from "../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";

const RUN = "11111111-1111-4111-8111-111111111111";
function admitted(): PilotageDecision {
  return {
    outcome: "HANDOFF", phase: "FUNDAMENTALS",
    reasonCode: "RESEARCH_FINAL_ADMITTED_START_REQUIRED",
    exactPrompt: "verified immutable FINAL / seven exact Research V2 outputs",
    expectedRunStateVersion: 5, expectedStageStateVersion: 3,
    stateFingerprintSha256: "a".repeat(64),
    publicationAuthorized: false, mutationPerformed: false,
  };
}
function plan(d: PilotageDecision = admitted(), version = 5, stageVersion = 3) {
  return prepareResearchToFundamentalsStart({
    runId: RUN, expectedRunStateVersion: version,
    expectedResearchStageStateVersion: stageVersion, verifiedDecision: d,
  });
}
function rejected(d: PilotageDecision, code: string, version = 5, stageVersion = 3) {
  assert.throws(() => plan(d, version, stageVersion), (e: unknown) =>
    e instanceof ResearchStageStartError && e.code === code);
}
test("D11 admin product: exact pinned Deep Dive start RPC and stable retry fingerprint", () => {
  const prepared = plan();
  assert.equal(prepared.rpc, "start_orotitan_stage");
  assert.equal(prepared.args.p_run_id, RUN);
  assert.equal(prepared.args.p_stage_code, "DEEP_DIVE");
  assert.equal(prepared.args.p_expected_run_state_version, 5);
  assert.equal(prepared.args.p_actor_type, "PILOTAGE");
  assert.equal(prepared.args.p_stage_contract_name, pins.contract_pins.deep_dive_stage.name);
  assert.equal(prepared.args.p_stage_contract_version, pins.contract_pins.deep_dive_stage.version);
  assert.equal(prepared.args.p_stage_contract_sha256, pins.contract_pins.deep_dive_stage.content_sha256);
  assert.equal(prepared.args.p_request_fingerprint_sha256.length, 64);
  assert.deepEqual(prepared, plan());
});
test("D11 admin product: CHECKPOINT and any nonadmitted decision never produce a stage RPC", () => {
  for (const changed of [
    { outcome: "CONTINUE" as const },
    { outcome: "BLOCKED" as const },
    { phase: "RESEARCH" as const },
    { phase: "VALUATION" as const },
    { reasonCode: "RESEARCH_ADMITTED" },
    { exactPrompt: null },
    { publicationAuthorized: true as false },
    { mutationPerformed: true as false },
  ]) {
    rejected({ ...admitted(), ...changed }, "RESEARCH_FINAL_NOT_VERIFIED");
  }
});
test("D11 admin product: stale versions and invalid fingerprint fail closed", () => {
  rejected(admitted(), "STALE_RUN_CAS", 4);
  rejected(admitted(), "STALE_STAGE_CAS", 5, 2);
  rejected({ ...admitted(), stateFingerprintSha256: "x" }, "VERIFIED_STATE_FINGERPRINT_INVALID");
  assert.throws(() => prepareResearchToFundamentalsStart({
    runId: "not-uuid", expectedRunStateVersion: 5,
    expectedResearchStageStateVersion: 3, verifiedDecision: admitted(),
  }), (e: unknown) => e instanceof ResearchStageStartError && e.code === "RUN_ID_INVALID");
});
