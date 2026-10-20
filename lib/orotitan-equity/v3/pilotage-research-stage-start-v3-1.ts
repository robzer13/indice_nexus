/**
 * D11 product handoff: one narrowly scoped, CAS-protected Registry stage start.
 * Only a byte-verified D2->D1 Research FINAL decision may produce this RPC.
 * Never creates a Research FINAL, changes Certification, or publishes a snapshot.
 */
import { createHash } from "node:crypto";
import pins from "../../../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";
import type { PilotageDecision } from "./pilotage-adapter-v3-1";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SHA = /^[a-f0-9]{64}$/;

export class ResearchStageStartError extends Error {
  constructor(readonly code: string) { super(code); }
}

function admit(value: unknown, code: string): asserts value {
  if (value !== true) throw new ResearchStageStartError(code);
}

export function prepareResearchToFundamentalsStart(input: {
  runId: string;
  expectedRunStateVersion: number;
  expectedResearchStageStateVersion: number;
  verifiedDecision: PilotageDecision;
}) {
  const d = input.verifiedDecision;
  admit(UUID.test(input.runId), "RUN_ID_INVALID");
  admit(Number.isSafeInteger(input.expectedRunStateVersion) &&
    input.expectedRunStateVersion > 0, "RUN_CAS_INVALID");
  admit(Number.isSafeInteger(input.expectedResearchStageStateVersion) &&
    input.expectedResearchStageStateVersion > 0, "STAGE_CAS_INVALID");
  admit(d.outcome === "HANDOFF" && d.phase === "FUNDAMENTALS" &&
    d.reasonCode === "RESEARCH_FINAL_ADMITTED_START_REQUIRED" &&
    typeof d.exactPrompt === "string" && d.exactPrompt.length > 0 &&
    d.publicationAuthorized === false && d.mutationPerformed === false,
    "RESEARCH_FINAL_NOT_VERIFIED");
  admit(d.expectedRunStateVersion === input.expectedRunStateVersion, "STALE_RUN_CAS");
  admit(d.expectedStageStateVersion === input.expectedResearchStageStateVersion, "STALE_STAGE_CAS");
  admit(SHA.test(d.stateFingerprintSha256), "VERIFIED_STATE_FINGERPRINT_INVALID");

  const target = pins.contract_pins.deep_dive_stage;
  const requestFingerprint = createHash("sha256").update([
    "OROTITAN_V31_D11_RESEARCH_FINAL_STAGE_START",
    input.runId, String(d.expectedRunStateVersion), String(d.expectedStageStateVersion),
    d.stateFingerprintSha256, target.name, target.version, target.content_sha256,
  ].join("\n"), "utf8").digest("hex");

  return {
    rpc: "start_orotitan_stage" as const,
    args: {
      p_run_id: input.runId,
      p_stage_code: "DEEP_DIVE",
      p_expected_run_state_version: d.expectedRunStateVersion,
      p_stage_contract_name: target.name,
      p_stage_contract_version: target.version,
      p_stage_contract_sha256: target.content_sha256,
      p_idempotency_key: "orotitan:v31:d11:fundamentals:" + requestFingerprint,
      p_request_fingerprint_sha256: requestFingerprint,
      p_actor_type: "PILOTAGE",
    },
  };
}
