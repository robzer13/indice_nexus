import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("STMicro human adjudication export remains private and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_HUMAN_ADJUDICATION_EXPORT_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    output_policy: {
      public_repo_persistence: boolean;
      generated_content_publication: boolean;
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      auto_repair_authorized: boolean;
      final_moat_conclusion_authorized: boolean;
      model_ranking_authority: boolean;
      routing_authority: boolean;
      production_mutation: boolean;
      publication_authority: boolean;
    };
  };

  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-stmicro-human-adjudication-export.ts",
    "utf8",
  );

  assert.equal(
    prep.status,
    "PREPARED_LOCAL_PRIVATE_NO_INFERENCE",
  );
  assert.equal(prep.output_policy.public_repo_persistence, false);
  assert.equal(
    prep.output_policy.generated_content_publication,
    false,
  );
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.auto_repair_authorized, false);
  assert.equal(
    prep.authority.final_moat_conclusion_authorized,
    false,
  );
  assert.equal(prep.authority.model_ranking_authority, false);
  assert.equal(prep.authority.routing_authority, false);
  assert.equal(prep.authority.production_mutation, false);
  assert.equal(prep.authority.publication_authority, false);

  assert.match(source, /buildVerifiedGate18V10MoatPacket/);
  assert.match(source, /packetSha256/);
  assert.match(source, /promptSha256/);
  assert.match(source, /READY_FOR_HUMAN_ADJUDICATION/);
  assert.match(source, /publicRepoPersistence: false/);
  assert.match(source, /noInferenceExecuted: true/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
