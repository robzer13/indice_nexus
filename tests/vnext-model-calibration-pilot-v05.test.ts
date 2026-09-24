import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  GATE18_PHASE_B_MODEL_CANDIDATES,
} from "../runtime/vnext/model-calibration-phase-b-profiles";
import {
  GATE18_PHASE_B_PROTOCOL_VERSION,
  GATE18_PHASE_B_V04_MAX_OUTPUT_TOKENS,
} from "../runtime/vnext/model-calibration-pilot-v05";

test("Gate 18 Phase B v0.5 keeps physical identities and bounds SOL reasoning", () => {
  assert.equal(GATE18_PHASE_B_PROTOCOL_VERSION, "0.5");
  assert.equal(GATE18_PHASE_B_V04_MAX_OUTPUT_TOKENS, 4096);

  assert.deepEqual(
    GATE18_PHASE_B_MODEL_CANDIDATES.map((candidate) => ({
      label: candidate.label,
      modelId: candidate.modelId,
      reasoning: candidate.reasoning,
    })),
    [
      {
        label: "LUNA",
        modelId: "openai/gpt-5.6-luna",
        reasoning: "minimal",
      },
      {
        label: "TERRA",
        modelId: "openai/gpt-5.6-terra",
        reasoning: "medium",
      },
      {
        label: "SOL",
        modelId: "openai/gpt-5.6-sol",
        reasoning: "medium",
      },
      {
        label: "ASTRA",
        modelId: "openai/gpt-6-astra",
        reasoning: "high",
      },
    ],
  );
});

test("Gate 18 Phase B v0.5 runner uses Phase B profiles without per-model output cap exceptions", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-b.ts",
    "utf8",
  );

  assert.match(source, /model-calibration-phase-b-profiles/);
  assert.match(source, /GATE18_PHASE_B_MODEL_CANDIDATES/);
  assert.match(source, /model-calibration-pilot-v09/);
  assert.match(
    source,
    /maxOutputTokens:\s*GATE18_PHASE_B_V09_MAX_OUTPUT_TOKENS/,
  );

  assert.doesNotMatch(
    source,
    /candidate\.label\s*===\s*["']SOL["'][\s\S]{0,200}maxOutputTokens/,
  );
});

test("Gate 18 Phase B v0.5 preserves explicit paid-call cost guards", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-b.ts",
    "utf8",
  );

  assert.match(
    source,
    /VNEXT_GATE18_PHASE_B_EXECUTION_SPEND_CAP_REQUIRED/,
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
