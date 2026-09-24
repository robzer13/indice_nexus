import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  GATE18_PHASE_B_PROTOCOL_VERSION,
  GATE18_PHASE_B_V04_MAX_OUTPUT_TOKENS,
  GATE18_PHASE_B_V03_GENERATION_SCHEMA_ID,
  GATE18_PHASE_B_V03_GENERATION_SCHEMA_VERSION,
  GATE18_PHASE_B_V03_PROMPT_TEMPLATE_ID,
  GATE18_PHASE_B_V03_PROMPT_TEMPLATE_VERSION,
} from "../runtime/vnext/model-calibration-pilot-v04";

test("Gate 18 v0.4 changes only the common transport envelope", () => {
  assert.equal(GATE18_PHASE_B_PROTOCOL_VERSION, "0.4");
  assert.equal(GATE18_PHASE_B_V04_MAX_OUTPUT_TOKENS, 4096);

  assert.equal(
    GATE18_PHASE_B_V03_PROMPT_TEMPLATE_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_V0_3",
  );
  assert.equal(
    GATE18_PHASE_B_V03_PROMPT_TEMPLATE_VERSION,
    "0.3",
  );
  assert.equal(
    GATE18_PHASE_B_V03_GENERATION_SCHEMA_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_3",
  );
  assert.equal(
    GATE18_PHASE_B_V03_GENERATION_SCHEMA_VERSION,
    "0.3",
  );
});

test("Gate 18 v0.4 runner applies one common max output contract", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-b.ts",
    "utf8",
  );

  assert.match(source, /model-calibration-pilot-v07/);
  assert.match(
    source,
    /maxOutputTokens:\s*GATE18_PHASE_B_V07_MAX_OUTPUT_TOKENS/,
  );
  assert.match(
    source,
    /GATE18_PHASE_B_V07_MAX_OUTPUT_TOKENS \*\s*modelPricing\.output/,
  );

  assert.doesNotMatch(
    source,
    /candidate\.label\s*===\s*["']SOL["'][\s\S]{0,200}maxOutputTokens/,
  );
  assert.doesNotMatch(
    source,
    /candidate\.label\s*===\s*["']ASTRA["'][\s\S]{0,200}maxOutputTokens/,
  );
});

test("Gate 18 v0.4 preserves explicit single-model paid-call guards", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-b.ts",
    "utf8",
  );

  assert.match(
    source,
    /VNEXT_GATE18_PHASE_B_EXPLICIT_MODEL_REQUIRED/,
  );
  assert.match(
    source,
    /VNEXT_GATE18_PHASE_B_PRECALL_SPEND_CAP_WOULD_BE_EXCEEDED/,
  );
  assert.match(source, /--allow-multi-model/);
});
