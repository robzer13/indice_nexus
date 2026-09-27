import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Gemma 3 Constellation counterevidence forensic is read-only and non-inferential", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GEMMA3_COUNTEREVIDENCE_FORENSIC_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-gemma3-counterevidence-link-forensic.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_LOCAL_READ_ONLY_NO_INFERENCE");
  assert.equal(prep.diagnostic_normalization.in_memory_only, true);
  assert.equal(prep.diagnostic_normalization.source_artifact_mutation, false);
  assert.equal(prep.diagnostic_normalization.retroactive_pass_allowed, false);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);

  assert.match(source, /VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS/);
  assert.match(source, /normalizeGate18V11Presentation/);
  assert.match(source, /v11\.normalizedPaths\.length !== 13/);
  assert.match(source, /finding\.counterevidence_link = null/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.match(source, /rawNarrativeTextIncludedInConsole:false/);
  assert.match(source, /retroactivePassAllowed:false/);
  assert.match(source, /inferenceExecuted:false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
