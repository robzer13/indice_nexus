import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import acceptanceJson from "../calibration/vnext/OROTITAN_VNEXT_GATE17_ACCEPTANCE_V1.json";
import corpusJson from "../calibration/vnext/AUDIT_CORPUS_VNEXT_V1.json";

import {
  AUDIT_CORPUS_VNEXT_V1_ID,
  AUDIT_CORPUS_VNEXT_V1_SIZE,
  assertNoProductionMutation,
  assertValidAuditCorpus,
  type AuditCorpusManifest,
  type ProductionFingerprint,
} from "../runtime/vnext/shadow-runner";

const corpus = corpusJson as unknown as AuditCorpusManifest;

test("Gate 17 freeze pins implementation merge and PASS state", () => {
  const freeze = readFileSync(
    "contracts/orotitan-equity/vnext/OROTITAN_VNEXT_GATE17_SHADOW_RUNNER_FREEZE_V1.0.md",
    "utf8",
  );

  assert.match(
    freeze,
    /IMPLEMENTATION_MERGE_SHA[\s\S]*be916177ca8b921315d8132bc189774888b3f555/,
  );
  assert.match(freeze, /GATE = 17/);
  assert.match(freeze, /RESULT = PASS \/ FROZEN/);
  assert.match(freeze, /NEXT = GATE 18/);
});

test("Gate 17 frozen corpus remains exactly 33 pinned unique members", () => {
  assert.doesNotThrow(() => assertValidAuditCorpus(corpus));
  assert.equal(corpus.format, AUDIT_CORPUS_VNEXT_V1_ID);
  assert.equal(
    corpus.expected_member_count,
    AUDIT_CORPUS_VNEXT_V1_SIZE,
  );
  assert.equal(corpus.members.length, 33);
});

test("Gate 17 acceptance record proves 33/33 live source preflight", () => {
  assert.equal(acceptanceJson.gate, 17);
  assert.equal(acceptanceJson.result, "PASS");

  assert.deepEqual(acceptanceJson.corpus, {
    id: "AUDIT_CORPUS_VNEXT_V1",
    expected_members: 33,
    pinned_members: 33,
    live_source_runs_found: 33,
    issuer_matches: 33,
    data_cutoff_matches: 33,
    research_complete: 33,
    deep_dive_complete: 33,
    integration_complete: 33,
    runs_with_artifacts: 33,
    runs_with_research_artifacts: 33,
    runs_with_deep_dive_artifacts: 33,
  });
});

test("Gate 17 acceptance record distinguishes runner assurance from Gate 18 model calibration", () => {
  assert.equal(
    acceptanceJson.runner_assurance.exact_corpus_attempted,
    33,
  );
  assert.equal(
    acceptanceJson.runner_assurance.exact_corpus_completed,
    33,
  );
  assert.equal(
    acceptanceJson.runner_assurance.exact_corpus_failed,
    0,
  );
  assert.equal(
    acceptanceJson.runner_assurance.deterministic_executor,
    true,
  );
  assert.equal(
    acceptanceJson.runner_assurance.synthetic_comparison_layer,
    true,
  );
  assert.equal(
    acceptanceJson.runner_assurance.physical_model_calibration_executed,
    false,
  );
  assert.equal(
    acceptanceJson.runner_assurance.physical_model_calibration_gate,
    18,
  );
  assert.equal(
    acceptanceJson.invariants.real_model_comparison_claimed,
    false,
  );
});

test("Gate 17 production before/after fingerprints are identical", () => {
  const before =
    acceptanceJson.production_before as ProductionFingerprint;
  const after =
    acceptanceJson.production_after as ProductionFingerprint;

  assert.deepEqual(before, after);
  assert.doesNotThrow(() =>
    assertNoProductionMutation(before, after),
  );

  assert.deepEqual(before, corpus.production_before_fingerprint);
});

test("Gate 17 leaves shadow analytical and registry rows empty", () => {
  assert.equal(
    acceptanceJson.shadow_after.project_ref,
    "awgsurdyvsyolcgpnygh",
  );
  assert.equal(acceptanceJson.shadow_after.research_dossiers, 0);
  assert.equal(acceptanceJson.shadow_after.research_snapshots, 0);
  assert.equal(acceptanceJson.shadow_after.orotitan_runs, 0);
  assert.equal(acceptanceJson.shadow_after.orotitan_artifacts, 0);
});

test("Gate 17 frozen runner remains provider-neutral and mutation-free", () => {
  const source = readFileSync(
    "runtime/vnext/shadow-runner.ts",
    "utf8",
  );

  const forbidden = [
    /createClient\s*\(/,
    /supabase/i,
    /process\.env/,
    /fetch\s*\(/,
    /\.(insert|update|upsert|delete)\s*\(/,
    /cugpgtzygqqlxetyeven/,
    /openai\.azure\.com/i,
    /AzureProvider/,
    /OpenAIProvider/,
    /AnthropicProvider/,
    /GeminiProvider/,
    /GO PUBLISH/,
    /current_snapshot_id/,
  ];

  for (const pattern of forbidden) {
    assert.equal(
      pattern.test(source),
      false,
      `unexpected Gate 17 frozen runtime dependency: ${pattern}`,
    );
  }
});
