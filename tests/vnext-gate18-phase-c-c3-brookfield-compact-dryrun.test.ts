import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c3-brookfield-compact-dryrun.ts",
  "utf8",
);

test("Brookfield C3 dry-run cannot infer or download", () => {
  assert.match(source, /DRY_RUN_NO_INFERENCE/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.match(source, /modelDownloadExecuted: false/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\/api\/chat/);
  assert.doesNotMatch(source, /\/api\/pull/);
});

test("Brookfield C3 dry-run pins Qwen3 1.7B exactly", () => {
  assert.match(source, /MODEL_NAME = "qwen3:1\.7b"/);
  assert.match(
    source,
    /8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7/,
  );
  assert.match(source, /MODEL_DIGEST_MISMATCH/);
});

test("Brookfield C3 dry-run keeps only exact regression evidence", () => {
  for (const id of ["E-036", "E-037", "E-039", "E-042"]) {
    assert.match(source, new RegExp(id));
  }
  assert.match(source, /conflicts: \[\]/);
  assert.match(source, /evidence_items: evidenceItems/);
});

test("Brookfield C3 dry-run targets schema-aware 8k context with bounded output", () => {
  assert.match(source, /LOCAL_CONTEXT_TOKENS = 8192/);
  assert.match(source, /LOCAL_MAX_OUTPUT_TOKENS = 768/);
  assert.match(source, /temperature: 0/);
  assert.match(source, /inferenceAuthorized: false/);
});

test("Brookfield C3 dry-run preserves exact historical polarity target", () => {
  assert.match(source, /BROOKFIELD_PEER_ROLE_CORE_001_LOCAL_COMPACT/);
  assert.match(source, /evidenceIds: \["E-036", "E-037", "E-039"\]/);
  assert.match(source, /evidenceIds: \["E-042"\]/);
  assert.match(source, /counterevidenceIds: \[\]/);
});

test("Brookfield C3 dry-run includes schema in the conservative input guard", () => {
  assert.match(source, /schemaAwareInputChars/);
  assert.match(source, /combinedPromptChars \+ outputSchemaJson\.length/);
  assert.match(source, /contextHeadroomTokens/);
});

test("Brookfield C3 dry-run estimates local KV-cache pressure without inference", () => {
  assert.match(source, /"\/api\/show"/);
  assert.match(source, /estimatedKvCacheMiBAtPlannedContext/);
  assert.match(source, /attention\.head_count_kv/);
  assert.match(source, /freeVramMiB/);
});
