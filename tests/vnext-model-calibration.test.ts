import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";

import {
  GATE18_MODEL_CANDIDATES,
  assertGate18SmokeReceipt,
  type Gate18SmokeReceipt,
} from "../runtime/vnext/model-calibration";

test("Gate 18 pins exactly the four initial physical candidates", () => {
  assert.deepEqual(
    GATE18_MODEL_CANDIDATES.map((candidate) => ({
      label: candidate.label,
      modelId: candidate.modelId,
      intendedTier: candidate.intendedTier,
      reasoning: candidate.reasoning,
    })),
    [
      {
        label: "LUNA",
        modelId: "openai/gpt-5.6-luna",
        intendedTier: "T1_STRUCTURED_EXTRACTION",
        reasoning: "minimal",
      },
      {
        label: "TERRA",
        modelId: "openai/gpt-5.6-terra",
        intendedTier: "T2_STANDARD_ANALYSIS",
        reasoning: "medium",
      },
      {
        label: "SOL",
        modelId: "openai/gpt-5.6-sol",
        intendedTier: "T3_PREMIUM_REASONING",
        reasoning: "high",
      },
      {
        label: "ASTRA",
        modelId: "openai/gpt-6-astra",
        intendedTier: "T4_FRONTIER_ESCALATION",
        reasoning: "high",
      },
    ],
  );
});

test("Gate 18 pilot is exactly five diverse pinned source runs", () => {
  assert.equal(pilotJson.format, "OROTITAN_GATE18_PILOT_V0.1");
  assert.equal(pilotJson.sample_size, 5);
  assert.equal(pilotJson.companies.length, 5);

  assert.deepEqual(
    pilotJson.companies.map((company) => company.role),
    [
      "SIMPLE_CLEAN_COMPOUNDER",
      "SERIAL_ACQUIRER",
      "CYCLICAL_SEGMENTED",
      "ACCOUNTING_HEAVY",
      "DIFFICULT_MOAT_CONFLICTING_EVIDENCE",
    ],
  );

  assert.equal(
    new Set(
      pilotJson.companies.map((company) => company.source_run_id),
    ).size,
    5,
  );

  for (const company of pilotJson.companies) {
    assert.match(company.data_cutoff, /^\d{4}-\d{2}-\d{2}$/);
    assert.match(company.evidence_ledger.sha256, /^[a-f0-9]{64}$/);
    assert.match(company.conflict_ledger.sha256, /^[a-f0-9]{64}$/);
    assert.equal(
      company.evidence_ledger.repository,
      "robzer13/real-orotitan",
    );
    assert.equal(
      company.conflict_ledger.repository,
      "robzer13/real-orotitan",
    );
  }
});

test("valid Gate 18 smoke receipt is accepted", () => {
  const receipt: Gate18SmokeReceipt = {
    label: "SOL",
    modelId: "openai/gpt-5.6-sol",
    intendedTier: "T3_PREMIUM_REASONING",
    requestedReasoning: "high",
    schemaValid: true,
    latencyMs: 1234,
    inputTokens: 100,
    outputTokens: 20,
    reasoningTokens: 30,
    totalTokens: 150,
    finishReason: "stop",
    providerMetadata: null,
  };

  assert.doesNotThrow(() => assertGate18SmokeReceipt(receipt));
});

test("Gate 18 smoke receipt rejects unknown physical substitution", () => {
  const receipt = {
    label: "SOL",
    modelId: "openai/gpt-5.4",
    intendedTier: "T3_PREMIUM_REASONING",
    requestedReasoning: "high",
    schemaValid: true,
    latencyMs: 1234,
    inputTokens: 100,
    outputTokens: 20,
    reasoningTokens: 30,
    totalTokens: 150,
    finishReason: "stop",
    providerMetadata: null,
  } as Gate18SmokeReceipt;

  assert.throws(
    () => assertGate18SmokeReceipt(receipt),
    /VNEXT_GATE18_UNKNOWN_MODEL_CANDIDATE/,
  );
});

test("Gate 18 smoke route is preview-only and cannot call models in production", () => {
  const source = readFileSync(
    "app/api/vnext/calibration/gate18-smoke/route.ts",
    "utf8",
  );

  assert.match(source, /process\.env\.VERCEL_ENV === "production"/);
  assert.match(source, /GATE18_SMOKE_PREVIEW_ONLY/);
  assert.match(source, /status: 403/);
  assert.match(source, /modelWinnerSelected: false/);
});

test("Gate 18 candidate contract forbids premature routing freeze", () => {
  const contract = readFileSync(
    "contracts/orotitan-equity/vnext/OROTITAN_VNEXT_MODEL_CALIBRATION_V0.1.md",
    "utf8",
  );

  assert.match(contract, /NO THRESHOLD FREEZE/);
  assert.match(contract, /NO MODEL PROMOTION/);
  assert.match(contract, /NO ROUTING WINNER/);
  assert.match(contract, /GATE 18[\s\S]*IN PROGRESS[\s\S]*NOT FROZEN/);
});
