import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Brookfield output1024 forensic selects output1280 and timeout600 without execution authority", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_OUTPUT1024_TRUNCATION_FORENSICS_V1_RESULT_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    structural_progress: {
      all_top_level_sections_present: boolean;
      priority_findings_started: number;
      material_conflicts_started: number;
      weak_link_candidates_started: number;
      unresolved_questions_started: number;
    };
    remediation_selection: {
      selected_max_output_tokens: number;
      proposed_client_timeout_ms: number;
      timeout_headroom_vs_output1280_linear_projection_ms: number;
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      max_output_change_execution_authorized: boolean;
      timeout_change_execution_authorized: boolean;
    };
  };

  assert.equal(
    result.status,
    "PASS_TRUNCATION_STRUCTURE_MEASURED_NO_INFERENCE",
  );
  assert.equal(
    result.structural_progress.all_top_level_sections_present,
    true,
  );
  assert.equal(result.structural_progress.priority_findings_started, 3);
  assert.equal(result.structural_progress.material_conflicts_started, 2);
  assert.equal(result.structural_progress.weak_link_candidates_started, 2);
  assert.equal(result.structural_progress.unresolved_questions_started, 2);
  assert.equal(
    result.remediation_selection.selected_max_output_tokens,
    1280,
  );
  assert.equal(
    result.remediation_selection.proposed_client_timeout_ms,
    600000,
  );
  assert.ok(
    result.remediation_selection
      .timeout_headroom_vs_output1280_linear_projection_ms > 0,
  );
  assert.equal(result.authority.inference_authorized, false);
  assert.equal(result.authority.retry_authorized, false);
  assert.equal(
    result.authority.max_output_change_execution_authorized,
    false,
  );
  assert.equal(
    result.authority.timeout_change_execution_authorized,
    false,
  );
});

test("Brookfield output1280 timeout600 static preflight is exact-packet-bound and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_OUTPUT1280_TIMEOUT600_STATIC_PREFLIGHT_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    proposed_runtime: {
      context_tokens: number;
      max_output_tokens: number;
      temperature: number;
      client_timeout_ms: number;
      transport: string;
    };
    remediation_delta: {
      max_output_tokens_from: number;
      max_output_tokens_to: number;
      client_timeout_ms_from: number;
      client_timeout_ms_to: number;
      model_change: boolean;
      prompt_change: boolean;
      schema_change: boolean;
      packet_change: boolean;
      context_change: boolean;
      temperature_change: boolean;
      transport_change: boolean;
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      max_output_change_execution_authorized: boolean;
      timeout_change_execution_authorized: boolean;
    };
  };

  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-brookfield-output1280-timeout600-static-preflight.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_NO_INFERENCE");
  assert.equal(prep.proposed_runtime.context_tokens, 16384);
  assert.equal(prep.proposed_runtime.max_output_tokens, 1280);
  assert.equal(prep.proposed_runtime.temperature, 0);
  assert.equal(prep.proposed_runtime.client_timeout_ms, 600000);
  assert.equal(
    prep.proposed_runtime.transport,
    "NODE_HTTP_REQUEST_LOOPBACK",
  );
  assert.equal(prep.remediation_delta.max_output_tokens_from, 1024);
  assert.equal(prep.remediation_delta.max_output_tokens_to, 1280);
  assert.equal(prep.remediation_delta.client_timeout_ms_from, 480000);
  assert.equal(prep.remediation_delta.client_timeout_ms_to, 600000);

  for (const key of [
    "model_change",
    "prompt_change",
    "schema_change",
    "packet_change",
    "context_change",
    "temperature_change",
    "transport_change",
  ] as const) {
    assert.equal(prep.remediation_delta[key], false);
  }

  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(
    prep.authority.max_output_change_execution_authorized,
    false,
  );
  assert.equal(
    prep.authority.timeout_change_execution_authorized,
    false,
  );

  assert.match(source, /Brookfield Corporation/);
  assert.match(source, /const MAX_OUTPUT_TOKENS = 1280;/);
  assert.match(source, /const CLIENT_TIMEOUT_MS = 600_000;/);
  assert.match(
    source,
    /eb2d779b95fa5f207e91fd3485ece39b2bd101c2589eaaf8c48481f750ef75b3/,
  );
  assert.match(
    source,
    /516496bf4dc9bcdce47897c056282eb2bda63b014cd63c1b76e3992b1c0cdad2/,
  );
  assert.match(source, /buildVerifiedGate18V10MoatPacket/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.match(source, /ollamaApiCalled: false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
