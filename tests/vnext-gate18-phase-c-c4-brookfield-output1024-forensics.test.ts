import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Brookfield output1024 failure is classified as output-budget truncation with no retry authority", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_OUTPUT1024_RESULT_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    execution: {
      done_reason: string;
      eval_count: number;
      runtime_error: null;
      schema_valid: boolean;
      semantic_valid: boolean;
    };
    diagnosis: {
      classification: string;
      output_budget_fully_consumed: boolean;
      output_budget_inadequacy_proven: boolean;
      transport_failure: boolean;
      explicit_480s_timeout_reached: boolean;
    };
    authority: {
      retry_authorized: boolean;
      max_output_change_authorized: boolean;
      timeout_change_authorized: boolean;
    };
  };

  assert.equal(
    result.status,
    "FAIL_OUTPUT_BUDGET_EXHAUSTED_TRUNCATED_JSON",
  );
  assert.equal(result.execution.done_reason, "length");
  assert.equal(result.execution.eval_count, 1024);
  assert.equal(result.execution.runtime_error, null);
  assert.equal(result.execution.schema_valid, false);
  assert.equal(result.execution.semantic_valid, false);
  assert.equal(
    result.diagnosis.classification,
    "OUTPUT_BUDGET_EXHAUSTED_TRUNCATED_JSON_NO_SEMANTIC_RESULT",
  );
  assert.equal(result.diagnosis.output_budget_fully_consumed, true);
  assert.equal(result.diagnosis.output_budget_inadequacy_proven, true);
  assert.equal(result.diagnosis.transport_failure, false);
  assert.equal(
    result.diagnosis.explicit_480s_timeout_reached,
    false,
  );
  assert.equal(result.authority.retry_authorized, false);
  assert.equal(result.authority.max_output_change_authorized, false);
  assert.equal(result.authority.timeout_change_authorized, false);
});

test("Brookfield truncation forensics is local, read-only, non-inference, and does not print raw output", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_OUTPUT1024_TRUNCATION_FORENSICS_V1_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    collector: {
      local_private_file_read: boolean;
      raw_output_persisted_publicly: boolean;
      raw_output_printed_to_console: boolean;
      measures_top_level_section_markers: boolean;
      counts_completed_item_keys: boolean;
      measures_terminal_json_structure: boolean;
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      max_output_change_authorized: boolean;
      timeout_change_authorized: boolean;
      auto_repair_authorized: boolean;
    };
  };

  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-brookfield-output1024-truncation-forensics-v1.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_READ_ONLY_NO_INFERENCE");
  assert.equal(prep.collector.local_private_file_read, true);
  assert.equal(prep.collector.raw_output_persisted_publicly, false);
  assert.equal(prep.collector.raw_output_printed_to_console, false);
  assert.equal(
    prep.collector.measures_top_level_section_markers,
    true,
  );
  assert.equal(prep.collector.counts_completed_item_keys, true);
  assert.equal(
    prep.collector.measures_terminal_json_structure,
    true,
  );

  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.max_output_change_authorized, false);
  assert.equal(prep.authority.timeout_change_authorized, false);
  assert.equal(prep.authority.auto_repair_authorized, false);

  assert.match(source, /response\?\.rawText/);
  assert.match(source, /priority_findings/);
  assert.match(source, /material_conflicts/);
  assert.match(source, /weak_link_candidates/);
  assert.match(source, /unresolved_points/);
  assert.match(source, /rawOutputIncludedInConsole: false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
