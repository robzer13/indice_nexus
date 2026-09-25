import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Constellation timeout480 automated result is natural-stop PASS pending human adjudication", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_TIMEOUT480_RESULT_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    execution: {
      wall_clock_ms: number;
      done_reason: string;
      prompt_eval_count: number;
      eval_count: number;
      runtime_error: null;
      schema_valid: boolean;
      semantic_valid: boolean;
    };
    output_budget: {
      max_output_tokens: number;
      eval_count: number;
      remaining_token_margin: number;
      fully_consumed: boolean;
      finish_reason_stop: boolean;
      classification: string;
    };
    automated_disposition: {
      automated_engineering_pass: boolean;
      human_quality_adjudication_required: boolean;
      matrix_cell_completed: boolean;
    };
    authority: {
      retry_authorized: boolean;
      second_retry_authorized: boolean;
    };
  };

  assert.equal(
    result.status,
    "PASS_AUTOMATED_HUMAN_ADJUDICATION_PENDING",
  );
  assert.equal(result.execution.wall_clock_ms, 376327);
  assert.equal(result.execution.done_reason, "stop");
  assert.equal(result.execution.prompt_eval_count, 3348);
  assert.equal(result.execution.eval_count, 962);
  assert.equal(result.execution.runtime_error, null);
  assert.equal(result.execution.schema_valid, true);
  assert.equal(result.execution.semantic_valid, true);
  assert.equal(result.output_budget.max_output_tokens, 1024);
  assert.equal(result.output_budget.eval_count, 962);
  assert.equal(result.output_budget.remaining_token_margin, 62);
  assert.equal(result.output_budget.fully_consumed, false);
  assert.equal(result.output_budget.finish_reason_stop, true);
  assert.equal(
    result.output_budget.classification,
    "NATURAL_STOP_WITH_OUTPUT_BUDGET_MARGIN",
  );
  assert.equal(
    result.automated_disposition.automated_engineering_pass,
    true,
  );
  assert.equal(
    result.automated_disposition.human_quality_adjudication_required,
    true,
  );
  assert.equal(
    result.automated_disposition.matrix_cell_completed,
    false,
  );
  assert.equal(result.authority.retry_authorized, false);
  assert.equal(result.authority.second_retry_authorized, false);
});

test("Constellation human adjudication export is private, exact-identity-bound, and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_HUMAN_ADJUDICATION_EXPORT_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    engineering_observations: {
      done_reason: string;
      output_budget_fully_consumed: boolean;
      output_token_margin: number;
    };
    output_policy: {
      public_repo_persistence: boolean;
      generated_content_publication: boolean;
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      auto_repair_authorized: boolean;
      final_moat_conclusion_authorized: boolean;
      publication_authority: boolean;
    };
  };

  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-human-adjudication-export.ts",
    "utf8",
  );

  assert.equal(
    prep.status,
    "PREPARED_LOCAL_PRIVATE_NO_INFERENCE",
  );
  assert.equal(prep.engineering_observations.done_reason, "stop");
  assert.equal(
    prep.engineering_observations.output_budget_fully_consumed,
    false,
  );
  assert.equal(
    prep.engineering_observations.output_token_margin,
    62,
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
  assert.equal(prep.authority.publication_authority, false);

  assert.match(source, /Constellation Software/);
  assert.match(
    source,
    /TIMEOUT480_LOOPBACK_GUARDED_001/,
  );
  assert.match(
    source,
    /9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8/,
  );
  assert.match(
    source,
    /0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8/,
  );
  assert.match(source, /buildVerifiedGate18V10MoatPacket/);
  assert.match(source, /READY_FOR_HUMAN_ADJUDICATION/);
  assert.match(source, /publicRepoPersistence: false/);
  assert.match(source, /noInferenceExecuted: true/);
  assert.doesNotMatch(source, /STMicroelectronics/);
  assert.doesNotMatch(source, /TIMEOUT420_LOOPBACK_GUARDED_001/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
