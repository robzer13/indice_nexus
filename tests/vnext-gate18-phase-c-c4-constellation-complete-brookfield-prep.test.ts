import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Constellation human adjudication closes the C4 cell with noncritical carry", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_HUMAN_ADJUDICATION_RESULT_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    engineering_carry: {
      disposition: string;
      output_budget_fully_consumed: boolean;
    };
    human_quality: Record<string, string>;
    carry_codes: string[];
    disposition: {
      human_adjudication_completed: boolean;
      critical_human_quality_failure: boolean;
      clean_pass: boolean;
      pass_with_carry: boolean;
      matrix_cell_completed: boolean;
    };
    authority: {
      retry_authorized: boolean;
      auto_repair_authorized: boolean;
      parameter_change_authorized: boolean;
    };
  };

  assert.equal(
    result.status,
    "COMPLETED_WITH_HUMAN_QUALITY_CARRY",
  );
  assert.equal(
    result.engineering_carry.output_budget_fully_consumed,
    false,
  );
  assert.equal(
    result.engineering_carry.disposition,
    "NO_OUTPUT_BUDGET_SATURATION_CARRY",
  );
  assert.equal(Object.keys(result.human_quality).length, 10);
  assert.equal(result.carry_codes.length, 5);
  assert.equal(
    result.disposition.human_adjudication_completed,
    true,
  );
  assert.equal(
    result.disposition.critical_human_quality_failure,
    false,
  );
  assert.equal(result.disposition.clean_pass, false);
  assert.equal(result.disposition.pass_with_carry, true);
  assert.equal(result.disposition.matrix_cell_completed, true);
  assert.equal(result.authority.retry_authorized, false);
  assert.equal(result.authority.auto_repair_authorized, false);
  assert.equal(result.authority.parameter_change_authorized, false);
});

test("Brookfield prep is exact-hash-bound and has no execution authority", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_SINGLE_CELL_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    frozen_inference: {
      company: string;
      evidence_count: number;
      conflict_count: number;
      packet_sha256: string;
      prompt_sha256: string;
      request_sha256: string;
      prompt_bytes: number;
      context_tokens: number;
      max_output_tokens: number;
      client_timeout_ms: number;
      transport: string;
    };
    constraints: {
      authorized_run_count: number;
      inference_authorized: boolean;
      automatic_retry_authorized: boolean;
    };
  };

  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-brookfield-qwen4b-context16384-output1024-timeout480-loopback-guarded.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_NOT_AUTHORIZED");
  assert.equal(prep.frozen_inference.company, "Brookfield Corporation");
  assert.equal(prep.frozen_inference.evidence_count, 12);
  assert.equal(prep.frozen_inference.conflict_count, 2);
  assert.equal(
    prep.frozen_inference.packet_sha256,
    "eb2d779b95fa5f207e91fd3485ece39b2bd101c2589eaaf8c48481f750ef75b3",
  );
  assert.equal(
    prep.frozen_inference.prompt_sha256,
    "516496bf4dc9bcdce47897c056282eb2bda63b014cd63c1b76e3992b1c0cdad2",
  );
  assert.equal(
    prep.frozen_inference.request_sha256,
    "28d1917a888bf5981fb4b524243af2d78ee67feb33e5e1554a5c4d787520dd80",
  );
  assert.equal(prep.frozen_inference.prompt_bytes, 9643);
  assert.equal(prep.frozen_inference.context_tokens, 16384);
  assert.equal(prep.frozen_inference.max_output_tokens, 1024);
  assert.equal(prep.frozen_inference.client_timeout_ms, 480000);
  assert.equal(
    prep.frozen_inference.transport,
    "NODE_HTTP_REQUEST_LOOPBACK",
  );
  assert.equal(prep.constraints.authorized_run_count, 0);
  assert.equal(prep.constraints.inference_authorized, false);
  assert.equal(prep.constraints.automatic_retry_authorized, false);

  assert.match(source, /Brookfield Corporation/);
  assert.doesNotMatch(source, /Constellation Software/);
  assert.match(
    source,
    /AUTHORIZED_SINGLE_LOCAL_C4_CELL_INFERENCE/,
  );
  assert.match(source, /const CLIENT_TIMEOUT_MS = 480_000;/);
  assert.match(source, /EXPECTED_PACKET_SHA256/);
  assert.match(source, /EXPECTED_PROMPT_SHA256/);
  assert.match(source, /EXPECTED_REQUEST_SHA256/);
  assert.match(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
