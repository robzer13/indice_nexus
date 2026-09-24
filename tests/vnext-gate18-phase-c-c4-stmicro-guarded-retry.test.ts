import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("guarded STMicro retry remains authorization-gated and invariant-preserving", () => {
  const runner = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-stmicro-qwen4b-context16384-guarded-retry.ts",
    "utf8",
  );
  const helper = readFileSync(
    "runtime/vnext/windows-sleep-guard.ts",
    "utf8",
  );
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_GUARDED_RETRY_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    attempt: Record<string, unknown>;
    future_authorization: Record<string, unknown>;
    authority: Record<string, unknown>;
  };

  assert.equal(prep.status, "PREPARED_NOT_AUTHORIZED");
  assert.equal(prep.future_authorization.authorized, false);
  assert.equal(prep.future_authorization.authorized_run_count, 0);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.stmicro_retry_authorized, false);
  assert.equal(prep.authority.automatic_retry_authorized, false);

  assert.equal(prep.attempt.model_name, "qwen3:4b-instruct");
  assert.equal(prep.attempt.context_tokens, 16384);
  assert.equal(prep.attempt.max_output_tokens, 768);
  assert.equal(prep.attempt.temperature, 0);
  assert.equal(prep.attempt.client_timeout_ms, 300000);
  assert.equal(
    prep.attempt.prompt_sha256,
    "3f06d6e22dbbc88abbbc64fd470811f438a595c71154f7fcb8c004a6d4f5bc91",
  );

  assert.match(runner, /AUTHORIZED_SINGLE_LOCAL_GUARDED_RETRY/);
  assert.match(runner, /sleep_guard_required !== true/);
  assert.match(runner, /acquireWindowsSystemRequiredGuard/);
  assert.match(runner, /await sleepGuard\.release\(\)/);

  assert.match(helper, /ToUInt32\('80000000', 16\)/);
  assert.match(helper, /ES_SYSTEM_REQUIRED/);
  assert.match(helper, /SetThreadExecutionState\(\$ES_CONTINUOUS\)/);
  assert.doesNotMatch(helper, /ES_DISPLAY_REQUIRED/);
  assert.doesNotMatch(helper, /ES_AWAYMODE_REQUIRED/);
});
