import assert from 'node:assert/strict';
import test from 'node:test';

import {
  handleContinuationRequest,
  parseContinuationRequest,
  type ContinuationRouteDependencies,
} from '../app/api/orotitan/pilotage/continuation/route';
import type {
  ContinuationGuardDecision,
  LosslessResumeEnvelope,
} from '../lib/orotitan-equity/post-c7/pilotage-continuation-guard';

const RUN_ID = '40000000-0000-4000-8000-000000000001';
const HASH = 'a'.repeat(64);

function envelope(): LosslessResumeEnvelope {
  return {
    envelope_version: '1.0.0',
    source: 'DURABLE_LOAD_RESULT',
    chat_memory_authority: false,
    mutation_allowed: false,
    run_id: RUN_ID,
    issuer_id: '10000000-0000-4000-8000-000000000001',
    security_id: '20000000-0000-4000-8000-000000000001',
    dossier_id: '30000000-0000-4000-8000-000000000001',
    run_state_version: 7,
    run_status: 'ACTIVE',
    run_type: 'INITIAL',
    canonical_mode: 'ANALYZE',
    data_cutoff: '2026-09-22',
    contract_set_sha256: 'c'.repeat(64),
    current_stage: 'DEEP_DIVE',
    stage_revision: 2,
    stage_state_version: 11,
    lifecycle_status: 'IN_PROGRESS',
    handoff_gate_state: 'NOT_EVALUATED',
    active_manifest: {
      artifact_id: '50000000-0000-4000-8000-000000000001',
      version: 1,
      content_sha256: HASH,
      required_authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
    },
    process_state_artifact: null,
    blockers: [],
    artifact_index: [
      {
        artifact_id: '50000000-0000-4000-8000-000000000001',
        version: 1,
        content_sha256: HASH,
        required_authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
      },
    ],
    context_plan: {
      l0: [],
      l1: [],
      l2: [],
      l3: [],
    },
    state_fingerprint_sha256: 'd'.repeat(64),
    exact_next_action: 'CONTINUE_STAGE',
    exact_next_action_reason:
      'continue from persisted stage state and exact durable refs',
  };
}

function request(body: unknown): Request {
  return new Request(
    'https://orotitan.example/api/orotitan/pilotage/continuation',
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    },
  );
}

function proceedDecision(): ContinuationGuardDecision {
  return {
    decision: 'PROCEED',
    dispatch_allowed: true,
    retry_allowed: false,
    current_state_fingerprint_sha256: 'd'.repeat(64),
    exact_next_action: 'CONTINUE_STAGE',
    reason: 'no persisted prior attempt for this continuation',
  };
}

function noProgressDecision(): ContinuationGuardDecision {
  return {
    decision: 'NO_PROGRESS',
    dispatch_allowed: false,
    retry_allowed: false,
    current_state_fingerprint_sha256: 'd'.repeat(64),
    exact_next_action: 'CONTINUE_STAGE',
    reason:
      'same authoritative state already exists in persisted continuation ledger',
  };
}

test('continuation request parser accepts only bounded Pilotage inputs', () => {
  assert.deepEqual(
    parseContinuationRequest({
      issuer_query: 'Veolia',
      run_id: RUN_ID,
      requested_operation: 'CONTINUE_STAGE',
    }),
    {
      issuerQuery: 'Veolia',
      runId: RUN_ID,
      requestedOperation: 'CONTINUE_STAGE',
    },
  );

  assert.throws(
    () =>
      parseContinuationRequest({
        issuer_query: 'Veolia',
        run_id: RUN_ID,
        requested_operation: 'CONTINUE_STAGE',
        bypass_guard: true,
      }),
    /unsupported request field: bypass_guard/,
  );

  assert.throws(
    () =>
      parseContinuationRequest({
        issuer_query: 'Veolia',
        run_id: RUN_ID,
        requested_operation: 'PUBLISH',
      }),
    /requested_operation is not a supported Pilotage operation/,
  );
});

test('runtime continuation entrypoint requires admin before durable LOAD', async () => {
  let loaded = false;
  const dependencies: ContinuationRouteDependencies = {
    async requireAdmin() {
      throw new Error('UNAUTHORIZED');
    },
    async loadEnvelope() {
      loaded = true;
      return envelope();
    },
    async registerAttempt() {
      throw new Error('should not register');
    },
  };

  const response = await handleContinuationRequest(
    request({
      issuer_query: 'Veolia',
      run_id: RUN_ID,
      requested_operation: 'CONTINUE_STAGE',
    }),
    dependencies,
  );

  assert.equal(response.status, 401);
  assert.equal(loaded, false);
  assert.equal(
    response.headers.get('cache-control'),
    'private, no-store, max-age=0',
  );
});

test('runtime continuation performs durable LOAD then anti-loop registration before handoff admission', async () => {
  const calls: string[] = [];
  const current = envelope();

  const dependencies: ContinuationRouteDependencies = {
    async requireAdmin() {
      calls.push('AUTH');
    },
    async loadEnvelope(input) {
      calls.push('LOAD');
      assert.deepEqual(input, {
        issuerQuery: 'Veolia',
        runId: RUN_ID,
      });
      return current;
    },
    async registerAttempt(input) {
      calls.push('REGISTER');
      assert.equal(input.envelope, current);
      assert.equal(input.requestedOperation, 'CONTINUE_STAGE');
      return proceedDecision();
    },
  };

  const response = await handleContinuationRequest(
    request({
      issuer_query: 'Veolia',
      run_id: RUN_ID,
      requested_operation: 'CONTINUE_STAGE',
    }),
    dependencies,
  );

  assert.equal(response.status, 200);
  assert.deepEqual(calls, ['AUTH', 'LOAD', 'REGISTER']);

  const body = (await response.json()) as Record<string, unknown>;
  assert.equal(body.source, 'DURABLE_LOAD_RESULT');
  assert.equal(body.chat_memory_authority, false);
  assert.equal(body.mutation_scope, 'PILOTAGE_ATTEMPT_LEDGER_ONLY');
  assert.equal(body.admission_status, 'CREATED');
  assert.equal(body.handoff_allowed, true);
  assert.equal(body.guard_decision, 'PROCEED');
  assert.equal(body.guard_dispatch_allowed, true);
  assert.equal(body.blind_retry_allowed, false);
  assert.equal(body.exact_next_action, 'CONTINUE_STAGE');
  assert.equal(body.handoff_idempotency_key, body.admission_id);
  assert.match(String(body.admission_id), /^[0-9a-f]{64}$/);
  assert.match(
    String(body.lossless_resume_prompt),
    /CHAT_MEMORY_AUTHORITY = NO/,
  );
  assert.match(
    String(body.lossless_resume_prompt),
    /EXACT_NEXT_ACTION = CONTINUE_STAGE/,
  );
});

test('lost response can recover the same durable admission without creating a new handoff identity', async () => {
  let registrationCalls = 0;
  const dependencies: ContinuationRouteDependencies = {
    async requireAdmin() {},
    async loadEnvelope() {
      return envelope();
    },
    async registerAttempt() {
      registrationCalls += 1;
      return registrationCalls === 1
        ? proceedDecision()
        : noProgressDecision();
    },
  };

  const body = {
    issuer_query: 'Veolia',
    run_id: RUN_ID,
    requested_operation: 'CONTINUE_STAGE',
  };

  const firstResponse = await handleContinuationRequest(
    request(body),
    dependencies,
  );
  const replayResponse = await handleContinuationRequest(
    request(body),
    dependencies,
  );

  assert.equal(firstResponse.status, 200);
  assert.equal(replayResponse.status, 200);
  assert.equal(registrationCalls, 2);

  const first = (await firstResponse.json()) as Record<string, unknown>;
  const replay = (await replayResponse.json()) as Record<string, unknown>;

  assert.equal(first.admission_status, 'CREATED');
  assert.equal(replay.admission_status, 'RECOVERED');
  assert.equal(first.handoff_allowed, true);
  assert.equal(replay.handoff_allowed, true);
  assert.equal(first.guard_decision, 'PROCEED');
  assert.equal(replay.guard_decision, 'NO_PROGRESS');
  assert.equal(first.guard_dispatch_allowed, true);
  assert.equal(replay.guard_dispatch_allowed, false);
  assert.equal(first.blind_retry_allowed, false);
  assert.equal(replay.blind_retry_allowed, false);
  assert.equal(first.admission_id, replay.admission_id);
  assert.equal(
    first.handoff_idempotency_key,
    replay.handoff_idempotency_key,
  );
  assert.equal(
    replay.handoff_idempotency_key,
    replay.admission_id,
  );
  assert.equal(
    first.lossless_resume_prompt,
    replay.lossless_resume_prompt,
  );
});

test('route mismatch remains denied and never becomes a recoverable admission', async () => {
  const mismatch: ContinuationGuardDecision = {
    decision: 'ROUTE_MISMATCH',
    dispatch_allowed: false,
    retry_allowed: false,
    current_state_fingerprint_sha256: 'd'.repeat(64),
    exact_next_action: 'CONTINUE_STAGE',
    reason: 'requested operation does not match authoritative next action',
  };

  const dependencies: ContinuationRouteDependencies = {
    async requireAdmin() {},
    async loadEnvelope() {
      return envelope();
    },
    async registerAttempt() {
      return mismatch;
    },
  };

  const response = await handleContinuationRequest(
    request({
      issuer_query: 'Veolia',
      run_id: RUN_ID,
      requested_operation: 'RESUME_STAGE',
    }),
    dependencies,
  );

  assert.equal(response.status, 409);
  const body = (await response.json()) as Record<string, unknown>;
  assert.equal(body.admission_status, 'DENIED');
  assert.equal(body.handoff_allowed, false);
  assert.equal(body.handoff_idempotency_key, null);
  assert.equal(body.guard_decision, 'ROUTE_MISMATCH');
});

test('runtime continuation sanitizes downstream failures and fails closed', async () => {
  const dependencies: ContinuationRouteDependencies = {
    async requireAdmin() {},
    async loadEnvelope() {
      return envelope();
    },
    async registerAttempt() {
      throw new Error('sensitive database detail');
    },
  };

  const response = await handleContinuationRequest(
    request({
      issuer_query: 'Veolia',
      run_id: RUN_ID,
      requested_operation: 'CONTINUE_STAGE',
    }),
    dependencies,
  );

  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), {
    error: 'OROTITAN_CONTINUATION_RUNTIME_FAILURE',
    message: 'Pilotage continuation admission failed closed',
  });
});
