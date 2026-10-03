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
    exact_next_action_reason: 'continue from persisted stage state and exact durable refs',
  };
}

function request(body: unknown): Request {
  return new Request('https://orotitan.example/api/orotitan/pilotage/continuation', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
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
  assert.equal(response.headers.get('cache-control'), 'private, no-store, max-age=0');
});

test('runtime continuation performs durable LOAD then anti-loop registration before dispatch admission', async () => {
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
  assert.equal(body.decision, 'PROCEED');
  assert.equal(body.dispatch_allowed, true);
  assert.equal(body.retry_allowed, false);
  assert.equal(body.exact_next_action, 'CONTINUE_STAGE');
  assert.match(String(body.lossless_resume_prompt), /CHAT_MEMORY_AUTHORITY = NO/);
  assert.match(String(body.lossless_resume_prompt), /EXACT_NEXT_ACTION = CONTINUE_STAGE/);
});

test('runtime continuation returns NO_PROGRESS without dispatch admission', async () => {
  const noProgress: ContinuationGuardDecision = {
    decision: 'NO_PROGRESS',
    dispatch_allowed: false,
    retry_allowed: false,
    current_state_fingerprint_sha256: 'd'.repeat(64),
    exact_next_action: 'CONTINUE_STAGE',
    reason: 'same authoritative state already exists in persisted continuation ledger',
  };

  const dependencies: ContinuationRouteDependencies = {
    async requireAdmin() {},
    async loadEnvelope() {
      return envelope();
    },
    async registerAttempt() {
      return noProgress;
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

  assert.equal(response.status, 409);
  const body = (await response.json()) as Record<string, unknown>;
  assert.equal(body.decision, 'NO_PROGRESS');
  assert.equal(body.dispatch_allowed, false);
  assert.equal(body.mutation_scope, 'NONE');
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
