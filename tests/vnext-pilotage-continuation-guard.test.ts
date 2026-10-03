import assert from 'node:assert/strict';
import test from 'node:test';

import type { LoadResult } from '../lib/orotitan-equity/post-c7/chatgpt-supabase-bridge';
import {
  type ContinuationAttemptPort,
  PilotageContinuationError,
  buildLosslessResumeEnvelope,
  buildLosslessResumePrompt,
  evaluateContinuationGuard,
  registerContinuationAttempt,
} from '../lib/orotitan-equity/post-c7/pilotage-continuation-guard';

const RUN_ID = '40000000-0000-4000-8000-000000000001';
const HASH_A = 'a'.repeat(64);
const HASH_B = 'b'.repeat(64);

function ref(id: string, hash = HASH_A) {
  return {
    artifact_id: id,
    version: 1,
    content_sha256: hash,
    required_authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
  };
}

function load(overrides: Partial<LoadResult> = {}): LoadResult {
  const manifest = ref('50000000-0000-4000-8000-000000000001');
  const process = ref('50000000-0000-4000-8000-000000000002', HASH_B);
  const base: LoadResult = {
    contract_version: '0.1.0',
    operation: 'LOAD_RESULT',
    mutation_allowed: false,
    issuer_id: '10000000-0000-4000-8000-000000000001',
    security_id: '20000000-0000-4000-8000-000000000001',
    dossier_id: '30000000-0000-4000-8000-000000000001',
    run_id: RUN_ID,
    run_status: 'ACTIVE',
    run_state_version: 7,
    run_type: 'INITIAL',
    canonical_mode: 'ANALYZE',
    data_cutoff: '2026-09-22',
    contract_set_sha256: 'c'.repeat(64),
    current_stage: 'DEEP_DIVE',
    stage: {
      stage_code: 'DEEP_DIVE',
      stage_revision: 2,
      lifecycle_status: 'IN_PROGRESS',
      stage_state_version: 11,
      handoff_gate_state: 'NOT_EVALUATED',
      active_manifest: manifest,
    },
    blockers: [],
    artifact_index: [manifest, process],
    process_state_artifact: process,
    context_plan: {
      l0: [process, manifest],
      l1: [manifest, process],
      l2: [],
      l3: [],
    },
  };
  return { ...base, ...overrides };
}

test('lossless resume fingerprint is stable across blocker and context ordering', () => {
  const left = load({
    blockers: [{ code: 'B' }, { code: 'A' }],
  });
  const right = load({
    blockers: [{ code: 'A' }, { code: 'B' }],
    context_plan: {
      l0: [...left.context_plan.l0].reverse(),
      l1: [...left.context_plan.l1].reverse(),
      l2: [],
      l3: [],
    },
  });

  assert.equal(
    buildLosslessResumeEnvelope(left).state_fingerprint_sha256,
    buildLosslessResumeEnvelope(right).state_fingerprint_sha256,
  );
});

test('durable state version change changes the continuation fingerprint', () => {
  const before = buildLosslessResumeEnvelope(load());
  const after = buildLosslessResumeEnvelope(
    load({
      run_state_version: 8,
      stage: { ...load().stage!, stage_state_version: 12 },
    }),
  );

  assert.notEqual(
    before.state_fingerprint_sha256,
    after.state_fingerprint_sha256,
  );
});

test('blocked state routes to blocker resolution, never blind continuation', () => {
  const current = buildLosslessResumeEnvelope(
    load({
      stage: { ...load().stage!, lifecycle_status: 'BLOCKED' },
      blockers: [{ code: 'ECONOMIC_SHARE_COUNT_UNRESOLVED' }],
    }),
  );

  assert.equal(current.exact_next_action, 'RESOLVE_BLOCKER');
  const wrong = evaluateContinuationGuard({
    current,
    requestedOperation: 'CONTINUE_STAGE',
  });
  assert.equal(wrong.decision, 'ROUTE_MISMATCH');
  assert.equal(wrong.dispatch_allowed, false);
});

test('same persisted state and same operation is NO_PROGRESS', () => {
  const current = buildLosslessResumeEnvelope(load());
  const decision = evaluateContinuationGuard({
    current,
    requestedOperation: 'CONTINUE_STAGE',
    priorAttempt: {
      source: 'PERSISTED_CONTINUATION_LEDGER',
      state_fingerprint_sha256: current.state_fingerprint_sha256,
      requested_operation: 'CONTINUE_STAGE',
    },
  });

  assert.equal(decision.decision, 'NO_PROGRESS');
  assert.equal(decision.dispatch_allowed, false);
  assert.equal(decision.retry_allowed, false);
});

test('changed authoritative state permits continuation after reload', () => {
  const prior = buildLosslessResumeEnvelope(load());
  const current = buildLosslessResumeEnvelope(
    load({
      run_state_version: 8,
      stage: { ...load().stage!, stage_state_version: 12 },
    }),
  );

  const decision = evaluateContinuationGuard({
    current,
    requestedOperation: 'CONTINUE_STAGE',
    priorAttempt: {
      source: 'PERSISTED_CONTINUATION_LEDGER',
      state_fingerprint_sha256: prior.state_fingerprint_sha256,
      requested_operation: 'CONTINUE_STAGE',
    },
  });

  assert.equal(decision.decision, 'PROCEED');
  assert.equal(decision.dispatch_allowed, true);
  assert.equal(decision.retry_allowed, false);
});

test('paused run without durable anchor must establish a checkpoint before resume', () => {
  const current = buildLosslessResumeEnvelope(
    load({
      stage: {
        ...load().stage!,
        lifecycle_status: 'PAUSED',
        active_manifest: null,
      },
      process_state_artifact: null,
      context_plan: { l0: [], l1: [], l2: [], l3: [] },
    }),
  );

  assert.equal(current.exact_next_action, 'SAVE_DURABLE_CHECKPOINT');
  const decision = evaluateContinuationGuard({
    current,
    requestedOperation: 'RESUME_STAGE',
  });
  assert.equal(decision.decision, 'ROUTE_MISMATCH');
});

test('completed Integration never turns into publication without explicit GO PUBLISH', () => {
  const current = buildLosslessResumeEnvelope(
    load({
      current_stage: 'INTEGRATION',
      stage: {
        ...load().stage!,
        stage_code: 'INTEGRATION',
        lifecycle_status: 'COMPLETE',
        handoff_gate_state: 'YES',
      },
    }),
  );

  assert.equal(current.exact_next_action, 'AWAIT_EXPLICIT_GO_PUBLISH');
  const decision = evaluateContinuationGuard({
    current,
    requestedOperation: 'GO_PUBLISH',
  });
  assert.equal(decision.decision, 'FAIL_CLOSED');
  assert.equal(decision.dispatch_allowed, false);
});

test('completed non-Integration stage with YES gate routes to exact handoff', () => {
  const current = buildLosslessResumeEnvelope(
    load({
      stage: {
        ...load().stage!,
        lifecycle_status: 'COMPLETE',
        handoff_gate_state: 'YES',
      },
    }),
  );

  assert.equal(current.exact_next_action, 'HANDOFF_NEXT_STAGE');
  assert.equal(
    evaluateContinuationGuard({
      current,
      requestedOperation: 'HANDOFF_NEXT_STAGE',
    }).decision,
    'PROCEED',
  );
});

test('resume prompt explicitly rejects chat memory as authority', () => {
  const current = buildLosslessResumeEnvelope(load());
  const prompt = buildLosslessResumePrompt(current);

  assert.match(prompt, /CHAT_MEMORY_AUTHORITY = NO/);
  assert.match(prompt, new RegExp('RUN_ID = ' + RUN_ID));
  assert.match(prompt, /CANONICAL_MODE = ANALYZE/);
  assert.match(prompt, /DATA_CUTOFF = 2026-09-22/);
  assert.match(
    prompt,
    new RegExp('STATE_FINGERPRINT_SHA256 = ' + current.state_fingerprint_sha256),
  );
  assert.match(prompt, /ARTIFACT_INDEX = /);
  assert.match(prompt, /EXACT_NEXT_ACTION = CONTINUE_STAGE/);
  assert.match(prompt, /reload durable state before any mutation/);
});

test('non-exact context refs fail closed instead of being reconstructed', () => {
  const broken = load();
  broken.context_plan = {
    ...broken.context_plan,
    l1: [
      {
        artifact_id: '50000000-0000-4000-8000-000000000003',
        version: 1,
        content_sha256: null,
        required_authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
      },
    ],
  };

  assert.throws(
    () => buildLosslessResumeEnvelope(broken),
    (error: unknown) =>
      error instanceof PilotageContinuationError &&
      error.code === 'NON_EXACT_ARTIFACT_REF',
  );
});

test('conflicting duplicate refs fail closed', () => {
  const broken = load();
  broken.context_plan = {
    ...broken.context_plan,
    l1: [
      ref('50000000-0000-4000-8000-000000000003', HASH_A),
      ref('50000000-0000-4000-8000-000000000003', HASH_B),
    ],
  };

  assert.throws(
    () => buildLosslessResumeEnvelope(broken),
    (error: unknown) =>
      error instanceof PilotageContinuationError &&
      error.code === 'CONTEXT_REF_CONFLICT',
  );
});

test('resume refs absent from artifact_index fail closed', () => {
  const broken = load();
  broken.context_plan = {
    ...broken.context_plan,
    l2: [ref('50000000-0000-4000-8000-000000000099')],
  };

  assert.throws(
    () => buildLosslessResumeEnvelope(broken),
    (error: unknown) =>
      error instanceof PilotageContinuationError &&
      error.code === 'CONTEXT_REF_NOT_IN_INDEX',
  );
});

test('active manifest metadata must match artifact_index exactly', () => {
  const broken = load();
  broken.stage = {
    ...broken.stage!,
    active_manifest: {
      ...broken.stage!.active_manifest!,
      content_sha256: 'd'.repeat(64),
    },
  };

  assert.throws(
    () => buildLosslessResumeEnvelope(broken),
    (error: unknown) =>
      error instanceof PilotageContinuationError &&
      error.code === 'CONTEXT_REF_NOT_IN_INDEX',
  );
});

test('resume cannot be reconstructed when LOAD has no active run', () => {
  assert.throws(
    () =>
      buildLosslessResumeEnvelope(
        load({
          run_id: null,
          run_status: null,
          run_state_version: null,
          run_type: null,
          canonical_mode: null,
          data_cutoff: null,
          contract_set_sha256: null,
          current_stage: null,
          stage: null,
          artifact_index: [],
          process_state_artifact: null,
          context_plan: { l0: [], l1: [], l2: [], l3: [] },
        }),
      ),
    (error: unknown) =>
      error instanceof PilotageContinuationError &&
      error.code === 'NO_ACTIVE_RUN',
  );
});


test('durable continuation registration allows only the first attempt for the same state and operation', async () => {
  const current = buildLosslessResumeEnvelope(load());
  let count = 0;
  const port: ContinuationAttemptPort = {
    async registerAttempt(args) {
      count += 1;
      return {
        decision: count === 1 ? 'FIRST_ATTEMPT' as const : 'NO_PROGRESS_REPLAY' as const,
        run_id: args.p_run_id,
        stage_code: args.p_stage_code,
        state_fingerprint_sha256: args.p_state_fingerprint_sha256,
        requested_operation: args.p_requested_operation,
        exact_next_action: args.p_exact_next_action,
        attempt_count: 1,
        retry_without_reload_allowed: false as const,
      };
    },
  };

  const first = await registerContinuationAttempt(
    port,
    current,
    'CONTINUE_STAGE',
  );
  const replay = await registerContinuationAttempt(
    port,
    current,
    'CONTINUE_STAGE',
  );

  assert.equal(first.decision, 'PROCEED');
  assert.equal(first.dispatch_allowed, true);
  assert.equal(first.retry_allowed, false);
  assert.equal(replay.decision, 'NO_PROGRESS');
  assert.equal(replay.dispatch_allowed, false);
});

test('route mismatch is rejected before durable attempt registration', async () => {
  const current = buildLosslessResumeEnvelope(
    load({
      stage: { ...load().stage!, lifecycle_status: 'BLOCKED' },
      blockers: [{ code: 'BLOCKER' }],
    }),
  );
  let called = false;

  const decision = await registerContinuationAttempt(
    {
      async registerAttempt() {
        called = true;
        throw new Error('should not be called');
      },
    },
    current,
    'CONTINUE_STAGE',
  );

  assert.equal(decision.decision, 'ROUTE_MISMATCH');
  assert.equal(called, false);
});


test('malformed persisted continuation registration fails closed', async () => {
  const current = buildLosslessResumeEnvelope(load());
  const port: ContinuationAttemptPort = {
    async registerAttempt(args) {
      return {
        decision: 'FIRST_ATTEMPT',
        run_id: args.p_run_id,
        stage_code: args.p_stage_code,
        state_fingerprint_sha256: '0'.repeat(64),
        requested_operation: args.p_requested_operation,
        exact_next_action: args.p_exact_next_action,
        attempt_count: 1,
        retry_without_reload_allowed: false,
      };
    },
  };

  await assert.rejects(
    registerContinuationAttempt(port, current, 'CONTINUE_STAGE'),
    (error: unknown) =>
      error instanceof PilotageContinuationError &&
      error.code === 'INVALID_PRIOR_ATTEMPT',
  );
});
