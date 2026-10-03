import { createHash } from 'node:crypto';
import type { LoadResult } from './chatgpt-supabase-bridge';

const SHA256 = /^[0-9a-f]{64}$/;

export type PilotageContinuationAction =
  | 'RESOLVE_BLOCKER'
  | 'RESUME_STAGE'
  | 'CONTINUE_STAGE'
  | 'HANDOFF_NEXT_STAGE'
  | 'SAVE_DURABLE_CHECKPOINT'
  | 'AWAIT_EXPLICIT_GO_PUBLISH'
  | 'FAIL_CLOSED';

export type PilotageRequestedOperation =
  | 'RESOLVE_BLOCKER'
  | 'RESUME_STAGE'
  | 'CONTINUE_STAGE'
  | 'HANDOFF_NEXT_STAGE'
  | 'SAVE_DURABLE_CHECKPOINT'
  | 'GO_PUBLISH';

export type ExactArtifactRef = {
  artifact_id: string;
  version: number;
  content_sha256: string;
  required_authority_class: string;
};

export type LosslessResumeEnvelope = {
  envelope_version: '1.0.0';
  source: 'DURABLE_LOAD_RESULT';
  chat_memory_authority: false;
  mutation_allowed: false;
  run_id: string;
  run_state_version: number;
  run_status: string;
  contract_set_sha256: string;
  current_stage: NonNullable<LoadResult['current_stage']>;
  stage_revision: number;
  stage_state_version: number;
  lifecycle_status: NonNullable<LoadResult['stage']>['lifecycle_status'];
  handoff_gate_state: NonNullable<LoadResult['stage']>['handoff_gate_state'];
  active_manifest: ExactArtifactRef | null;
  process_state_artifact: ExactArtifactRef | null;
  blockers: Record<string, unknown>[];
  artifact_index: ExactArtifactRef[];
  context_plan: {
    l0: ExactArtifactRef[];
    l1: ExactArtifactRef[];
    l2: ExactArtifactRef[];
    l3: ExactArtifactRef[];
  };
  state_fingerprint_sha256: string;
  exact_next_action: PilotageContinuationAction;
  exact_next_action_reason: string;
};

export type PersistedContinuationAttempt = {
  source: 'PERSISTED_CONTINUATION_LEDGER';
  state_fingerprint_sha256: string;
  requested_operation: PilotageRequestedOperation;
};

export type PersistedAttemptRegistration = {
  decision: 'FIRST_ATTEMPT' | 'NO_PROGRESS_REPLAY';
  run_id: string;
  stage_code: string;
  state_fingerprint_sha256: string;
  requested_operation: PilotageRequestedOperation;
  exact_next_action: PilotageContinuationAction;
  attempt_count: number;
  retry_without_reload_allowed: false;
};

export type ContinuationAttemptPort = {
  registerAttempt(args: {
    p_run_id: string;
    p_stage_code: string;
    p_expected_run_state_version: number;
    p_expected_stage_state_version: number;
    p_state_fingerprint_sha256: string;
    p_requested_operation: PilotageRequestedOperation;
    p_exact_next_action: PilotageContinuationAction;
  }): Promise<PersistedAttemptRegistration>;
};

export type ContinuationGuardDecision =
  | {
      decision: 'PROCEED';
      dispatch_allowed: true;
      retry_allowed: false;
      current_state_fingerprint_sha256: string;
      exact_next_action: PilotageContinuationAction;
      reason: string;
    }
  | {
      decision: 'NO_PROGRESS' | 'ROUTE_MISMATCH' | 'FAIL_CLOSED';
      dispatch_allowed: false;
      retry_allowed: false;
      current_state_fingerprint_sha256: string;
      exact_next_action: PilotageContinuationAction;
      reason: string;
    };

export class PilotageContinuationError extends Error {
  constructor(
    readonly code:
      | 'NO_ACTIVE_RUN'
      | 'LOAD_INCONSISTENT'
      | 'NON_EXACT_ARTIFACT_REF'
      | 'CONTEXT_REF_CONFLICT'
      | 'CONTEXT_REF_NOT_IN_INDEX'
      | 'INVALID_PRIOR_ATTEMPT',
    message: string,
  ) {
    super(message);
    this.name = 'PilotageContinuationError';
  }
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const entries = Object.entries(value as Record<string, unknown>).sort(
    ([left], [right]) => left.localeCompare(right),
  );
  return `{${entries
    .map(([key, child]) => `${JSON.stringify(key)}:${canonicalJson(child)}`)
    .join(',')}}`;
}

function sha256(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

function exactRef(
  value: {
    artifact_id: string;
    version: number;
    content_sha256?: string | null;
    required_authority_class?: string | null;
  } | null,
  label: string,
): ExactArtifactRef | null {
  if (value === null) return null;
  if (
    !value.artifact_id ||
    !Number.isInteger(value.version) ||
    value.version < 1 ||
    !value.content_sha256 ||
    !SHA256.test(value.content_sha256) ||
    !value.required_authority_class
  ) {
    throw new PilotageContinuationError(
      'NON_EXACT_ARTIFACT_REF',
      `${label} is not an exact artifact identity`,
    );
  }
  return {
    artifact_id: value.artifact_id,
    version: value.version,
    content_sha256: value.content_sha256,
    required_authority_class: value.required_authority_class,
  };
}

function refKey(ref: ExactArtifactRef): string {
  return `${ref.artifact_id}:${ref.version}`;
}

function sortExactRefs(
  refs: Array<{
    artifact_id: string;
    version: number;
    content_sha256?: string | null;
    required_authority_class?: string | null;
  }>,
  label: string,
): ExactArtifactRef[] {
  const exact = refs.map((ref, index) => exactRef(ref, `${label}[${index}]`) as ExactArtifactRef);
  const byIdentity = new Map<string, ExactArtifactRef>();

  for (const ref of exact) {
    const key = refKey(ref);
    const prior = byIdentity.get(key);
    if (
      prior &&
      (prior.content_sha256 !== ref.content_sha256 ||
        prior.required_authority_class !== ref.required_authority_class)
    ) {
      throw new PilotageContinuationError(
        'CONTEXT_REF_CONFLICT',
        `${label} contains conflicting exact refs for ${key}`,
      );
    }
    byIdentity.set(key, ref);
  }

  return [...byIdentity.values()].sort((left, right) => {
    const identityOrder = refKey(left).localeCompare(refKey(right));
    if (identityOrder !== 0) return identityOrder;
    const hashOrder = left.content_sha256.localeCompare(right.content_sha256);
    if (hashOrder !== 0) return hashOrder;
    return left.required_authority_class.localeCompare(
      right.required_authority_class,
    );
  });
}

function assertRefsBackedByArtifactIndex(
  artifactIndex: ExactArtifactRef[],
  refs: Array<{ label: string; ref: ExactArtifactRef | null }>,
): void {
  const byIdentity = new Map(
    artifactIndex.map((ref) => [refKey(ref), ref] as const),
  );

  for (const { label, ref } of refs) {
    if (ref === null) continue;
    const indexed = byIdentity.get(refKey(ref));
    if (
      !indexed ||
      indexed.content_sha256 !== ref.content_sha256 ||
      indexed.required_authority_class !== ref.required_authority_class
    ) {
      throw new PilotageContinuationError(
        'CONTEXT_REF_NOT_IN_INDEX',
        `${label} is not backed by the exact LOAD_RESULT artifact_index`,
      );
    }
  }
}

function sortedBlockers(
  blockers: Record<string, unknown>[],
): Record<string, unknown>[] {
  return [...blockers].sort((left, right) =>
    canonicalJson(left).localeCompare(canonicalJson(right)),
  );
}

function nextAction(input: {
  stage: NonNullable<LoadResult['stage']>;
  currentStage: NonNullable<LoadResult['current_stage']>;
  blockers: Record<string, unknown>[];
  activeManifest: ExactArtifactRef | null;
  processStateArtifact: ExactArtifactRef | null;
}): {
  action: PilotageContinuationAction;
  reason: string;
} {
  const lifecycle = input.stage.lifecycle_status;

  if (lifecycle === 'COMPLETE' && input.blockers.length > 0) {
    return {
      action: 'FAIL_CLOSED',
      reason: 'COMPLETE stage cannot retain active blockers',
    };
  }

  if (
    input.stage.handoff_gate_state === 'YES' &&
    lifecycle !== 'COMPLETE'
  ) {
    return {
      action: 'FAIL_CLOSED',
      reason: 'handoff gate YES is incompatible with a non-COMPLETE stage',
    };
  }

  if (lifecycle === 'BLOCKED' || input.blockers.length > 0) {
    return {
      action: 'RESOLVE_BLOCKER',
      reason: 'authoritative stage state is blocked; blind analytical retry is forbidden',
    };
  }

  if (lifecycle === 'PAUSED') {
    if (!input.activeManifest && !input.processStateArtifact) {
      return {
        action: 'SAVE_DURABLE_CHECKPOINT',
        reason: 'paused stage has no durable resume anchor',
      };
    }
    return {
      action: 'RESUME_STAGE',
      reason: 'resume from persisted stage state and exact durable refs',
    };
  }

  if (lifecycle === 'IN_PROGRESS') {
    if (!input.activeManifest && !input.processStateArtifact) {
      return {
        action: 'SAVE_DURABLE_CHECKPOINT',
        reason: 'in-progress stage has no durable resume anchor',
      };
    }
    return {
      action: 'CONTINUE_STAGE',
      reason: 'continue from persisted stage state and exact durable refs',
    };
  }

  if (lifecycle === 'COMPLETE') {
    if (input.stage.handoff_gate_state !== 'YES') {
      return {
        action: 'FAIL_CLOSED',
        reason: `COMPLETE stage handoff gate is ${input.stage.handoff_gate_state}, expected YES`,
      };
    }
    if (input.currentStage === 'INTEGRATION') {
      return {
        action: 'AWAIT_EXPLICIT_GO_PUBLISH',
        reason: 'Integration is complete; publication requires a separate explicit GO PUBLISH',
      };
    }
    return {
      action: 'HANDOFF_NEXT_STAGE',
      reason: 'stage is complete and handoff gate is YES',
    };
  }

  return {
    action: 'FAIL_CLOSED',
    reason: `stage lifecycle ${lifecycle} is not resumable`,
  };
}

function fingerprintPayload(
  envelope: Omit<
    LosslessResumeEnvelope,
    'state_fingerprint_sha256' | 'exact_next_action' | 'exact_next_action_reason'
  >,
): Record<string, unknown> {
  return {
    envelope_version: envelope.envelope_version,
    source: envelope.source,
    run_id: envelope.run_id,
    run_state_version: envelope.run_state_version,
    run_status: envelope.run_status,
    contract_set_sha256: envelope.contract_set_sha256,
    current_stage: envelope.current_stage,
    stage_revision: envelope.stage_revision,
    stage_state_version: envelope.stage_state_version,
    lifecycle_status: envelope.lifecycle_status,
    handoff_gate_state: envelope.handoff_gate_state,
    active_manifest: envelope.active_manifest,
    process_state_artifact: envelope.process_state_artifact,
    artifact_index: envelope.artifact_index,
    blockers: envelope.blockers,
    context_plan: envelope.context_plan,
  };
}

export function buildLosslessResumeEnvelope(
  load: LoadResult,
): LosslessResumeEnvelope {
  if (
    !load.run_id ||
    !load.run_status ||
    !load.run_state_version ||
    !load.contract_set_sha256 ||
    !load.current_stage ||
    !load.stage
  ) {
    throw new PilotageContinuationError(
      'NO_ACTIVE_RUN',
      'lossless resume requires an active loaded run and current stage',
    );
  }

  if (load.stage.stage_code !== load.current_stage) {
    throw new PilotageContinuationError(
      'LOAD_INCONSISTENT',
      'current_stage differs from loaded stage identity',
    );
  }

  const activeManifest = exactRef(
    load.stage.active_manifest,
    'stage.active_manifest',
  );
  const processStateArtifact = exactRef(
    load.process_state_artifact ?? null,
    'process_state_artifact',
  );
  const blockers = sortedBlockers(load.blockers);
  const artifactIndex = sortExactRefs(load.artifact_index, 'artifact_index');
  const contextPlan = {
    l0: sortExactRefs(load.context_plan.l0, 'context_plan.l0'),
    l1: sortExactRefs(load.context_plan.l1, 'context_plan.l1'),
    l2: sortExactRefs(load.context_plan.l2, 'context_plan.l2'),
    l3: sortExactRefs(load.context_plan.l3, 'context_plan.l3'),
  };

  assertRefsBackedByArtifactIndex(artifactIndex, [
    { label: 'stage.active_manifest', ref: activeManifest },
    { label: 'process_state_artifact', ref: processStateArtifact },
    ...contextPlan.l0.map((ref, index) => ({
      label: `context_plan.l0[${index}]`,
      ref,
    })),
    ...contextPlan.l1.map((ref, index) => ({
      label: `context_plan.l1[${index}]`,
      ref,
    })),
    ...contextPlan.l2.map((ref, index) => ({
      label: `context_plan.l2[${index}]`,
      ref,
    })),
    ...contextPlan.l3.map((ref, index) => ({
      label: `context_plan.l3[${index}]`,
      ref,
    })),
  ]);

  const base = {
    envelope_version: '1.0.0' as const,
    source: 'DURABLE_LOAD_RESULT' as const,
    chat_memory_authority: false as const,
    mutation_allowed: false as const,
    run_id: load.run_id,
    run_state_version: load.run_state_version,
    run_status: load.run_status,
    contract_set_sha256: load.contract_set_sha256,
    current_stage: load.current_stage,
    stage_revision: load.stage.stage_revision,
    stage_state_version: load.stage.stage_state_version,
    lifecycle_status: load.stage.lifecycle_status,
    handoff_gate_state: load.stage.handoff_gate_state,
    active_manifest: activeManifest,
    process_state_artifact: processStateArtifact,
    blockers,
    artifact_index: artifactIndex,
    context_plan: contextPlan,
  };

  const route = nextAction({
    stage: load.stage,
    currentStage: load.current_stage,
    blockers,
    activeManifest,
    processStateArtifact,
  });

  return {
    ...base,
    state_fingerprint_sha256: sha256(canonicalJson(fingerprintPayload(base))),
    exact_next_action: route.action,
    exact_next_action_reason: route.reason,
  };
}

function operationForNextAction(
  action: PilotageContinuationAction,
): PilotageRequestedOperation | null {
  if (action === 'AWAIT_EXPLICIT_GO_PUBLISH' || action === 'FAIL_CLOSED') {
    return null;
  }
  return action;
}

export function evaluateContinuationGuard(input: {
  current: LosslessResumeEnvelope;
  requestedOperation: PilotageRequestedOperation;
  priorAttempt?: PersistedContinuationAttempt | null;
}): ContinuationGuardDecision {
  const requiredOperation = operationForNextAction(input.current.exact_next_action);

  if (requiredOperation === null) {
    return {
      decision: 'FAIL_CLOSED',
      dispatch_allowed: false,
      retry_allowed: false,
      current_state_fingerprint_sha256:
        input.current.state_fingerprint_sha256,
      exact_next_action: input.current.exact_next_action,
      reason: input.current.exact_next_action_reason,
    };
  }

  if (input.requestedOperation !== requiredOperation) {
    return {
      decision: 'ROUTE_MISMATCH',
      dispatch_allowed: false,
      retry_allowed: false,
      current_state_fingerprint_sha256:
        input.current.state_fingerprint_sha256,
      exact_next_action: input.current.exact_next_action,
      reason:
        `requested operation ${input.requestedOperation} does not match authoritative next action ${requiredOperation}`,
    };
  }

  const prior = input.priorAttempt ?? null;
  if (prior !== null) {
    if (prior.source !== 'PERSISTED_CONTINUATION_LEDGER') {
      throw new PilotageContinuationError(
        'INVALID_PRIOR_ATTEMPT',
        'prior attempt must come from the persisted continuation ledger',
      );
    }
    if (!SHA256.test(prior.state_fingerprint_sha256)) {
      throw new PilotageContinuationError(
        'INVALID_PRIOR_ATTEMPT',
        'prior attempt state fingerprint is invalid',
      );
    }

    if (
      prior.state_fingerprint_sha256 ===
        input.current.state_fingerprint_sha256 &&
      prior.requested_operation === input.requestedOperation
    ) {
      return {
        decision: 'NO_PROGRESS',
        dispatch_allowed: false,
        retry_allowed: false,
        current_state_fingerprint_sha256:
          input.current.state_fingerprint_sha256,
        exact_next_action: input.current.exact_next_action,
        reason:
          'same authoritative state and same requested operation; reload or persist a real state/input delta before retrying',
      };
    }
  }

  return {
    decision: 'PROCEED',
    dispatch_allowed: true,
    retry_allowed: false,
    current_state_fingerprint_sha256:
      input.current.state_fingerprint_sha256,
    exact_next_action: input.current.exact_next_action,
    reason:
      prior === null
        ? 'no persisted prior attempt for this continuation'
        : 'authoritative state or requested operation changed',
  };
}

export async function registerContinuationAttempt(
  port: ContinuationAttemptPort,
  current: LosslessResumeEnvelope,
  requestedOperation: PilotageRequestedOperation,
): Promise<ContinuationGuardDecision> {
  const route = evaluateContinuationGuard({
    current,
    requestedOperation,
  });
  if (route.decision !== 'PROCEED') return route;

  const registration = await port.registerAttempt({
    p_run_id: current.run_id,
    p_stage_code: current.current_stage,
    p_expected_run_state_version: current.run_state_version,
    p_expected_stage_state_version: current.stage_state_version,
    p_state_fingerprint_sha256: current.state_fingerprint_sha256,
    p_requested_operation: requestedOperation,
    p_exact_next_action: current.exact_next_action,
  });

  if (
    registration.run_id !== current.run_id ||
    registration.stage_code !== current.current_stage ||
    registration.state_fingerprint_sha256 !== current.state_fingerprint_sha256 ||
    registration.requested_operation !== requestedOperation ||
    registration.exact_next_action !== current.exact_next_action ||
    registration.retry_without_reload_allowed !== false ||
    !Number.isInteger(registration.attempt_count) ||
    registration.attempt_count < 1
  ) {
    throw new PilotageContinuationError(
      'INVALID_PRIOR_ATTEMPT',
      'persisted continuation registration does not match current durable state',
    );
  }

  if (registration.decision === 'NO_PROGRESS_REPLAY') {
    return {
      decision: 'NO_PROGRESS',
      dispatch_allowed: false,
      retry_allowed: false,
      current_state_fingerprint_sha256: current.state_fingerprint_sha256,
      exact_next_action: current.exact_next_action,
      reason:
        'same authoritative state and same requested operation already exists in the persisted continuation ledger',
    };
  }

  if (
    registration.decision !== 'FIRST_ATTEMPT' ||
    registration.attempt_count !== 1
  ) {
    throw new PilotageContinuationError(
      'INVALID_PRIOR_ATTEMPT',
      'first continuation registration returned an invalid decision or count',
    );
  }

  return route;
}

function formatRef(ref: ExactArtifactRef | null): string {
  if (!ref) return 'NONE';
  return [
    `${ref.artifact_id}@${ref.version}`,
    `sha256=${ref.content_sha256}`,
    `authority=${ref.required_authority_class}`,
  ].join(' ');
}

function formatTier(refs: ExactArtifactRef[]): string {
  return refs.length === 0 ? 'NONE' : refs.map(formatRef).join(' | ');
}

export function buildLosslessResumePrompt(
  envelope: LosslessResumeEnvelope,
): string {
  return [
    'OROTITAN — LOSSLESS RESUME',
    'SOURCE = DURABLE_LOAD_RESULT',
    'CHAT_MEMORY_AUTHORITY = NO',
    'MUTATION_ALLOWED = NO',
    `RUN_ID = ${envelope.run_id}`,
    `RUN_STATE_VERSION = ${envelope.run_state_version}`,
    `RUN_STATUS = ${envelope.run_status}`,
    `CONTRACT_SET_SHA256 = ${envelope.contract_set_sha256}`,
    `CURRENT_STAGE = ${envelope.current_stage}`,
    `STAGE_REVISION = ${envelope.stage_revision}`,
    `STAGE_STATE_VERSION = ${envelope.stage_state_version}`,
    `LIFECYCLE_STATUS = ${envelope.lifecycle_status}`,
    `HANDOFF_GATE_STATE = ${envelope.handoff_gate_state}`,
    `STATE_FINGERPRINT_SHA256 = ${envelope.state_fingerprint_sha256}`,
    `ACTIVE_MANIFEST = ${formatRef(envelope.active_manifest)}`,
    `PROCESS_STATE_ARTIFACT = ${formatRef(envelope.process_state_artifact)}`,
    `ARTIFACT_INDEX = ${formatTier(envelope.artifact_index)}`,
    `CONTEXT_L0 = ${formatTier(envelope.context_plan.l0)}`,
    `CONTEXT_L1 = ${formatTier(envelope.context_plan.l1)}`,
    `CONTEXT_L2 = ${formatTier(envelope.context_plan.l2)}`,
    `CONTEXT_L3 = ${formatTier(envelope.context_plan.l3)}`,
    `BLOCKERS = ${canonicalJson(envelope.blockers)}`,
    `EXACT_NEXT_ACTION = ${envelope.exact_next_action}`,
    `EXACT_NEXT_ACTION_REASON = ${envelope.exact_next_action_reason}`,
    'RULE = reload durable state before any mutation; never reconstruct authority from chat memory',
  ].join('\n');
}
