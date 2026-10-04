import { createHash } from 'node:crypto';
import {
  buildLosslessResumePrompt,
  type ContinuationGuardDecision,
  type LosslessResumeEnvelope,
  type PilotageRequestedOperation,
} from '@/lib/orotitan-equity/post-c7/pilotage-continuation-guard';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

const OPERATIONS = new Set<PilotageRequestedOperation>([
  'RESOLVE_BLOCKER',
  'RESUME_STAGE',
  'CONTINUE_STAGE',
  'HANDOFF_NEXT_STAGE',
  'SAVE_DURABLE_CHECKPOINT',
  'GO_PUBLISH',
]);

type ContinuationRequest = {
  issuerQuery: string;
  runId: string;
  requestedOperation: PilotageRequestedOperation;
};

type AdmissionStatus = 'CREATED' | 'RECOVERED' | 'DENIED';

export type ContinuationRouteDependencies = {
  requireAdmin: () => Promise<void>;
  loadEnvelope: (input: {
    issuerQuery: string;
    runId: string;
  }) => Promise<LosslessResumeEnvelope>;
  registerAttempt: (input: {
    envelope: LosslessResumeEnvelope;
    requestedOperation: PilotageRequestedOperation;
  }) => Promise<ContinuationGuardDecision>;
};

const defaultDependencies: ContinuationRouteDependencies = {
  async requireAdmin() {
    const { requireAdminSession } = await import('@/lib/auth/admin-session');
    await requireAdminSession();
  },
  async loadEnvelope(input) {
    const { loadServerLosslessResumeEnvelope } = await import(
      '@/lib/orotitan-equity/post-c7/pilotage-continuation-guard-server'
    );
    return loadServerLosslessResumeEnvelope(input);
  },
  async registerAttempt(input) {
    const { registerServerContinuationAttempt } = await import(
      '@/lib/orotitan-equity/post-c7/pilotage-continuation-guard-server'
    );
    return registerServerContinuationAttempt(input);
  },
};

function json(body: unknown, status = 200): Response {
  return Response.json(body, {
    status,
    headers: {
      'Cache-Control': 'private, no-store, max-age=0',
      Vary: 'Cookie',
    },
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function admissionId(
  envelope: LosslessResumeEnvelope,
  requestedOperation: PilotageRequestedOperation,
): string {
  const durableAttemptIdentity = [
    'OROTITAN_PILOTAGE_ADMISSION_V1',
    envelope.run_id,
    envelope.current_stage,
    String(envelope.run_state_version),
    String(envelope.stage_state_version),
    requestedOperation,
    envelope.state_fingerprint_sha256,
  ].join('\n');

  return createHash('sha256')
    .update(durableAttemptIdentity, 'utf8')
    .digest('hex');
}

function admissionStatus(
  decision: ContinuationGuardDecision,
): AdmissionStatus {
  if (decision.decision === 'PROCEED') return 'CREATED';
  if (decision.decision === 'NO_PROGRESS') return 'RECOVERED';
  return 'DENIED';
}

function handoffAllowed(decision: ContinuationGuardDecision): boolean {
  return (
    decision.decision === 'PROCEED' ||
    decision.decision === 'NO_PROGRESS'
  );
}

export function parseContinuationRequest(value: unknown): ContinuationRequest {
  if (!isRecord(value)) {
    throw new Error('request body must be a JSON object');
  }

  const allowed = new Set(['issuer_query', 'run_id', 'requested_operation']);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      throw new Error(`unsupported request field: ${key}`);
    }
  }

  const issuerQuery =
    typeof value.issuer_query === 'string' ? value.issuer_query.trim() : '';
  const runId = typeof value.run_id === 'string' ? value.run_id.trim() : '';
  const requestedOperation =
    typeof value.requested_operation === 'string'
      ? value.requested_operation.trim()
      : '';

  if (issuerQuery.length < 1 || issuerQuery.length > 256) {
    throw new Error('issuer_query must contain 1 to 256 characters');
  }
  if (!UUID_PATTERN.test(runId)) {
    throw new Error('run_id must be a UUID');
  }
  if (!OPERATIONS.has(requestedOperation as PilotageRequestedOperation)) {
    throw new Error('requested_operation is not a supported Pilotage operation');
  }

  return {
    issuerQuery,
    runId,
    requestedOperation: requestedOperation as PilotageRequestedOperation,
  };
}

export async function handleContinuationRequest(
  request: Request,
  dependencies: ContinuationRouteDependencies = defaultDependencies,
): Promise<Response> {
  try {
    await dependencies.requireAdmin();
  } catch {
    return json(
      {
        error: 'UNAUTHORIZED',
        message: 'Pilotage continuation requires an authenticated admin session',
      },
      401,
    );
  }

  let parsed: ContinuationRequest;
  try {
    const contentType = request.headers.get('content-type') ?? '';
    if (!contentType.toLowerCase().startsWith('application/json')) {
      throw new Error('content-type must be application/json');
    }
    parsed = parseContinuationRequest(await request.json());
  } catch (error) {
    return json(
      {
        error: 'OROTITAN_CONTINUATION_REQUEST_INVALID',
        message:
          error instanceof Error ? error.message : 'invalid continuation request',
      },
      400,
    );
  }

  try {
    const envelope = await dependencies.loadEnvelope({
      issuerQuery: parsed.issuerQuery,
      runId: parsed.runId,
    });
    const decision = await dependencies.registerAttempt({
      envelope,
      requestedOperation: parsed.requestedOperation,
    });

    const status = admissionStatus(decision);
    const canHandoff = handoffAllowed(decision);
    const durableAdmissionId = admissionId(
      envelope,
      parsed.requestedOperation,
    );

    return json(
      {
        continuation_version: '1.0.0',
        source: envelope.source,
        chat_memory_authority: envelope.chat_memory_authority,
        mutation_scope: canHandoff
          ? 'PILOTAGE_ATTEMPT_LEDGER_ONLY'
          : 'NONE',
        run_id: envelope.run_id,
        current_stage: envelope.current_stage,
        run_state_version: envelope.run_state_version,
        stage_state_version: envelope.stage_state_version,
        state_fingerprint_sha256: envelope.state_fingerprint_sha256,
        exact_next_action: envelope.exact_next_action,
        requested_operation: parsed.requestedOperation,
        admission_id: durableAdmissionId,
        admission_status: status,
        handoff_allowed: canHandoff,
        handoff_idempotency_key: canHandoff ? durableAdmissionId : null,
        guard_decision: decision.decision,
        guard_dispatch_allowed: decision.dispatch_allowed,
        blind_retry_allowed: decision.retry_allowed,
        reason: decision.reason,
        lossless_resume_prompt: buildLosslessResumePrompt(envelope),
      },
      canHandoff ? 200 : 409,
    );
  } catch {
    return json(
      {
        error: 'OROTITAN_CONTINUATION_RUNTIME_FAILURE',
        message: 'Pilotage continuation admission failed closed',
      },
      503,
    );
  }
}

export async function POST(request: Request): Promise<Response> {
  return handleContinuationRequest(request);
}
