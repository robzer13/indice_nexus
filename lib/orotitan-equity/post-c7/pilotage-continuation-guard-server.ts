import 'server-only';

import { createServerSupabaseClient } from '../../supabase/server';
import {
  buildLosslessResumeEnvelope,
  registerContinuationAttempt,
  type ContinuationAttemptPort,
  type LosslessResumeEnvelope,
  type PersistedAttemptRegistration,
  type PilotageRequestedOperation,
  type ContinuationGuardDecision,
} from './pilotage-continuation-guard';
import {
  executeServerControlledOperation,
} from './chatgpt-supabase-bridge-server';

type ServerClient = ReturnType<typeof createServerSupabaseClient>;

function databaseError(prefix: string, error: { message: string } | null): Error {
  return new Error(`${prefix}: ${error?.message ?? 'unknown Supabase error'}`);
}

function registration(data: unknown): PersistedAttemptRegistration {
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    throw new Error('register_orotitan_pilotage_attempt returned a non-object payload');
  }
  return data as PersistedAttemptRegistration;
}

export function createServerContinuationAttemptPort(
  client: ServerClient = createServerSupabaseClient(),
): ContinuationAttemptPort {
  return {
    async registerAttempt(args): Promise<PersistedAttemptRegistration> {
      const { data, error } = await client.rpc(
        'register_orotitan_pilotage_attempt',
        args,
      );
      if (error) {
        throw databaseError('register_orotitan_pilotage_attempt failed', error);
      }
      return registration(data);
    },
  };
}


export async function loadServerLosslessResumeEnvelope(input: {
  issuerQuery: string;
  runId: string;
}): Promise<LosslessResumeEnvelope> {
  const result = await executeServerControlledOperation({
    contract_version: '0.1.0',
    operation: 'LOAD',
    issuer_query: input.issuerQuery,
    run_id: input.runId,
    requested_context_tiers: ['L0', 'L1', 'L2', 'L3'],
  });

  if (result.operation !== 'LOAD_RESULT') {
    throw new Error(
      'lossless resume LOAD failed: ' +
        ('message' in result ? result.message : result.operation),
    );
  }

  return buildLosslessResumeEnvelope(result);
}

export async function registerServerContinuationAttempt(input: {
  envelope: LosslessResumeEnvelope;
  requestedOperation: PilotageRequestedOperation;
}): Promise<ContinuationGuardDecision> {
  return registerContinuationAttempt(
    createServerContinuationAttemptPort(),
    input.envelope,
    input.requestedOperation,
  );
}
