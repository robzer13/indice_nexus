import 'server-only';

import { createServerSupabaseClient } from '../../supabase/server';
import type {
  ContinuationAttemptPort,
  PersistedAttemptRegistration,
} from './pilotage-continuation-guard';

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
