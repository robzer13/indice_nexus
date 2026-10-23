/**
 * Candidate shared Research intake repository for the existing server-only
 * Supabase client. NOT imported by any live route or admin action.
 *
 * Production cloud writes are NOT authorized by this module. It requires an
 * independently reviewed schema and admin-only server adapter before wiring.
 */
import type { SupabaseClient } from '@supabase/supabase-js';
import type { EnqueueOutcome, ResearchInboxRequest } from './local-research-inbox-v1';

const TABLE = 'orotitan_research_intake_requests';
const SELECT = 'request_id,company_id,company_name,ticker,exchange,status,registry_run_id,created_at';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export class SharedResearchInboxError extends Error {
  constructor(readonly code: string) { super(code); }
}

function mapRow(raw: unknown): ResearchInboxRequest {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new SharedResearchInboxError('SHARED_INBOX_INVALID_RECEIPT');
  }
  const row = raw as Record<string, unknown>;
  if (typeof row.request_id !== 'string' || !UUID.test(row.request_id) ||
    typeof row.company_id !== 'string' || !UUID.test(row.company_id) ||
    typeof row.company_name !== 'string' || row.company_name.length < 2 ||
    typeof row.ticker !== 'string' || !row.ticker ||
    typeof row.exchange !== 'string' || !row.exchange ||
    typeof row.created_at !== 'string' ||
    !Number.isFinite(Date.parse(row.created_at)) ||
    row.status !== 'REQUESTED_NOT_EXECUTED' ||
    row.registry_run_id !== null) {
    throw new SharedResearchInboxError('SHARED_INBOX_INVALID_RECEIPT');
  }
  return {
    requestId: row.request_id, companyId: row.company_id,
    companyName: row.company_name, ticker: row.ticker,
    exchange: row.exchange, status: 'REQUESTED_NOT_EXECUTED',
    registryRunId: null, createdAt: row.created_at,
  };
}

export async function listSharedResearchInboxCandidate(
  client: SupabaseClient,
): Promise<ResearchInboxRequest[]> {
  const { data, error } = await client.from(TABLE)
    .select(SELECT).order('created_at', { ascending: false })
    .limit(500);
  if (error) throw new SharedResearchInboxError('SHARED_INBOX_READ_FAILED');
  return (data ?? []).map(mapRow);
}

/**
 * An authenticated admin server must supply its own privileged client.
 * Only company_id is sent. PostgreSQL independently loads company metadata,
 * checks eligibility and creates immutable snapshot + UUID.
 *
 * Unique-company constraint resolves duplicate submissions across processes.
 */
export async function enqueueSharedResearchCandidate(args: {
  client: SupabaseClient; companyId: string;
}): Promise<EnqueueOutcome> {
  if (!UUID.test(args.companyId)) {
    throw new SharedResearchInboxError('SHARED_INBOX_COMPANY_ID_INVALID');
  }

  const inserted = await args.client.from(TABLE)
    .insert({ company_id: args.companyId })
    .select(SELECT).single();
  if (!inserted.error && inserted.data) {
    return { request: mapRow(inserted.data), duplicate: false };
  }
  if (inserted.error?.code !== '23505') {
    throw new SharedResearchInboxError('SHARED_INBOX_WRITE_FAILED');
  }

  // Only a unique-constraint conflict is interpreted as an existing receipt.
  // Never mistake network/RLS/privilege/validation failures for idempotency.
  const existing = await args.client.from(TABLE)
    .select(SELECT).eq('company_id', args.companyId).maybeSingle();
  if (existing.error || !existing.data) {
    throw new SharedResearchInboxError('SHARED_INBOX_CONFLICT_UNRESOLVED');
  }
  return { request: mapRow(existing.data), duplicate: true };
}
