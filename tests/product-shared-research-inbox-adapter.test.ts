import assert from 'node:assert/strict';
import test from 'node:test';
import type { SupabaseClient } from '@supabase/supabase-js';
import {
  enqueueSharedResearchCandidate, listSharedResearchInboxCandidate,
  SharedResearchInboxError,
} from '../lib/orotitan-equity/product/shared-research-inbox-candidate-v1';

const companyId = '9c6c18b1-708a-4c44-960b-b122bca24b55';
const requestId = '27b098b6-7196-4659-9c27-d6e3c0dd0d0a';
const row = {
  request_id: requestId, company_id: companyId,
  company_name: 'Synthetic Corp', ticker: 'SYN', exchange: 'XPAR',
  status: 'REQUESTED_NOT_EXECUTED',
  registry_run_id: null, created_at: '2026-10-23T10:00:00.000Z',
};
type QueryResult = { data: unknown; error: { code: string } | null };

function mockClient(options: {
  insert: QueryResult; existing?: QueryResult; listing?: QueryResult;
}) {
  const insertedValues: unknown[] = [];
  const from = (name: string) => {
    assert.equal(name, 'orotitan_research_intake_requests');
    return {
      insert: (payload: unknown) => {
        insertedValues.push(payload);
        return { select: () => ({ single: async () => options.insert }) };
      },
      select: () => ({
        eq: (key: string, value: string) => {
          assert.equal(key, 'company_id');
          assert.equal(value, companyId);
          return { maybeSingle: async () => options.existing ?? { data: null, error: null } };
        },
        order: () => ({
          limit: async () => options.listing ?? { data: [row], error: null },
        }),
      }),
    };
  };
  return { client: { from } as unknown as SupabaseClient, insertedValues };
}

test('Shared product adapter inserts company ID only and returns DB receipt', async () => {
  const { client, insertedValues } = mockClient({ insert: { data: row, error: null } });
  const result = await enqueueSharedResearchCandidate({ client, companyId });
  assert.equal(result.duplicate, false);
  assert.equal(result.request.requestId, requestId);
  assert.equal(result.request.registryRunId, null);
  assert.deepEqual(insertedValues, [{ company_id: companyId }]);
});

test('Shared product adapter resolves only PG unique-conflict 23505 as duplicate', async () => {
  const { client } = mockClient({
    insert: { data: null, error: { code: '23505' } },
    existing: { data: row, error: null },
  });
  const result = await enqueueSharedResearchCandidate({ client, companyId });
  assert.equal(result.duplicate, true);
  assert.equal(result.request.requestId, requestId);
});

test('Shared product adapter fails closed on authorization errors and missing conflict receipt', async () => {
  for (const input of [
    { insert: { data: null, error: { code: '42501' } } },
    { insert: { data: null, error: { code: '23505' } } },
  ]) {
    const { client } = mockClient(input);
    await assert.rejects(enqueueSharedResearchCandidate({ client, companyId }),
      (err: unknown) => err instanceof SharedResearchInboxError &&
        (err.code === 'SHARED_INBOX_WRITE_FAILED' ||
         err.code === 'SHARED_INBOX_CONFLICT_UNRESOLVED'));
  }
});

test('Shared product adapter refuses forged IDs and invented Registry states', async () => {
  const { client } = mockClient({ insert: { data: row, error: null } });
  await assert.rejects(
    enqueueSharedResearchCandidate({ client, companyId: 'invalid' }),
    (err: unknown) => err instanceof SharedResearchInboxError &&
      err.code === 'SHARED_INBOX_COMPANY_ID_INVALID');
  const corrupt = mockClient({
    insert: { data: { ...row, status: 'RESEARCH_RUNNING' }, error: null },
  });
  await assert.rejects(
    enqueueSharedResearchCandidate({ client: corrupt.client, companyId }),
    (err: unknown) => err instanceof SharedResearchInboxError &&
      err.code === 'SHARED_INBOX_INVALID_RECEIPT');
});

test('Shared product adapter reads persisted snapshots in request order', async () => {
  const { client } = mockClient({
    insert: { data: row, error: null },
    listing: { data: [row], error: null },
  });
  const results = await listSharedResearchInboxCandidate(client);
  assert.equal(results.length, 1);
  assert.equal(results[0].requestId, requestId);
  assert.equal(results[0].status, 'REQUESTED_NOT_EXECUTED');
});
