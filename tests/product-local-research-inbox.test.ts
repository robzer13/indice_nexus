import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import {
  enqueueLocalResearchRequest, listLocalResearchRequests, ResearchInboxError,
  type ResearchInboxCompany,
} from '../lib/orotitan-equity/product/local-research-inbox-v1';

const companyId = 'bbcf36af-b4e7-4bca-a4d2-164072e968fa';
const secondId = 'c81a603d-96ee-49da-8a8c-5247a2316d9c';
const companies: ResearchInboxCompany[] = [
  { id: companyId, name: 'Synthetic Research A', ticker: 'SYNTA', exchange: 'XPAR', active: true },
  { id: secondId, name: 'Synthetic Research B', ticker: 'SYNTB', exchange: 'XAMS', active: true },
];

async function fixture(work: (directory: string) => Promise<void>) {
  const directory = await mkdtemp(join(tmpdir(), 'orotitan-product-inbox-'));
  try { await work(directory); }
  finally { await rm(directory, { recursive: true, force: true }); }
}

test('Product MVP: admin-selected Screener company creates a persisted, non-executed request', async () =>
  fixture(async directory => {
    const empty = await listLocalResearchRequests(directory);
    assert.deepEqual(empty, []);
    const one = await enqueueLocalResearchRequest({ directory, companyId, companies });
    assert.equal(one.duplicate, false);
    assert.equal(one.request.companyId, companyId);
    assert.equal(one.request.status, 'REQUESTED_NOT_EXECUTED');
    assert.equal(one.request.registryRunId, null);
    const persisted = await listLocalResearchRequests(directory);
    assert.equal(persisted.length, 1);
    assert.deepEqual(persisted[0], one.request);
    const file = await readFile(join(directory, 'research-requests-v1.json'), 'utf8');
    assert.ok(file.includes(one.request.requestId));
    assert.ok(!file.includes('registry_run_id'));
  }));

test('Product MVP: 12 concurrent confirmations create one request, with stable idempotent receipt', async () =>
  fixture(async directory => {
    const results = await Promise.all(Array.from({ length: 12 }, () =>
      enqueueLocalResearchRequest({ directory, companyId, companies })));
    assert.equal(results.filter(r => !r.duplicate).length, 1);
    assert.equal(new Set(results.map(r => r.request.requestId)).size, 1);
    assert.equal((await listLocalResearchRequests(directory)).length, 1);
  }));

test('Product MVP: separate tracked companies have separate persisted requests', async () =>
  fixture(async directory => {
    const first = await enqueueLocalResearchRequest({ directory, companyId, companies });
    const second = await enqueueLocalResearchRequest({ directory, companyId: secondId, companies });
    assert.notEqual(first.request.requestId, second.request.requestId);
    assert.equal((await listLocalResearchRequests(directory)).length, 2);
  }));

test('Product MVP: refuses unknown/inactive/non-UUID company IDs, never invents issuer mapping', async () =>
  fixture(async directory => {
    const inactive = companies.map(c => c.id === companyId ? { ...c, active: false } : c);
    for (const args of [
      { companyId: 'not-a-uuid', companies },
      { companyId, companies: inactive },
      { companyId: '4658050f-05eb-4955-b889-0c4ab9f14ed1', companies },
    ]) {
      await assert.rejects(
        enqueueLocalResearchRequest({ directory, ...args }),
        (error: unknown) => error instanceof ResearchInboxError &&
          error.code.startsWith('INBOX_COMPANY'),
      );
    }
    assert.deepEqual(await listLocalResearchRequests(directory), []);
  }));

test('Product MVP: corrupted persistence fails closed instead of silently resetting history', async () =>
  fixture(async directory => {
    await writeFile(join(directory, 'research-requests-v1.json'), '{not-json', 'utf8');
    await assert.rejects(listLocalResearchRequests(directory),
      (error: unknown) => error instanceof ResearchInboxError &&
        error.code === 'INBOX_FILE_INVALID');
    await assert.rejects(enqueueLocalResearchRequest({ directory, companyId, companies }),
      (error: unknown) => error instanceof ResearchInboxError &&
        error.code === 'INBOX_FILE_INVALID');
    assert.equal(await readFile(join(directory, 'research-requests-v1.json'), 'utf8'), '{not-json');
  }));
