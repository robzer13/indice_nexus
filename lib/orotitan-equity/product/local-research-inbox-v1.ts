/**
 * Local-only Research intake inbox. Product pilot, NOT the OroTitan Registry.
 * No Supabase / cloud / Research execution code in this module.
 * Storage is intentionally injected for isolated unit tests.
 */
import { randomUUID } from 'node:crypto';
import { mkdir, open, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

export type ResearchInboxCompany = Readonly<{
  id: string; name: string; ticker: string; exchange: string; active: boolean;
}>;
export type ResearchInboxRequest = Readonly<{
  requestId: string;
  companyId: string;
  companyName: string;
  ticker: string;
  exchange: string;
  createdAt: string;
  status: 'REQUESTED_NOT_EXECUTED';
  registryRunId: null;
}>;
export type EnqueueOutcome = Readonly<{
  request: ResearchInboxRequest; duplicate: boolean;
}>;
type LocalFile = { version: 1; requests: ResearchInboxRequest[] };

const FILE_NAME = 'research-requests-v1.json';
const LOCK_NAME = 'research-requests-v1.lock';
const MAX_ROWS = 5000;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export class ResearchInboxError extends Error {
  constructor(readonly code: string) { super(code); }
}

function emptyFile(): LocalFile { return { version: 1, requests: [] }; }

function validateFile(data: unknown): LocalFile {
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    throw new ResearchInboxError('INBOX_FILE_INVALID');
  }
  const value = data as Record<string, unknown>;
  if (value.version !== 1 || !Array.isArray(value.requests) ||
    value.requests.length > MAX_ROWS || Object.keys(value).length !== 2) {
    throw new ResearchInboxError('INBOX_FILE_INVALID');
  }
  const requests = value.requests as unknown[];
  const ids = new Set<string>();
  const companies = new Set<string>();
  for (const item of requests) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      throw new ResearchInboxError('INBOX_FILE_INVALID');
    }
    const r = item as Record<string, unknown>;
    if (Object.keys(r).length !== 8 ||
      typeof r.requestId !== 'string' || !UUID.test(r.requestId) ||
      typeof r.companyId !== 'string' || !UUID.test(r.companyId) ||
      typeof r.companyName !== 'string' || r.companyName.length < 2 ||
      typeof r.ticker !== 'string' || !r.ticker ||
      typeof r.exchange !== 'string' || !r.exchange ||
      typeof r.createdAt !== 'string' ||
      !Number.isFinite(Date.parse(r.createdAt)) ||
      r.status !== 'REQUESTED_NOT_EXECUTED' || r.registryRunId !== null ||
      ids.has(r.requestId) || companies.has(r.companyId)) {
      throw new ResearchInboxError('INBOX_FILE_INVALID');
    }
    ids.add(r.requestId);
    companies.add(r.companyId);
  }
  return value as LocalFile;
}

async function readStore(directory: string): Promise<LocalFile> {
  try {
    const text = await readFile(join(directory, FILE_NAME), 'utf8');
    let parsed: unknown;
    try { parsed = JSON.parse(text); }
    catch { throw new ResearchInboxError('INBOX_FILE_INVALID'); }
    return validateFile(parsed);
  } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') return emptyFile();
    throw error;
  }
}

export async function listLocalResearchRequests(directory: string):
  Promise<ResearchInboxRequest[]> {
  const file = await readStore(directory);
  return [...file.requests].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

async function withExclusiveLock<T>(directory: string, operation: () => Promise<T>):
  Promise<T> {
  await mkdir(directory, { recursive: true, mode: 0o700 });
  const lockPath = join(directory, LOCK_NAME);
  for (let attempt = 0; attempt < 60; attempt++) {
    try {
      const handle = await open(lockPath, 'wx', 0o600);
      try { return await operation(); }
      finally { await handle.close(); await unlink(lockPath); }
    } catch (error) {
      if ((error as NodeJS.ErrnoException)?.code !== 'EEXIST') throw error;
      await new Promise(resolve => setTimeout(resolve, 25));
    }
  }
  throw new ResearchInboxError('INBOX_BUSY');
}

export async function enqueueLocalResearchRequest(args: {
  directory: string; companyId: string; companies: readonly ResearchInboxCompany[];
  now?: Date;
}): Promise<EnqueueOutcome> {
  if (!UUID.test(args.companyId)) {
    throw new ResearchInboxError('INBOX_COMPANY_ID_INVALID');
  }
  const company = args.companies.find(c => c.id === args.companyId && c.active);
  if (!company || !company.name || !company.ticker || !company.exchange) {
    throw new ResearchInboxError('INBOX_COMPANY_NOT_ELIGIBLE');
  }
  const now = args.now ?? new Date();
  if (!Number.isFinite(now.getTime())) throw new ResearchInboxError('INBOX_TIME_INVALID');
  return withExclusiveLock(args.directory, async () => {
    const file = await readStore(args.directory);
    const previous = file.requests.find(r => r.companyId === company.id);
    if (previous) return { request: previous, duplicate: true };
    if (file.requests.length >= MAX_ROWS) throw new ResearchInboxError('INBOX_FULL');

    const request: ResearchInboxRequest = {
      requestId: randomUUID(), companyId: company.id,
      companyName: company.name, ticker: company.ticker,
      exchange: company.exchange, createdAt: now.toISOString(),
      status: 'REQUESTED_NOT_EXECUTED', registryRunId: null,
    };
    const next: LocalFile = { version: 1, requests: [...file.requests, request] };
    const temporary = join(args.directory, FILE_NAME + '.' + randomUUID() + '.tmp');
    try {
      await writeFile(temporary, JSON.stringify(next, null, 2) + '\n', {
        encoding: 'utf8', mode: 0o600, flag: 'wx',
      });
      await rename(temporary, join(args.directory, FILE_NAME));
    } catch (error) {
      await unlink(temporary).catch(() => {});
      throw error;
    }
    return { request, duplicate: false };
  });
}
