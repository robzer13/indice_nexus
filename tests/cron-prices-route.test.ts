import assert from 'node:assert/strict';
import { register } from 'node:module';
import test from 'node:test';
import type { NextRequest } from 'next/server';

interface RouteRefreshResult {
  ok: boolean;
  failed: number;
  [key: string]: unknown;
}

interface RouteMocks {
  calls: string[];
  refresh: (trigger: string) => Promise<RouteRefreshResult>;
}

const routeState = globalThis as typeof globalThis & { __cronRouteMocks: RouteMocks };
routeState.__cronRouteMocks = {
  calls: [],
  refresh: async () => ({ ok: true, failed: 0 }),
};

const mockSources = new Map<string, string>([
  ['mock:market-refresh', `
    export function refreshMarketPrices(trigger) {
      globalThis.__cronRouteMocks.calls.push(trigger);
      return globalThis.__cronRouteMocks.refresh(trigger);
    }
  `],
]);

register('./module-mock-hooks.mjs', import.meta.url, {
  data: {
    resolutions: [['@/lib/market/refresh-prices', 'mock:market-refresh']],
    sources: [...mockSources],
    modulePrefixes: [new URL('../app/', import.meta.url).href],
  },
});

const routeModule = import('../app/api/cron/prices/route');
const originalCronSecret = process.env.CRON_SECRET;

function request(authorization?: string): NextRequest {
  return new Request('http://localhost/api/cron/prices', {
    headers: authorization ? { authorization } : undefined,
  }) as NextRequest;
}

test.after(() => {
  if (originalCronSecret === undefined) delete process.env.CRON_SECRET;
  else process.env.CRON_SECRET = originalCronSecret;
});

test.beforeEach(() => {
  process.env.CRON_SECRET = 'test-cron-secret';
  routeState.__cronRouteMocks.calls = [];
  routeState.__cronRouteMocks.refresh = async () => ({ ok: true, failed: 0 });
});

test('CRON route refuses access when CRON_SECRET is absent', async () => {
  const { GET } = await routeModule;
  delete process.env.CRON_SECRET;

  const response = await GET(request('Bearer test-cron-secret'));

  assert.equal(response.status, 401);
  assert.deepEqual(await response.json(), { ok: false, error: 'Unauthorized' });
  assert.deepEqual(routeState.__cronRouteMocks.calls, []);
});

test('CRON route rejects a missing or incorrect Authorization header', async () => {
  const { GET } = await routeModule;
  for (const authorization of [undefined, 'Basic test-cron-secret', 'Bearer wrong-secret']) {
    const response = await GET(request(authorization));
    assert.equal(response.status, 401);
    assert.deepEqual(await response.json(), { ok: false, error: 'Unauthorized' });
  }
  assert.deepEqual(routeState.__cronRouteMocks.calls, []);
});

test('CRON route emits non-secret diagnostics only for denied Vercel cron calls', async () => {
  const { GET } = await routeModule;
  const originalWarn = console.warn;
  const events: unknown[][] = [];
  console.warn = (...args: unknown[]) => { events.push(args); };

  try {
    const unauthenticated = new Request('http://localhost/api/cron/prices', {
      headers: { 'user-agent': 'vercel-cron/1.0' },
    }) as NextRequest;
    const rejected = new Request('http://localhost/api/cron/prices', {
      headers: { 'user-agent': 'vercel-cron/1.0', authorization: 'Bearer wrong-secret' },
    }) as NextRequest;

    assert.equal((await GET(unauthenticated)).status, 401);
    assert.equal((await GET(rejected)).status, 401);
    assert.deepEqual(events, [
      ['CRON_AUTH_DIAGNOSTIC', {
        secretConfigured: true,
        authorizationHeaderPresent: false,
        bearerPrefixPresent: false,
        secretHasInvalidHeaderChars: false,
      }],
      ['CRON_AUTH_DIAGNOSTIC', {
        secretConfigured: true,
        authorizationHeaderPresent: true,
        bearerPrefixPresent: true,
        secretHasInvalidHeaderChars: false,
      }],
    ]);
    assert.equal(JSON.stringify(events).includes('test-cron-secret'), false);
    assert.deepEqual(routeState.__cronRouteMocks.calls, []);
  } finally {
    console.warn = originalWarn;
  }
});

test('CRON route returns 200 for a successful authenticated synchronization', async () => {
  const { GET } = await routeModule;
  routeState.__cronRouteMocks.refresh = async () => ({
    ok: true,
    failed: 0,
    inserted: 2,
    companies: 2,
  });

  const response = await GET(request('Bearer test-cron-secret'));

  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { ok: true, failed: 0, inserted: 2, companies: 2 });
  assert.deepEqual(routeState.__cronRouteMocks.calls, ['CRON']);
});

test('CRON route returns 207 for an authenticated partial synchronization', async () => {
  const { GET } = await routeModule;
  routeState.__cronRouteMocks.refresh = async () => ({
    ok: false,
    failed: 1,
    inserted: 1,
    companies: 2,
  });

  const response = await GET(request('Bearer test-cron-secret'));

  assert.equal(response.status, 207);
  assert.deepEqual(await response.json(), { ok: false, failed: 1, inserted: 1, companies: 2 });
});

test('CRON route returns 500 when the refresh throws globally', async () => {
  const { GET } = await routeModule;
  routeState.__cronRouteMocks.refresh = async () => {
    throw new Error('COMPANY_LOAD_FAILED');
  };

  const response = await GET(request('Bearer test-cron-secret'));

  assert.equal(response.status, 500);
  assert.deepEqual(await response.json(), { ok: false, error: 'COMPANY_LOAD_FAILED' });
});
