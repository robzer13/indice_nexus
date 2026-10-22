/**
 * P0 resilience for Supabase's intermittent PostgREST PGRST303 error.
 *
 * Scope is deliberately narrow: an idempotent GET/HEAD to our exact
 * configured Supabase REST API, rejected as 401/PGRST303 specifically for
 * "JWT issued at future". Never retry writes, Auth, RPC POSTs or unrelated
 * 401s. Responses are never replaced with empty/made-up data.
 *
 * This is a temporary mitigation, NOT a fix for Supabase-managed clock drift.
 */
const RETRY_DELAYS_MS = [400, 1600] as const;
type RequestInput = Parameters<typeof fetch>[0];
type FetchInit = Parameters<typeof fetch>[1];
type Sleep = (ms: number) => Promise<void>;

const defaultSleep: Sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function shouldInspect(input: RequestInput, init: FetchInit, expectedOrigin: string): boolean {
  const method = (init?.method ?? (input instanceof Request ? input.method : 'GET')).toUpperCase();
  if (method !== 'GET' && method !== 'HEAD') return false;
  const urlValue = input instanceof Request ? input.url : String(input);
  try {
    const url = new URL(urlValue);
    return url.origin === expectedOrigin && url.pathname.startsWith('/rest/v1/');
  } catch {
    return false;
  }
}

async function isPostgrestClockSkew(response: Response): Promise<boolean> {
  if (response.status !== 401) return false;
  // A PostgREST error is small JSON. Avoid reading unbounded error bodies.
  const length = Number(response.headers.get('content-length'));
  if (!Number.isFinite(length) || length > 4096) return false;
  if (!response.headers.get('content-type')?.toLowerCase().includes('application/json')) return false;
  try {
    const data: unknown = await response.clone().json();
    if (!data || typeof data !== 'object' || Array.isArray(data)) return false;
    const record = data as Record<string, unknown>;
    return record.code === 'PGRST303' &&
      typeof record.message === 'string' &&
      record.message.includes('JWT issued at future');
  } catch {
    return false;
  }
}

export function pgrstClockSkewRetryFetch(
  baseFetch: typeof fetch,
  supabaseUrl: string,
  sleep: Sleep = defaultSleep,
): typeof fetch {
  const origin = new URL(supabaseUrl).origin;
  return async (input, init) => {
    const allowRetry = shouldInspect(input, init, origin);
    for (let attempt = 0; ; attempt++) {
      const response = await baseFetch(input, init);
      if (!allowRetry || attempt >= RETRY_DELAYS_MS.length ||
        !(await isPostgrestClockSkew(response))) return response;
      // Do not retry an aborted request or leak authorization headers.
      const signal = init?.signal ?? (input instanceof Request ? input.signal : null);
      if (signal?.aborted) return response;
      console.warn('P0_SUPABASE_POSTGREST_PGRST303_READ_RETRY', {attempt: attempt + 1});
      await sleep(RETRY_DELAYS_MS[attempt]);
      if (signal?.aborted) return response;
    }
  };
}
