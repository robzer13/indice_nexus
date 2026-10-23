/**
 * Test-only minimal PostgREST-shaped HTTP fixture against disposable PG17.
 * Does NOT implement real Supabase/PostgREST authentication, tenancy, RLS,
 * JWT handling or SDK compatibility beyond the exact exercised requests.
 * Never exposed outside the GitHub Actions loopback runner.
 */
import http from 'node:http';
import { execFileSync } from 'node:child_process';

const db = process.argv[2];
const PORT = 48761;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const API_KEY = process.env.OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY;
if (!process.env.CI || process.env.OROTITAN_PRODUCT_BROWSER_FIXTURE !== '1' ||
  !/^orotitan_product_browser_[0-9]+$/.test(db ?? '') ||
  !['127.0.0.1', 'localhost'].includes(process.env.PGHOST ?? '127.0.0.1') ||
  (process.env.PGPORT ?? '5432') !== '5432' ||
  (process.env.PGUSER ?? 'postgres') !== 'postgres' ||
  !API_KEY || API_KEY.length < 40) {
  throw new Error('PRODUCT_BROWSER_FIXTURE_NOT_ALLOWED');
}
const fields = "json_build_object('request_id',r.request_id::text," +
  "'company_id',r.company_id::text,'company_name',r.company_name," +
  "'ticker',r.ticker,'exchange',r.exchange,'status',r.status," +
  "'registry_run_id',r.registry_run_id,'created_at',r.created_at)";
function query(sql) {
  return execFileSync('psql', [
    '-X', '-q', '-A', '-t', '-v', 'ON_ERROR_STOP=1', '-d', db, '-c', sql,
  ], {
    encoding: 'utf8', timeout: 8000, stdio: ['ignore', 'pipe', 'pipe'],
  }).trim();
}
function quotedUuid(id) {
  if (!UUID.test(id)) throw new Error('INVALID_COMPANY_UUID');
  return "'" + id + "'::uuid";
}
function emit(res, status, payload, extra = {}) {
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store', ...extra,
  });
  res.end(JSON.stringify(payload));
}
function rows() {
  const text = query("SET ROLE service_role; SELECT coalesce(json_agg(" + fields +
    " ORDER BY r.created_at DESC),'[]'::json)::text FROM " +
    "public.orotitan_research_intake_requests r; RESET ROLE;");
  return JSON.parse(text || '[]');
}
function companies() {
  const text = query("SET ROLE service_role; SELECT coalesce(json_agg(" +
    "json_build_object('id',id::text,'name',name,'ticker',ticker," +
    "'exchange',exchange,'active',active) ORDER BY name),'[]'::json)::text " +
    "FROM public.companies; RESET ROLE;");
  return JSON.parse(text || '[]');
}
function pathOf(req) {
  return new URL(req.url || '/', 'http://127.0.0.1:' + PORT);
}
const server = http.createServer(async (req, res) => {
  try {
    if (req.headers.apikey !== API_KEY) {
      emit(res, 403, { code: '42501', message: 'FIXTURE_KEY_REQUIRED' });
      return;
    }
    const url = pathOf(req);
    if (url.pathname === '/rest/v1/companies' && req.method === 'GET') {
      emit(res, 200, companies());
      return;
    }
    if ((url.pathname === '/rest/v1/latest_company_state' ||
      url.pathname === '/rest/v1/market_sync_runs' ||
      url.pathname === '/rest/v1/research_dossiers') && req.method === 'GET') {
      emit(res, 200, []);
      return;
    }
    if (url.pathname !== '/rest/v1/orotitan_research_intake_requests') {
      emit(res, 404, { code: 'PGRST404', message: 'Fixture route not implemented' });
      return;
    }
    if (req.method === 'GET') {
      const filtered = url.searchParams.get('company_id');
      const all = rows();
      if (filtered) {
        const id = filtered.match(/^eq\.([0-9a-f-]{36})$/)?.[1];
        if (!id || !UUID.test(id)) {
          emit(res, 400, { code: '22P02', message: 'Invalid company filter' });
          return;
        }
        const row = all.find(r => r.company_id === id);
        if (!row) {
          emit(res, 406, { code: 'PGRST116', message: 'No rows found' });
          return;
        }
        emit(res, 200, row);
      } else {
        emit(res, 200, all.slice(0, 500));
      }
      return;
    }
    if (req.method !== 'POST') {
      emit(res, 405, { code: '405', message: 'Read/insert only' });
      return;
    }
    let body = '';
    for await (const chunk of req) {
      body += chunk;
      if (body.length > 4096) throw new Error('PAYLOAD_TOO_LARGE');
    }
    const input = JSON.parse(body);
    if (!input || Array.isArray(input) ||
      Object.keys(input).length !== 1 ||
      typeof input.company_id !== 'string' ||
      !UUID.test(input.company_id)) {
      emit(res, 400, { code: '22P02', message: 'Only company_id accepted' });
      return;
    }
    try {
      const text = query("SET ROLE service_role; INSERT INTO " +
        "public.orotitan_research_intake_requests(company_id) VALUES (" +
        quotedUuid(input.company_id) + ") RETURNING " +
        "json_build_object('request_id',request_id::text," +
        "'company_id',company_id::text,'company_name',company_name," +
        "'ticker',ticker,'exchange',exchange,'status',status," +
        "'registry_run_id',registry_run_id,'created_at',created_at)::text; RESET ROLE;");
      emit(res, 201, JSON.parse(text));
    } catch (err) {
      const stderr = String(err?.stderr ?? '');
      if (stderr.includes('duplicate key') ||
        stderr.includes('unique constraint')) {
        emit(res, 409, { code: '23505', message: 'Unique request already exists' });
        return;
      }
      emit(res, 400, { code: 'PGRST400', message: 'Request rejected in fixture DB' });
    }
  } catch {
    emit(res, 500, { code: 'PGRST500', message: 'Fixture operation failed' });
  }
});
server.listen(PORT, '127.0.0.1', () =>
  process.stdout.write('PRODUCT_BROWSER_REST_READY\n'));
