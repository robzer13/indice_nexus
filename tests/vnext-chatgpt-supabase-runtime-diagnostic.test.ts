import assert from "node:assert/strict";
import test from "node:test";

import {
  GET,
  runSupabaseRuntimeDiagnostic,
} from "../app/api/vnext/chatgpt-supabase/diagnostic/route";

const EXPECTED_URL = "https://cugpgtzygqqlxetyeven.supabase.co";
const SECRET = "service-role-secret-that-must-never-leak";

function withVercelEnv(value: string | undefined, fn: () => Promise<void>): Promise<void> {
  const previous = process.env.VERCEL_ENV;
  if (value === undefined) delete process.env.VERCEL_ENV;
  else process.env.VERCEL_ENV = value;

  return fn().finally(() => {
    if (previous === undefined) delete process.env.VERCEL_ENV;
    else process.env.VERCEL_ENV = previous;
  });
}

test("Supabase diagnostic fails closed outside Vercel preview", async () => {
  await withVercelEnv("production", async () => {
    const response = await GET();
    assert.equal(response.status, 403);
    assert.equal(response.headers.get("cache-control"), "no-store");
    assert.deepEqual(await response.json(), {
      error: "OROTITAN_SUPABASE_DIAGNOSTIC_PREVIEW_ONLY",
    });
  });
});

test("Supabase diagnostic distinguishes URL failures without any network call", async () => {
  let networkCalled = false;
  const result = await runSupabaseRuntimeDiagnostic(
    {
      NEXT_PUBLIC_SUPABASE_URL: "https://wrong-project.supabase.co",
      SUPABASE_SERVICE_ROLE_KEY: SECRET,
    },
    {
      lookupHost: async () => {
        networkCalled = true;
        return { family: 4 };
      },
      probeTls: async () => {
        networkCalled = true;
        return { protocol: "TLSv1.3" };
      },
      fetchHttp: async () => {
        networkCalled = true;
        return new Response(null, { status: 200 });
      },
    },
  );

  assert.equal(result.classification, "URL_UNEXPECTED_HOST");
  assert.equal(result.url.status, "FAILED");
  assert.equal(networkCalled, false);
  assert.equal(JSON.stringify(result).includes(SECRET), false);
});

test("Supabase diagnostic classifies DNS failure deterministically and stops", async () => {
  let tlsCalled = false;
  let fetchCalled = false;
  const dnsError = Object.assign(new Error("dns lookup failed"), { code: "EAI_AGAIN" });

  const result = await runSupabaseRuntimeDiagnostic(
    {
      NEXT_PUBLIC_SUPABASE_URL: EXPECTED_URL,
      SUPABASE_SERVICE_ROLE_KEY: SECRET,
    },
    {
      lookupHost: async () => {
        throw dnsError;
      },
      probeTls: async () => {
        tlsCalled = true;
        return { protocol: "TLSv1.3" };
      },
      fetchHttp: async () => {
        fetchCalled = true;
        return new Response(null, { status: 200 });
      },
    },
  );

  assert.equal(result.classification, "DNS_FAILURE");
  assert.deepEqual(result.dns, { status: "FAILED", detail: "EAI_AGAIN" });
  assert.equal(result.tls.status, "NOT_RUN");
  assert.equal(tlsCalled, false);
  assert.equal(fetchCalled, false);
});

test("Supabase diagnostic distinguishes auth failure and never exposes the service role", async () => {
  let fetchCount = 0;
  const result = await runSupabaseRuntimeDiagnostic(
    {
      NEXT_PUBLIC_SUPABASE_URL: EXPECTED_URL,
      SUPABASE_SERVICE_ROLE_KEY: SECRET,
    },
    {
      lookupHost: async () => ({ family: 4 }),
      probeTls: async () => ({ protocol: "TLSv1.3" }),
      fetchHttp: async (_input, init) => {
        fetchCount += 1;
        if (fetchCount === 1) return new Response(null, { status: 401 });

        const headers = new Headers(init?.headers);
        assert.equal(headers.get("apikey"), SECRET);
        assert.equal(headers.get("authorization"), `Bearer ${SECRET}`);
        return new Response(null, { status: 401 });
      },
    },
  );

  assert.equal(result.classification, "AUTH_FAILURE");
  assert.equal(result.http_transport.status, "OK");
  assert.equal(result.auth_rest.status, "FAILED");
  assert.equal(result.auth_rest.http_status, 401);
  assert.equal(fetchCount, 2);
  assert.equal(JSON.stringify(result).includes(SECRET), false);
});
