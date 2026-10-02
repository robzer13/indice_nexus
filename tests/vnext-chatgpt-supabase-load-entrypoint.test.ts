import assert from "node:assert/strict";
import test from "node:test";

import {
  GET,
  parseLoadRequestFromUrl,
} from "../app/api/vnext/chatgpt-supabase/load/route";

const RUN_ID = "a4cf9002-b52d-4440-8902-adc70bd777dd";

function withVercelEnv(value: string | undefined, fn: () => Promise<void>): Promise<void> {
  const previous = process.env.VERCEL_ENV;
  if (value === undefined) delete process.env.VERCEL_ENV;
  else process.env.VERCEL_ENV = value;

  return fn().finally(() => {
    if (previous === undefined) delete process.env.VERCEL_ENV;
    else process.env.VERCEL_ENV = previous;
  });
}

test("LOAD entrypoint constructs only the frozen LOAD operation", () => {
  const request = parseLoadRequestFromUrl(
    new URL(
      `https://preview.example/api/vnext/chatgpt-supabase/load?issuer_query=Veolia&run_id=${RUN_ID}&tiers=L0,L1`,
    ),
  );

  assert.deepEqual(request, {
    contract_version: "0.1.0",
    operation: "LOAD",
    issuer_query: "Veolia",
    run_id: RUN_ID,
    requested_context_tiers: ["L0", "L1"],
  });
});

test("LOAD entrypoint rejects operation smuggling and malformed bounded inputs", () => {
  assert.throws(
    () =>
      parseLoadRequestFromUrl(
        new URL(
          `https://preview.example/api/vnext/chatgpt-supabase/load?issuer_query=Veolia&operation=CHECKPOINT_STAGE`,
        ),
      ),
    /unsupported query parameter: operation/,
  );

  assert.throws(
    () =>
      parseLoadRequestFromUrl(
        new URL(
          "https://preview.example/api/vnext/chatgpt-supabase/load?issuer_query=Veolia&tiers=L0,L0",
        ),
      ),
    /tiers must not contain duplicates/,
  );

  assert.throws(
    () =>
      parseLoadRequestFromUrl(
        new URL(
          "https://preview.example/api/vnext/chatgpt-supabase/load?issuer_query=Veolia&tiers=L4",
        ),
      ),
    /unsupported context tier: L4/,
  );

  assert.throws(
    () =>
      parseLoadRequestFromUrl(
        new URL(
          "https://preview.example/api/vnext/chatgpt-supabase/load?issuer_query=Veolia&run_id=not-a-uuid",
        ),
      ),
    /run_id must be a UUID/,
  );
});

test("LOAD entrypoint fails closed outside Vercel preview", async () => {
  await withVercelEnv("production", async () => {
    const response = await GET(
      new Request(
        `https://production.example/api/vnext/chatgpt-supabase/load?issuer_query=Veolia&run_id=${RUN_ID}`,
        {
          headers: {
            "x-vercel-trusted-oidc-idp-token": "opaque-test-token",
          },
        },
      ),
    );

    assert.equal(response.status, 403);
    assert.equal(response.headers.get("cache-control"), "no-store");
    assert.deepEqual(await response.json(), {
      error: "OROTITAN_BRIDGE_LOAD_PREVIEW_ONLY",
    });
  });
});

test("LOAD entrypoint requires the Trusted Sources OIDC header in preview", async () => {
  await withVercelEnv("preview", async () => {
    const response = await GET(
      new Request(
        `https://preview.example/api/vnext/chatgpt-supabase/load?issuer_query=Veolia&run_id=${RUN_ID}`,
      ),
    );

    assert.equal(response.status, 401);
    assert.deepEqual(await response.json(), {
      error: "OROTITAN_BRIDGE_LOAD_TRUSTED_OIDC_REQUIRED",
    });
  });
});

test("LOAD entrypoint sanitizes server initialization failures as INFRASTRUCTURE", async () => {
  const previousUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const previousServiceRole = process.env.SUPABASE_SERVICE_ROLE_KEY;
  delete process.env.NEXT_PUBLIC_SUPABASE_URL;
  delete process.env.SUPABASE_SERVICE_ROLE_KEY;

  try {
    await withVercelEnv("preview", async () => {
      const response = await GET(
        new Request(
          `https://preview.example/api/vnext/chatgpt-supabase/load?issuer_query=Veolia&run_id=${RUN_ID}`,
          {
            headers: {
              "x-vercel-trusted-oidc-idp-token": "opaque-test-token",
            },
          },
        ),
      );

      assert.equal(response.status, 503);
      assert.deepEqual(await response.json(), {
        contract_version: "0.1.0",
        operation: "OPERATION_FAILURE",
        error_class: "INFRASTRUCTURE",
        message: "LOAD bridge server initialization failed",
        retry_without_reload_allowed: false,
      });
    });
  } finally {
    if (previousUrl === undefined) delete process.env.NEXT_PUBLIC_SUPABASE_URL;
    else process.env.NEXT_PUBLIC_SUPABASE_URL = previousUrl;

    if (previousServiceRole === undefined) delete process.env.SUPABASE_SERVICE_ROLE_KEY;
    else process.env.SUPABASE_SERVICE_ROLE_KEY = previousServiceRole;
  }
});
