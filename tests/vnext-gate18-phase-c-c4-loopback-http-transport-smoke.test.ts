import assert from "node:assert/strict";
import http from "node:http";
import test from "node:test";
import {
  LoopbackHttpExplicitTimeoutError,
  requestLoopbackJson,
} from "../runtime/vnext/loopback-http-json-client.js";

test("loopback HTTP client waits for delayed headers and obeys only explicit timeout", async () => {
  const server = http.createServer((request, response) => {
    const delay = request.url === "/slow" ? 250 : 60;
    setTimeout(() => {
      const body = JSON.stringify({ ok: true });
      response.writeHead(200, {
        "content-type": "application/json",
        "content-length": Buffer.byteLength(body),
      });
      response.end(body);
    }, delay);
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });

  try {
    const address = server.address();
    assert.notEqual(address, null);
    assert.equal(typeof address, "object");
    if (address === null || typeof address === "string") {
      throw new Error("TEST_SERVER_ADDRESS_UNAVAILABLE");
    }

    const base = `http://127.0.0.1:${address.port}`;
    const response = await requestLoopbackJson<{ ok: boolean }>(
      `${base}/ok`,
      { timeoutMs: 1000 },
    );
    assert.equal(response.ok, true);

    await assert.rejects(
      requestLoopbackJson<{ ok: boolean }>(
        `${base}/slow`,
        { timeoutMs: 50 },
      ),
      (error: unknown) =>
        error instanceof LoopbackHttpExplicitTimeoutError &&
        error.timeoutMs === 50,
    );
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test("loopback HTTP client rejects non-loopback endpoints before network access", async () => {
  await assert.rejects(
    requestLoopbackJson("https://example.com/", {
      timeoutMs: 1000,
    }),
    /LOOPBACK_HTTP_NON_LOOPBACK_ENDPOINT_FORBIDDEN/,
  );
});

test("transport smoke is non-inference and avoids global fetch", () => {
  const { readFileSync } = require("node:fs") as typeof import("node:fs");
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-loopback-http-transport-smoke.ts",
    "utf8",
  );
  assert.doesNotMatch(source, /\bfetch\s*\(/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.match(source, /externalNetworkAccessRequested: false/);
});
