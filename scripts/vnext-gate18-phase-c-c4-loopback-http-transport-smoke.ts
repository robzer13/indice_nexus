import http from "node:http";
import {
  LoopbackHttpExplicitTimeoutError,
  requestLoopbackJson,
} from "../runtime/vnext/loopback-http-json-client.js";

interface SmokeResponse {
  ok: boolean;
  delayedMs: number;
}

const SUCCESS_DELAY_MS = 1500;
const SUCCESS_TIMEOUT_MS = 5000;
const EXPLICIT_TIMEOUT_DELAY_MS = 1500;
const EXPLICIT_TIMEOUT_MS = 250;

async function main(): Promise<void> {
  const server = http.createServer((request, response) => {
    if (request.url === "/delayed-success") {
      setTimeout(() => {
        const body = JSON.stringify({
          ok: true,
          delayedMs: SUCCESS_DELAY_MS,
        });
        response.writeHead(200, {
          "content-type": "application/json",
          "content-length": Buffer.byteLength(body),
        });
        response.end(body);
      }, SUCCESS_DELAY_MS);
      return;
    }

    if (request.url === "/delayed-timeout") {
      setTimeout(() => {
        const body = JSON.stringify({
          ok: true,
          delayedMs: EXPLICIT_TIMEOUT_DELAY_MS,
        });
        response.writeHead(200, {
          "content-type": "application/json",
          "content-length": Buffer.byteLength(body),
        });
        response.end(body);
      }, EXPLICIT_TIMEOUT_DELAY_MS);
      return;
    }

    response.writeHead(404);
    response.end();
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve());
  });

  try {
    const address = server.address();
    if (address === null || typeof address === "string") {
      throw new Error("LOOPBACK_SMOKE_SERVER_ADDRESS_UNAVAILABLE");
    }

    const base = `http://127.0.0.1:${address.port}`;

    const successStartedAt = Date.now();
    const success =
      await requestLoopbackJson<SmokeResponse>(
        `${base}/delayed-success`,
        { timeoutMs: SUCCESS_TIMEOUT_MS },
      );
    const successWallClockMs = Date.now() - successStartedAt;

    let timeoutObserved = false;
    let timeoutError: string | null = null;
    const timeoutStartedAt = Date.now();

    try {
      await requestLoopbackJson<SmokeResponse>(
        `${base}/delayed-timeout`,
        { timeoutMs: EXPLICIT_TIMEOUT_MS },
      );
    } catch (error) {
      timeoutError =
        error instanceof Error ? error.message : String(error);
      timeoutObserved =
        error instanceof LoopbackHttpExplicitTimeoutError;
    }

    const timeoutWallClockMs = Date.now() - timeoutStartedAt;

    const pass =
      success.ok === true &&
      success.delayedMs === SUCCESS_DELAY_MS &&
      successWallClockMs >= SUCCESS_DELAY_MS &&
      successWallClockMs < SUCCESS_TIMEOUT_MS &&
      timeoutObserved &&
      timeoutWallClockMs >= EXPLICIT_TIMEOUT_MS &&
      timeoutWallClockMs < EXPLICIT_TIMEOUT_DELAY_MS;

    console.log(JSON.stringify({
      format:
        "OROTITAN_GATE18_PHASE_C_C4_LOOPBACK_HTTP_TRANSPORT_SMOKE_V0.1",
      gate: 18,
      phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
      stage: "C4_RUNTIME_TRANSPORT_REMEDIATION",
      status: pass ? "PASS" : "FAIL",
      mode: "LOCAL_READ_ONLY_NO_INFERENCE",
      node: {
        version: process.version,
        undiciVersion:
          typeof process.versions.undici === "string"
            ? process.versions.undici
            : null,
      },
      successProbe: {
        delayedHeadersMs: SUCCESS_DELAY_MS,
        explicitTimeoutMs: SUCCESS_TIMEOUT_MS,
        wallClockMs: successWallClockMs,
        response: success,
      },
      explicitTimeoutProbe: {
        delayedHeadersMs: EXPLICIT_TIMEOUT_DELAY_MS,
        explicitTimeoutMs: EXPLICIT_TIMEOUT_MS,
        wallClockMs: timeoutWallClockMs,
        timeoutObserved,
        timeoutError,
      },
      transport: {
        implementation: "node:http.request",
        globalFetchUsed: false,
        undiciDispatcherUsed: false,
        loopbackOnly: true,
        hiddenHeadersTimeoutConfigured: false,
        explicitAbortTimerOnly: true,
      },
      authority: {
        inferenceAuthorized: false,
        retryAuthorized: false,
        modelLoadAuthorized: false,
        parameterChangeAuthorized: false,
      },
      safety: {
        externalNetworkAccessRequested: false,
        ollamaApiCalled: false,
        modelInferenceExecuted: false,
        modelLoadRequested: false,
        modelSwitchExecuted: false,
        productionMutation: false,
        publicationAuthority: false,
      },
    }, null, 2));
  } finally {
    await new Promise<void>((resolve) => {
      server.close(() => resolve());
    });
  }
}

void main();
