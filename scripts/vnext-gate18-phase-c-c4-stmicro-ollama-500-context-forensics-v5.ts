import { readFileSync, statSync } from "node:fs";
import path from "node:path";

const ROOT = process.env.LOCALAPPDATA ?? "";
const SERVER_LOG = path.join(ROOT, "Ollama", "server.log");
const TARGET_LOCAL_STAMP = "2026/09/24 - 22:08:47";
const TARGET_ROUTE = 'POST     "/api/generate"';
const BEFORE = 180;
const AFTER = 80;

function clip(value: string, max = 2200): string {
  return value.length > max ? value.slice(0, max) : value;
}

let raw: string;
let stat: ReturnType<typeof statSync>;

try {
  raw = readFileSync(SERVER_LOG, "utf8");
  stat = statSync(SERVER_LOG);
} catch (error) {
  console.log(JSON.stringify({
    format:
      "OROTITAN_GATE18_PHASE_C_C4_STMICRO_OLLAMA_500_CONTEXT_FORENSICS_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C4_RUNTIME_FAILURE_FORENSICS",
    status: "BLOCKED_SERVER_LOG_READ_FAILED",
    mode: "LOCAL_READ_ONLY_NO_INFERENCE",
    serverLogPath: SERVER_LOG,
    error: error instanceof Error ? error.message : String(error),
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
  process.exit(0);
}

const lines = raw.split(/\r?\n/);
const exactIndexes = lines
  .map((line, index) => ({ line, index }))
  .filter(({ line }) =>
    line.includes(TARGET_LOCAL_STAMP) &&
    line.includes("| 500 |") &&
    line.includes(TARGET_ROUTE),
  )
  .map(({ index }) => index);

const fallbackIndexes =
  exactIndexes.length > 0
    ? []
    : lines
        .map((line, index) => ({ line, index }))
        .filter(
          ({ line }) =>
            line.includes("| 500 |") &&
            line.includes(TARGET_ROUTE) &&
            line.includes("2026/09/24"),
        )
        .map(({ index }) => index);

const selectedIndex =
  exactIndexes.at(-1) ??
  fallbackIndexes.at(-1) ??
  null;

const context =
  selectedIndex === null
    ? []
    : lines.slice(
        Math.max(0, selectedIndex - BEFORE),
        Math.min(lines.length, selectedIndex + AFTER + 1),
      );

const contextStartIndex =
  selectedIndex === null ? null : Math.max(0, selectedIndex - BEFORE);

const numberedContext = context.map((line, offset) => ({
  lineNumber: (contextStartIndex ?? 0) + offset + 1,
  relativeToAnchor:
    selectedIndex === null
      ? null
      : (contextStartIndex ?? 0) + offset - selectedIndex,
  text: clip(line),
}));

const diagnosticPatterns = [
  /panic/i,
  /fatal/i,
  /out of memory/i,
  /\boom\b/i,
  /cuda/i,
  /runner/i,
  /llama-server/i,
  /exited/i,
  /exit code/i,
  /signal/i,
  /killed/i,
  /closed/i,
  /connection/i,
  /eof/i,
  /context deadline exceeded/i,
  /timed out/i,
  /error/i,
  /500/,
  /generate/i,
];

const diagnosticContextLines = numberedContext.filter(({ text }) =>
  diagnosticPatterns.some((pattern) => pattern.test(text)),
);

const anchorLine =
  selectedIndex === null ? null : clip(lines[selectedIndex] ?? "");

const immediatelyBefore =
  selectedIndex === null
    ? []
    : lines
        .slice(Math.max(0, selectedIndex - 25), selectedIndex)
        .map(clip);

const immediatelyAfter =
  selectedIndex === null
    ? []
    : lines
        .slice(selectedIndex + 1, Math.min(lines.length, selectedIndex + 26))
        .map(clip);

console.log(JSON.stringify({
  format:
    "OROTITAN_GATE18_PHASE_C_C4_STMICRO_OLLAMA_500_CONTEXT_FORENSICS_V0.1",
  gate: 18,
  phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
  stage: "C4_RUNTIME_FAILURE_FORENSICS",
  status:
    selectedIndex === null
      ? "BLOCKED_TARGET_500_ANCHOR_NOT_FOUND"
      : "SERVER_LOG_500_CONTEXT_COLLECTED_REVIEW_REQUIRED",
  mode: "LOCAL_READ_ONLY_NO_INFERENCE",
  sourceRun: {
    resultId:
      "G18-PHASEC-C4-STM-QWEN4B-CONTEXT16384-OUTPUT1024-TIMEOUT420-RESULT-001",
    forensicV4Result:
      "G18-PHASEC-C4-STM-OLLAMA-LOG-FORENSICS-V4-RESULT-001",
    runtimeError: "fetch failed",
    observedWallClockMs: 304446,
    configuredClientTimeoutMs: 420000,
  },
  serverLog: {
    path: SERVER_LOG,
    sizeBytes: stat.size,
    lastWriteTimeUtc: stat.mtime.toISOString(),
    totalLines: lines.length,
  },
  anchorSearch: {
    targetLocalStamp: TARGET_LOCAL_STAMP,
    targetRoute: TARGET_ROUTE,
    exactMatchCount: exactIndexes.length,
    fallbackMatchCount: fallbackIndexes.length,
    selectedLineNumber:
      selectedIndex === null ? null : selectedIndex + 1,
    anchorLine,
  },
  immediateContext: {
    before25: immediatelyBefore,
    after25: immediatelyAfter,
  },
  diagnosticContextLines,
  fullNumberedContext: numberedContext,
  interpretationBoundary: {
    localOllamaHttp500Observed: selectedIndex !== null,
    exactInternal500CauseConcluded: false,
    llamaServerProcessCrashConcluded: false,
    oomConcluded: false,
    cudaFailureConcluded: false,
    runnerExitConcluded: false,
    fetchFailureRootCauseConcluded: false,
    modelCapabilityFailureConcluded: false,
    semanticFailureConcluded: false,
    retryAuthorized: false,
    nextAction:
      "Human review of raw server.log lines surrounding the terminal /api/generate HTTP 500 before any retry or remediation.",
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
