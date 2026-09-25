import { readFileSync, statSync } from "node:fs";
import path from "node:path";

const ROOT = process.env.LOCALAPPDATA ?? "";
const SERVER_LOG = path.join(ROOT, "Ollama", "server.log");
const TARGET_UTC = Date.parse("2026-09-25T08:35:49.839Z");
const TARGET_LOCAL_LABEL = "2026-09-25T10:35:49.839+02:00";
const TARGET_DATE = "2026/09/25";
const TARGET_ROUTE = 'POST     "/api/generate"';
const MAX_ANCHOR_DISTANCE_MS = 120_000;
const BEFORE = 220;
const AFTER = 100;

function clip(value: string, max = 2400): string {
  return value.length > max ? value.slice(0, max) : value;
}

function parseGinGenerate(line: string): {
  epochMs: number;
  status: number;
  duration: string;
  line: string;
} | null {
  if (!line.includes(TARGET_ROUTE)) {
    return null;
  }

  const match = line.match(
    /\[GIN\]\s+(\d{4})\/(\d{2})\/(\d{2})\s+-\s+(\d{2}):(\d{2}):(\d{2})\s+\|\s+(\d{3})\s+\|\s+([^|]+)\|/,
  );

  if (!match) {
    return null;
  }

  const [, year, month, day, hour, minute, second, status, duration] =
    match;

  const epochMs = Date.parse(
    `${year}-${month}-${day}T${hour}:${minute}:${second}+02:00`,
  );

  if (!Number.isFinite(epochMs)) {
    return null;
  }

  return {
    epochMs,
    status: Number(status),
    duration: duration.trim(),
    line,
  };
}

let raw: string;
let stat: ReturnType<typeof statSync>;

try {
  raw = readFileSync(SERVER_LOG, "utf8");
  stat = statSync(SERVER_LOG);
} catch (error) {
  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_TIMEOUT420_LOG_FORENSICS_V0.1",
        gate: 18,
        phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
        stage: "C4_RUNTIME_FAILURE_FORENSICS",
        status: "BLOCKED_SERVER_LOG_READ_FAILED",
        mode: "LOCAL_READ_ONLY_NO_INFERENCE",
        serverLogPath: SERVER_LOG,
        error:
          error instanceof Error ? error.message : String(error),
        safety: {
          externalNetworkAccessRequested: false,
          ollamaApiCalled: false,
          modelInferenceExecuted: false,
          modelLoadRequested: false,
          modelSwitchExecuted: false,
          productionMutation: false,
          publicationAuthority: false,
        },
      },
      null,
      2,
    ),
  );
  process.exit(0);
}

const lines = raw.split(/\r?\n/);
const candidates = lines
  .map((line, index) => {
    const parsed = parseGinGenerate(line);
    return parsed === null
      ? null
      : {
          ...parsed,
          index,
          lineNumber: index + 1,
          distanceMs: Math.abs(parsed.epochMs - TARGET_UTC),
        };
  })
  .filter(
    (
      item,
    ): item is {
      epochMs: number;
      status: number;
      duration: string;
      line: string;
      index: number;
      lineNumber: number;
      distanceMs: number;
    } => item !== null,
  );

const nearby = candidates
  .filter((item) => item.distanceMs <= MAX_ANCHOR_DISTANCE_MS)
  .sort((a, b) => a.distanceMs - b.distanceMs);

const sameDate = candidates
  .filter((item) => item.line.includes(TARGET_DATE))
  .sort((a, b) => a.distanceMs - b.distanceMs);

const selected = nearby[0] ?? sameDate[0] ?? null;
const selectedIndex = selected?.index ?? null;

const context =
  selectedIndex === null
    ? []
    : lines.slice(
        Math.max(0, selectedIndex - BEFORE),
        Math.min(lines.length, selectedIndex + AFTER + 1),
      );

const contextStartIndex =
  selectedIndex === null
    ? null
    : Math.max(0, selectedIndex - BEFORE);

const numberedContext = context.map((line, offset) => ({
  lineNumber: (contextStartIndex ?? 0) + offset + 1,
  relativeToAnchor:
    selectedIndex === null
      ? null
      : (contextStartIndex ?? 0) + offset - selectedIndex,
  text: clip(line),
}));

const diagnosticPatterns = [
  /n_prompt/i,
  /n_gen/i,
  /prompt eval/i,
  /eval rate/i,
  /prompt done/i,
  /generation/i,
  /slot/i,
  /cancel/i,
  /context canceled/i,
  /context deadline exceeded/i,
  /client disconnected/i,
  /connection/i,
  /closed/i,
  /broken pipe/i,
  /eof/i,
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
  /timed out/i,
  /error/i,
  /\/api\/generate/i,
];

const diagnosticContextLines = numberedContext.filter(({ text }) =>
  diagnosticPatterns.some((pattern) => pattern.test(text)),
);

const progressLines = numberedContext.filter(({ text }) =>
  /n_prompt\s*=|n_gen\s*=|prompt done|eval rate/i.test(text),
);

const generationCounts = progressLines
  .map(({ lineNumber, relativeToAnchor, text }) => {
    const match = text.match(/n_gen\s*=\s*(\d+)/i);
    return match
      ? {
          lineNumber,
          relativeToAnchor,
          nGen: Number(match[1]),
          text,
        }
      : null;
  })
  .filter(
    (
      item,
    ): item is {
      lineNumber: number;
      relativeToAnchor: number | null;
      nGen: number;
      text: string;
    } => item !== null,
  );

const latestGenerationProgress =
  generationCounts.at(-1) ?? null;

const immediateBefore =
  selectedIndex === null
    ? []
    : lines
        .slice(Math.max(0, selectedIndex - 35), selectedIndex)
        .map(clip);

const immediateAfter =
  selectedIndex === null
    ? []
    : lines
        .slice(
          selectedIndex + 1,
          Math.min(lines.length, selectedIndex + 36),
        )
        .map(clip);

console.log(
  JSON.stringify(
    {
      format:
        "OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_TIMEOUT420_LOG_FORENSICS_V0.1",
      gate: 18,
      phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
      stage: "C4_RUNTIME_FAILURE_FORENSICS",
      status:
        selected === null
          ? "BLOCKED_GENERATE_ANCHOR_NOT_FOUND"
          : "SERVER_LOG_CONTEXT_COLLECTED_REVIEW_REQUIRED",
      mode: "LOCAL_READ_ONLY_NO_INFERENCE",
      sourceRun: {
        resultId:
          "G18-PHASEC-C4-CONSTELLATION-QWEN4B-CONTEXT16384-OUTPUT1024-TIMEOUT420-LOOPBACK-RESULT-001",
        runtimeError:
          "LOOPBACK_HTTP_EXPLICIT_TIMEOUT_420000MS",
        observedWallClockMs: 420485,
        configuredClientTimeoutMs: 420000,
        providerResponseObserved: false,
        estimatedStartUtc:
          "2026-09-25T08:28:49.354Z",
        estimatedEndUtc:
          "2026-09-25T08:35:49.839Z",
        estimatedStartLocal:
          "2026-09-25T10:28:49.354+02:00",
        estimatedEndLocal:
          TARGET_LOCAL_LABEL,
      },
      serverLog: {
        path: SERVER_LOG,
        sizeBytes: stat.size,
        lastWriteTimeUtc: stat.mtime.toISOString(),
        totalLines: lines.length,
      },
      anchorSearch: {
        targetEndUtc:
          "2026-09-25T08:35:49.839Z",
        targetEndLocal: TARGET_LOCAL_LABEL,
        targetRoute: TARGET_ROUTE,
        maxAnchorDistanceMs: MAX_ANCHOR_DISTANCE_MS,
        totalGenerateCandidates: candidates.length,
        nearbyCandidateCount: nearby.length,
        nearbyCandidates: nearby.slice(0, 8).map((item) => ({
          lineNumber: item.lineNumber,
          status: item.status,
          duration: item.duration,
          distanceMs: item.distanceMs,
          text: clip(item.line),
        })),
        selected:
          selected === null
            ? null
            : {
                lineNumber: selected.lineNumber,
                status: selected.status,
                duration: selected.duration,
                distanceMs: selected.distanceMs,
                text: clip(selected.line),
              },
      },
      progress: {
        latestGenerationProgress,
        generationProgressSample: generationCounts.slice(-20),
      },
      immediateContext: {
        before35: immediateBefore,
        after35: immediateAfter,
      },
      diagnosticContextLines,
      fullNumberedContext: numberedContext,
      interpretationBoundary: {
        explicitClientTimeoutObserved: true,
        serverRequestTerminationObserved:
          selected !== null,
        exactRuntimeMechanismConcluded: false,
        healthySlowGenerationConcluded: false,
        serverStallConcluded: false,
        oomConcluded: false,
        cudaFailureConcluded: false,
        runnerExitConcluded: false,
        timeoutIncreaseJustified: false,
        retryAuthorized: false,
        nextAction:
          "Human review of raw Ollama server.log context and latest generation progress before any timeout change or retry.",
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
    },
    null,
    2,
  ),
);
