import { execFileSync } from "node:child_process";
import os from "node:os";

const RUN_END_UTC = new Date("2026-09-24T18:38:03.196Z");
const RUN_WALL_CLOCK_MS = 4_632_182;
const WINDOW_MARGIN_MS = 5 * 60 * 1000;
const RUN_START_UTC = new Date(RUN_END_UTC.getTime() - RUN_WALL_CLOCK_MS);
const QUERY_START_UTC = new Date(RUN_START_UTC.getTime() - WINDOW_MARGIN_MS);
const QUERY_END_UTC = new Date(RUN_END_UTC.getTime() + WINDOW_MARGIN_MS);

function runText(
  command: string,
  args: readonly string[],
  timeout = 20_000,
): string | null {
  try {
    return execFileSync(command, [...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout,
      maxBuffer: 4 * 1024 * 1024,
    }).trim();
  } catch {
    return null;
  }
}

function gib(bytes: number): number {
  return Math.round((bytes / 1024 ** 3) * 100) / 100;
}

function parseNumber(value: string | undefined): number | null {
  if (value === undefined) return null;
  const parsed = Number(value.trim());
  return Number.isFinite(parsed) ? parsed : null;
}

function parseJsonArray(raw: string | null): unknown[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    return [];
  }
}

const nvidiaSmi = runText("nvidia-smi", [
  "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,pci.bus_id",
  "--format=csv,noheader,nounits",
]);

const nvidiaFields =
  nvidiaSmi?.split(/\r?\n/)[0]?.split(",").map((part) => part.trim()) ?? [];

const ollamaPsRaw = runText("ollama", ["ps"]);
const ollamaPsLines =
  ollamaPsRaw
    ?.split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean) ?? [];
const ollamaLoadedModelRows =
  ollamaPsLines.length > 1 ? ollamaPsLines.slice(1) : [];

const powerShellScript = [
  "$ErrorActionPreference='SilentlyContinue'",
  `$start=[datetime]::Parse('${QUERY_START_UTC.toISOString()}').ToLocalTime()`,
  `$end=[datetime]::Parse('${QUERY_END_UTC.toISOString()}').ToLocalTime()`,
  "$events=@()",
  "$events += Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Power';Id=42,107,506,507;StartTime=$start;EndTime=$end} | Select-Object TimeCreated,Id,ProviderName",
  "$events += Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Power-Troubleshooter';Id=1;StartTime=$start;EndTime=$end} | Select-Object TimeCreated,Id,ProviderName",
  "$events | Sort-Object TimeCreated | ConvertTo-Json -Compress",
].join(";");

const powerEventsRaw = runText(
  "powershell.exe",
  ["-NoProfile", "-Command", powerShellScript],
  30_000,
);
const powerEvents = parseJsonArray(powerEventsRaw);

const bootTimeRaw = runText(
  "powershell.exe",
  [
    "-NoProfile",
    "-Command",
    "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')",
  ],
  20_000,
);

const totalRamGiB = gib(os.totalmem());
const freeRamGiB = gib(os.freemem());

const payload = {
  format:
    "OROTITAN_GATE18_PHASE_C_C4_STMICRO_POST_ABORT_RUNTIME_FORENSICS_V0.1",
  gate: 18,
  phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
  stage: "C4_RUNTIME_FAILURE_FORENSICS",
  status: "MEASUREMENTS_COLLECTED_REVIEW_REQUIRED",
  mode: "LOCAL_READ_ONLY_NO_INFERENCE",
  sourceRun: {
    resultId: "G18-PHASEC-C4-STM-QWEN4B-CONTEXT16384-RUN-001",
    configuredTimeoutMs: 300000,
    observedWallClockMs: RUN_WALL_CLOCK_MS,
    runtimeError: "This operation was aborted",
    doneReason: null,
    promptEvalCount: null,
    evalCount: null,
  },
  runWindow: {
    estimatedStartUtc: RUN_START_UTC.toISOString(),
    observedEndUtc: RUN_END_UTC.toISOString(),
    queriedStartUtc: QUERY_START_UTC.toISOString(),
    queriedEndUtc: QUERY_END_UTC.toISOString(),
  },
  currentSystem: {
    totalRamGiB,
    freeRamGiB,
    bootTimeUtc: bootTimeRaw,
  },
  gpu: {
    name: nvidiaFields[0] ?? null,
    driverVersion: nvidiaFields[1] ?? null,
    memoryTotalMiB: parseNumber(nvidiaFields[2]),
    memoryFreeMiB: parseNumber(nvidiaFields[3]),
    memoryUsedMiB: parseNumber(nvidiaFields[4]),
    pciBusId: nvidiaFields[5] ?? null,
  },
  ollama: {
    psReachable: ollamaPsRaw !== null,
    loadedModelCount: ollamaLoadedModelRows.length,
    loadedModelRows: ollamaLoadedModelRows,
  },
  windowsPowerEvents: {
    querySucceeded: powerEventsRaw !== null,
    eventCount: powerEvents.length,
    events: powerEvents,
    relevantEventIds: [1, 42, 107, 506, 507],
  },
  interpretationBoundary: {
    sleepOrResumeEventObserved: powerEvents.length > 0,
    sleepOrResumeEventCausalityConcluded: false,
    timeoutGuardDelayRootCauseConcluded: false,
    modelCapabilityFailureConcluded: false,
    semanticFailureConcluded: false,
    inferenceFitConcluded: false,
    nextAction:
      "Human review of post-abort Ollama/resource state and Windows power-transition events before any reauthorization or runner timeout-guard remediation.",
  },
  safety: {
    networkAccessRequested: false,
    ollamaApiCalled: false,
    modelLoaded: false,
    modelInferenceExecuted: false,
    modelSwitchExecuted: false,
    productionMutation: false,
    publicationAuthority: false,
  },
};

console.log(JSON.stringify(payload, null, 2));
