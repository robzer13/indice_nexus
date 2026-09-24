import { execFileSync } from "node:child_process";

const RUN_START_UTC = "2026-09-24T17:20:51.014Z";
const RUN_END_UTC = "2026-09-24T18:38:03.196Z";
const QUERY_START_UTC = "2026-09-24T17:15:51.014Z";
const QUERY_END_UTC = "2026-09-24T18:43:03.196Z";

function runText(
  command: string,
  args: readonly string[],
  timeout = 30_000,
): string | null {
  try {
    return execFileSync(command, [...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout,
      maxBuffer: 8 * 1024 * 1024,
    }).trim();
  } catch {
    return null;
  }
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

const ps = [
  "$ErrorActionPreference='Stop'",
  `$start=[datetime]::Parse('${QUERY_START_UTC}').ToLocalTime()`,
  `$end=[datetime]::Parse('${QUERY_END_UTC}').ToLocalTime()`,
  "$events=@()",
  "$events += Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Power';Id=42,107,506,507;StartTime=$start;EndTime=$end}",
  "$events += Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Power-Troubleshooter';Id=1;StartTime=$start;EndTime=$end}",
  "$events | Sort-Object TimeCreated | ForEach-Object {",
  "  [pscustomobject]@{",
  "    TimeCreatedUtc=$_.TimeCreated.ToUniversalTime().ToString('o');",
  "    Id=$_.Id;",
  "    ProviderName=$_.ProviderName;",
  "    LevelDisplayName=$_.LevelDisplayName;",
  "    Message=$_.Message;",
  "    Xml=$_.ToXml()",
  "  }",
  "} | ConvertTo-Json -Depth 6 -Compress"
].join(";");

const raw = runText("powershell.exe", ["-NoProfile", "-Command", ps]);
const events = parseJsonArray(raw);

console.log(JSON.stringify({
  format:
    "OROTITAN_GATE18_PHASE_C_C4_STMICRO_POWER_EVENT_DETAIL_FORENSIC_V0.1",
  gate: 18,
  phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
  stage: "C4_RUNTIME_FAILURE_FORENSICS",
  status: raw === null
    ? "BLOCKED_EVENT_QUERY_FAILED"
    : "EVENT_DETAILS_COLLECTED_REVIEW_REQUIRED",
  mode: "LOCAL_READ_ONLY_NO_INFERENCE",
  sourceRun: {
    resultId: "G18-PHASEC-C4-STM-QWEN4B-CONTEXT16384-RUN-001",
    runStartUtc: RUN_START_UTC,
    runEndUtc: RUN_END_UTC,
    configuredTimeoutMs: 300000,
    observedWallClockMs: 4632182,
  },
  queryWindow: {
    startUtc: QUERY_START_UTC,
    endUtc: QUERY_END_UTC,
  },
  events,
  interpretationBoundary: {
    sleepWakeTimingConcluded: false,
    timeoutDelayRootCauseConcluded: false,
    inferenceFitConcluded: false,
    modelCapabilityFailureConcluded: false,
    nextAction:
      "Human review of exact power-event messages and XML sleep/wake timestamps before deciding runtime-remediation scope.",
  },
  safety: {
    networkAccessRequested: false,
    ollamaApiCalled: false,
    modelInferenceExecuted: false,
    modelLoaded: false,
    productionMutation: false,
    publicationAuthority: false,
  },
}, null, 2));
