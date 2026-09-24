import { execFileSync } from "node:child_process";

const RUN_START_UTC = "2026-09-24T17:20:51.014Z";
const RUN_END_UTC = "2026-09-24T18:38:03.196Z";
const QUERY_START_UTC = "2026-09-24T17:15:51.014Z";
const QUERY_END_UTC = "2026-09-24T18:43:03.196Z";

function runText(
  command: string,
  args: readonly string[],
  timeout = 45_000,
): { stdout: string | null; error: string | null } {
  try {
    const stdout = execFileSync(command, [...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout,
      maxBuffer: 16 * 1024 * 1024,
    }).trim();
    return { stdout, error: null };
  } catch (error) {
    const message =
      error instanceof Error ? error.message : String(error);
    return { stdout: null, error: message };
  }
}

function parseJsonObject(raw: string | null): unknown {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
}

const ps = [
  "$ErrorActionPreference='Continue'",
  `$start=[datetime]::Parse('${QUERY_START_UTC}').ToLocalTime()`,
  `$end=[datetime]::Parse('${QUERY_END_UTC}').ToLocalTime()`,
  "$specs=@(",
  "  [pscustomobject]@{Provider='Microsoft-Windows-Kernel-Power';Id=42},",
  "  [pscustomobject]@{Provider='Microsoft-Windows-Kernel-Power';Id=107},",
  "  [pscustomobject]@{Provider='Microsoft-Windows-Kernel-Power';Id=506},",
  "  [pscustomobject]@{Provider='Microsoft-Windows-Kernel-Power';Id=507},",
  "  [pscustomobject]@{Provider='Microsoft-Windows-Power-Troubleshooter';Id=1}",
  ")",
  "$results=@()",
  "foreach($spec in $specs){",
  "  try {",
  "    $items=@(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName=$spec.Provider;Id=$spec.Id;StartTime=$start;EndTime=$end} -ErrorAction Stop)",
  "    $events=@()",
  "    foreach($evt in $items){",
  "      $xmlText=$evt.ToXml()",
  "      [xml]$xml=$xmlText",
  "      $data=@{}",
  "      foreach($node in @($xml.Event.EventData.Data)){",
  "        $name=[string]$node.Name",
  "        if([string]::IsNullOrWhiteSpace($name)){ $name='UNNAMED' }",
  "        $data[$name]=[string]$node.'#text'",
  "      }",
  "      $events += [pscustomobject]@{",
  "        TimeCreatedUtc=$evt.TimeCreated.ToUniversalTime().ToString('o');",
  "        Id=$evt.Id;",
  "        ProviderName=$evt.ProviderName;",
  "        LevelDisplayName=$evt.LevelDisplayName;",
  "        Message=$evt.Message;",
  "        Data=$data;",
  "        Xml=$xmlText",
  "      }",
  "    }",
  "    $results += [pscustomobject]@{",
  "      Provider=$spec.Provider;",
  "      Id=$spec.Id;",
  "      QuerySucceeded=$true;",
  "      Error=$null;",
  "      EventCount=$events.Count;",
  "      Events=$events",
  "    }",
  "  } catch {",
  "    $results += [pscustomobject]@{",
  "      Provider=$spec.Provider;",
  "      Id=$spec.Id;",
  "      QuerySucceeded=$false;",
  "      Error=$_.Exception.Message;",
  "      EventCount=0;",
  "      Events=@()",
  "    }",
  "  }",
  "}",
  "[pscustomobject]@{Results=$results} | ConvertTo-Json -Depth 12 -Compress"
].join(";");

const execution = runText(
  "powershell.exe",
  ["-NoProfile", "-Command", ps],
);

const parsed = parseJsonObject(execution.stdout) as
  | { Results?: unknown[] }
  | null;

const results = Array.isArray(parsed?.Results)
  ? parsed.Results
  : [];

const eventCount = results.reduce((sum, item) => {
  if (
    item &&
    typeof item === "object" &&
    "EventCount" in item &&
    typeof (item as { EventCount?: unknown }).EventCount === "number"
  ) {
    return sum + ((item as { EventCount: number }).EventCount ?? 0);
  }
  return sum;
}, 0);

const failedQueryCount = results.filter((item) => {
  return (
    item &&
    typeof item === "object" &&
    "QuerySucceeded" in item &&
    (item as { QuerySucceeded?: unknown }).QuerySucceeded === false
  );
}).length;

console.log(JSON.stringify({
  format:
    "OROTITAN_GATE18_PHASE_C_C4_STMICRO_POWER_EVENT_DETAIL_FORENSIC_V0.2",
  gate: 18,
  phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
  stage: "C4_RUNTIME_FAILURE_FORENSICS",
  status:
    execution.stdout === null
      ? "BLOCKED_POWERSHELL_PROCESS_FAILED"
      : eventCount > 0
        ? "EVENT_DETAILS_COLLECTED_REVIEW_REQUIRED"
        : "NO_EVENTS_RETURNED_REVIEW_REQUIRED",
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
  powershellExecution: {
    processSucceeded: execution.stdout !== null,
    processError: execution.error,
    subqueryCount: results.length,
    failedQueryCount,
    totalEventCount: eventCount,
  },
  queryResults: results,
  interpretationBoundary: {
    sleepWakeTimingConcluded: false,
    timeoutDelayRootCauseConcluded: false,
    inferenceFitConcluded: false,
    modelCapabilityFailureConcluded: false,
    nextAction:
      "Human review of independently queried power events and XML Data fields before deciding runtime-remediation scope.",
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
