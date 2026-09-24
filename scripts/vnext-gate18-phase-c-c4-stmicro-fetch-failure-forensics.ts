import { execFileSync } from "node:child_process";
import os from "node:os";

const RUN_END_UTC = new Date("2026-09-24T20:08:47.736Z");
const RUN_WALL_CLOCK_MS = 304_446;
const RUN_START_UTC = new Date(RUN_END_UTC.getTime() - RUN_WALL_CLOCK_MS);
const WINDOW_MARGIN_MS = 5 * 60 * 1000;
const QUERY_START_UTC = new Date(RUN_START_UTC.getTime() - WINDOW_MARGIN_MS);
const QUERY_END_UTC = new Date(RUN_END_UTC.getTime() + WINDOW_MARGIN_MS);

function runText(
  command: string,
  args: readonly string[],
  timeout = 30_000,
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
    return {
      stdout: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function parseJson(raw: string | null): unknown {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
}

function gib(bytes: number): number {
  return Math.round((bytes / 1024 ** 3) * 100) / 100;
}

const ollamaVersion = runText("ollama", ["--version"], 10_000);
const ollamaPs = runText("ollama", ["ps"], 10_000);
const nvidia = runText("nvidia-smi", [
  "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,utilization.gpu",
  "--format=csv,noheader,nounits",
], 15_000);

const ps = [
  "$ErrorActionPreference='SilentlyContinue'",
  `$start=[datetime]::Parse('${QUERY_START_UTC.toISOString()}').ToLocalTime()`,
  `$end=[datetime]::Parse('${QUERY_END_UTC.toISOString()}').ToLocalTime()`,
  "$eventSpecs=@(",
  "  [pscustomobject]@{Log='System';Provider='Microsoft-Windows-Kernel-Power';Ids=@(41,42,107,506,507)},",
  "  [pscustomobject]@{Log='System';Provider='Microsoft-Windows-Power-Troubleshooter';Ids=@(1)},",
  "  [pscustomobject]@{Log='System';Provider='Microsoft-Windows-Resource-Exhaustion-Detector';Ids=@(2004)},",
  "  [pscustomobject]@{Log='System';Provider='Display';Ids=@(4101)},",
  "  [pscustomobject]@{Log='Application';Provider='Application Error';Ids=@(1000)},",
  "  [pscustomobject]@{Log='Application';Provider='Windows Error Reporting';Ids=@(1001)}",
  ")",
  "$events=@()",
  "foreach($spec in $eventSpecs){",
  "  foreach($id in $spec.Ids){",
  "    try {",
  "      $items=@(Get-WinEvent -FilterHashtable @{LogName=$spec.Log;ProviderName=$spec.Provider;Id=$id;StartTime=$start;EndTime=$end} -ErrorAction Stop)",
  "      foreach($evt in $items){",
  "        $msg=[string]$evt.Message",
  "        if($msg.Length -gt 1200){ $msg=$msg.Substring(0,1200) }",
  "        $events += [pscustomobject]@{TimeCreatedUtc=$evt.TimeCreated.ToUniversalTime().ToString('o');LogName=$spec.Log;Id=$evt.Id;ProviderName=$evt.ProviderName;Level=$evt.LevelDisplayName;Message=$msg}",
  "      }",
  "    } catch {}",
  "  }",
  "}",
  "$proc=@(Get-Process -Name 'ollama*' -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,StartTime,CPU,WorkingSet64,Path)",
  "$logRoot=Join-Path $env:LOCALAPPDATA 'Ollama'",
  "$logs=@()",
  "if(Test-Path $logRoot){",
  "  foreach($file in @(Get-ChildItem $logRoot -File -Filter '*.log' -ErrorAction SilentlyContinue)){",
  "    $matches=@()",
  "    try {",
  "      $tail=@(Get-Content $file.FullName -Tail 400 -ErrorAction Stop)",
  "      foreach($line in $tail){",
  "        if($line -match '(?i)(error|panic|fatal|cuda|out of memory|oom|connection|eof|failed|exit|killed|runner|crash)'){",
  "          $clean=[string]$line",
  "          if($clean.Length -gt 1000){$clean=$clean.Substring(0,1000)}",
  "          $matches += $clean",
  "        }",
  "      }",
  "    } catch {}",
  "    $logs += [pscustomobject]@{Name=$file.Name;LastWriteTimeUtc=$file.LastWriteTimeUtc.ToString('o');Length=$file.Length;MatchedTailLines=@($matches | Select-Object -Last 80)}",
  "  }",
  "}",
  "$boot=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')",
  "[pscustomobject]@{BootTimeUtc=$boot;Processes=$proc;Events=@($events | Sort-Object TimeCreatedUtc);OllamaLogs=$logs} | ConvertTo-Json -Depth 8 -Compress"
].join("\n");

const windows = runText("powershell.exe", ["-NoProfile", "-Command", ps], 45_000);
const windowsParsed = parseJson(windows.stdout);

const payload = {
  format: "OROTITAN_GATE18_PHASE_C_C4_STMICRO_FETCH_FAILURE_FORENSICS_V0.1",
  gate: 18,
  phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
  stage: "C4_RUNTIME_FAILURE_FORENSICS",
  status: "MEASUREMENTS_COLLECTED_REVIEW_REQUIRED",
  mode: "LOCAL_READ_ONLY_NO_INFERENCE",
  sourceRun: {
    resultId: "G18-PHASEC-C4-STM-QWEN4B-CONTEXT16384-OUTPUT1024-TIMEOUT420-RESULT-001",
    configuredClientTimeoutMs: 420000,
    observedWallClockMs: RUN_WALL_CLOCK_MS,
    runtimeError: "fetch failed",
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
    totalRamGiB: gib(os.totalmem()),
    freeRamGiB: gib(os.freemem()),
  },
  ollama: {
    version: ollamaVersion.stdout,
    versionError: ollamaVersion.error,
    ps: ollamaPs.stdout,
    psError: ollamaPs.error,
  },
  gpu: {
    nvidiaSmi: nvidia.stdout,
    nvidiaSmiError: nvidia.error,
  },
  windows: {
    querySucceeded: windows.stdout !== null,
    queryError: windows.error,
    measurements: windowsParsed,
  },
  interpretationBoundary: {
    fetchFailureObserved: true,
    clientTimeoutReached: false,
    providerResponseObserved: false,
    ollamaProcessCrashConcluded: false,
    resourceExhaustionConcluded: false,
    gpuResetConcluded: false,
    powerTransitionConcluded: false,
    networkCauseConcluded: false,
    modelCapabilityFailureConcluded: false,
    semanticFailureConcluded: false,
    retryAuthorized: false,
    nextAction:
      "Human review of Ollama process state, filtered local Ollama log errors, GPU/resource state, and Windows events before any retry or parameter change.",
  },
  safety: {
    externalNetworkAccessRequested: false,
    ollamaGenerateApiCalled: false,
    modelInferenceExecuted: false,
    modelLoadRequested: false,
    modelSwitchExecuted: false,
    persistentPowerMutation: false,
    productionMutation: false,
    publicationAuthority: false,
  },
};

console.log(JSON.stringify(payload, null, 2));
