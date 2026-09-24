import { execFileSync } from "node:child_process";
import os from "node:os";

const RUN_END_UTC = new Date("2026-09-24T20:08:47.736Z");
const RUN_WALL_CLOCK_MS = 304_446;
const RUN_START_UTC = new Date(RUN_END_UTC.getTime() - RUN_WALL_CLOCK_MS);
const WINDOW_MARGIN_MS = 5 * 60 * 1000;
const QUERY_START_UTC = new Date(RUN_START_UTC.getTime() - WINDOW_MARGIN_MS);
const QUERY_END_UTC = new Date(RUN_END_UTC.getTime() + WINDOW_MARGIN_MS);

interface ExecResult {
  stdout: string | null;
  stderr: string | null;
  error: string | null;
}

function runText(
  command: string,
  args: readonly string[],
  timeout = 30_000,
): ExecResult {
  try {
    const stdout = execFileSync(command, [...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout,
      maxBuffer: 16 * 1024 * 1024,
    }).trim();
    return { stdout, stderr: null, error: null };
  } catch (error) {
    const err = error as Error & { stderr?: string | Buffer };
    const stderr =
      typeof err.stderr === "string"
        ? err.stderr.trim()
        : Buffer.isBuffer(err.stderr)
          ? err.stderr.toString("utf8").trim()
          : null;
    return {
      stdout: null,
      stderr,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function runPowerShellEncoded(
  source: string,
  timeout = 30_000,
): ExecResult {
  const encoded = Buffer.from(source, "utf16le").toString("base64");
  return runText(
    "powershell.exe",
    ["-NoProfile", "-EncodedCommand", encoded],
    timeout,
  );
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

function eventQuery(
  logName: string,
  providerName: string,
  ids: readonly number[],
): string {
  return [
    "$ErrorActionPreference='Stop'",
    `$start=[datetime]::Parse('${QUERY_START_UTC.toISOString()}').ToLocalTime()`,
    `$end=[datetime]::Parse('${QUERY_END_UTC.toISOString()}').ToLocalTime()`,
    `$ids=@(${ids.join(",")})`,
    "$events=@()",
    "foreach($id in $ids){",
    "  try {",
    `    $items=@(Get-WinEvent -FilterHashtable @{LogName='${logName}';ProviderName='${providerName}';Id=$id;StartTime=$start;EndTime=$end} -ErrorAction Stop)`,
    "  } catch {",
    "    $items=@()",
    "  }",
    "  foreach($evt in $items){",
    "    $msg=[string]$evt.Message",
    "    if($msg.Length -gt 1400){$msg=$msg.Substring(0,1400)}",
    "    $events += [pscustomobject]@{TimeCreatedUtc=$evt.TimeCreated.ToUniversalTime().ToString('o');Id=$evt.Id;ProviderName=$evt.ProviderName;Level=$evt.LevelDisplayName;Message=$msg}",
    "  }",
    "}",
    "[pscustomobject]@{QuerySucceeded=$true;EventCount=$events.Count;Events=@($events | Sort-Object TimeCreatedUtc | Select-Object -First 80)} | ConvertTo-Json -Depth 7 -Compress",
  ].join("\n");
}

const ollamaVersion = runText("ollama", ["--version"], 10_000);
const ollamaPs = runText("ollama", ["ps"], 10_000);
const nvidia = runText(
  "nvidia-smi",
  [
    "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
  ],
  15_000,
);

const systemProbe = runPowerShellEncoded([
  "$ErrorActionPreference='Stop'",
  "$boot=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')",
  "$proc=@(Get-Process -Name 'ollama*' -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,@{Name='StartTimeUtc';Expression={try{$_.StartTime.ToUniversalTime().ToString('o')}catch{$null}}},CPU,WorkingSet64,Path)",
  "[pscustomobject]@{BootTimeUtc=$boot;Processes=$proc} | ConvertTo-Json -Depth 6 -Compress",
].join("\n"));

const powerProbe = runPowerShellEncoded(
  eventQuery(
    "System",
    "Microsoft-Windows-Kernel-Power",
    [41, 42, 107, 506, 507],
  ),
  40_000,
);

const wakeProbe = runPowerShellEncoded(
  eventQuery(
    "System",
    "Microsoft-Windows-Power-Troubleshooter",
    [1],
  ),
  40_000,
);

const resourceProbe = runPowerShellEncoded(
  eventQuery(
    "System",
    "Microsoft-Windows-Resource-Exhaustion-Detector",
    [2004],
  ),
  40_000,
);

const displayProbe = runPowerShellEncoded(
  eventQuery("System", "Display", [4101]),
  40_000,
);

const appErrorProbe = runPowerShellEncoded(
  eventQuery("Application", "Application Error", [1000]),
  40_000,
);

const werProbe = runPowerShellEncoded(
  eventQuery("Application", "Windows Error Reporting", [1001]),
  40_000,
);

const ollamaLogsProbe = runPowerShellEncoded([
  "$ErrorActionPreference='Stop'",
  "$root=Join-Path $env:LOCALAPPDATA 'Ollama'",
  "$logs=@()",
  "if(Test-Path $root){",
  "  foreach($file in @(Get-ChildItem $root -File -Filter '*.log' -ErrorAction SilentlyContinue)){",
  "    $matches=@()",
  "    try {",
  "      $tail=@(Get-Content $file.FullName -Tail 700 -ErrorAction Stop)",
  "      foreach($line in $tail){",
  "        if($line -match '(?i)(error|panic|fatal|cuda|out of memory|oom|connection|eof|failed|exit|killed|runner|crash|signal|closed)'){",
  "          $clean=[string]$line",
  "          if($clean.Length -gt 1200){$clean=$clean.Substring(0,1200)}",
  "          $matches += $clean",
  "        }",
  "      }",
  "    } catch {}",
  "    $logs += [pscustomobject]@{Name=$file.Name;LastWriteTimeUtc=$file.LastWriteTimeUtc.ToString('o');Length=$file.Length;MatchedTailLines=@($matches | Select-Object -Last 120)}",
  "  }",
  "}",
  "[pscustomobject]@{Root=$root;LogCount=$logs.Count;Logs=$logs} | ConvertTo-Json -Depth 7 -Compress",
].join("\n"), 45_000);

const probes = {
  system: {
    processSucceeded: systemProbe.stdout !== null,
    stderr: systemProbe.stderr,
    error: systemProbe.error,
    data: parseJson(systemProbe.stdout),
  },
  power: {
    processSucceeded: powerProbe.stdout !== null,
    stderr: powerProbe.stderr,
    error: powerProbe.error,
    data: parseJson(powerProbe.stdout),
  },
  wake: {
    processSucceeded: wakeProbe.stdout !== null,
    stderr: wakeProbe.stderr,
    error: wakeProbe.error,
    data: parseJson(wakeProbe.stdout),
  },
  resourceExhaustion: {
    processSucceeded: resourceProbe.stdout !== null,
    stderr: resourceProbe.stderr,
    error: resourceProbe.error,
    data: parseJson(resourceProbe.stdout),
  },
  displayReset: {
    processSucceeded: displayProbe.stdout !== null,
    stderr: displayProbe.stderr,
    error: displayProbe.error,
    data: parseJson(displayProbe.stdout),
  },
  applicationError: {
    processSucceeded: appErrorProbe.stdout !== null,
    stderr: appErrorProbe.stderr,
    error: appErrorProbe.error,
    data: parseJson(appErrorProbe.stdout),
  },
  windowsErrorReporting: {
    processSucceeded: werProbe.stdout !== null,
    stderr: werProbe.stderr,
    error: werProbe.error,
    data: parseJson(werProbe.stdout),
  },
  ollamaLogs: {
    processSucceeded: ollamaLogsProbe.stdout !== null,
    stderr: ollamaLogsProbe.stderr,
    error: ollamaLogsProbe.error,
    data: parseJson(ollamaLogsProbe.stdout),
  },
};

const successfulProbeCount = Object.values(probes).filter(
  (probe) => probe.processSucceeded,
).length;

console.log(JSON.stringify({
  format:
    "OROTITAN_GATE18_PHASE_C_C4_STMICRO_FETCH_FAILURE_FORENSICS_V0.2",
  gate: 18,
  phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
  stage: "C4_RUNTIME_FAILURE_FORENSICS",
  status:
    successfulProbeCount > 0
      ? "INDEPENDENT_MEASUREMENTS_COLLECTED_REVIEW_REQUIRED"
      : "BLOCKED_ALL_POWERSHELL_PROBES_FAILED",
  mode: "LOCAL_READ_ONLY_NO_INFERENCE",
  sourceRun: {
    resultId:
      "G18-PHASEC-C4-STM-QWEN4B-CONTEXT16384-OUTPUT1024-TIMEOUT420-RESULT-001",
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
  directTools: {
    ollamaVersion,
    ollamaPs,
    nvidiaSmi: nvidia,
  },
  probeSummary: {
    independentProbeCount: Object.keys(probes).length,
    successfulProbeCount,
  },
  probes,
  interpretationBoundary: {
    fetchFailureObserved: true,
    clientTimeoutReached: false,
    providerResponseObserved: false,
    ollamaProcessCrashConcluded: false,
    resourceExhaustionConcluded: false,
    gpuResetConcluded: false,
    powerTransitionConcluded: false,
    windowsApplicationCrashConcluded: false,
    modelCapabilityFailureConcluded: false,
    semanticFailureConcluded: false,
    retryAuthorized: false,
    nextAction:
      "Human review of independent probe evidence before any retry or runtime-parameter remediation.",
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
}, null, 2));
