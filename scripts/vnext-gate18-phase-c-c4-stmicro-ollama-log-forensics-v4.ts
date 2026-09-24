import { execFileSync } from "node:child_process";

const RUN_START_UTC = new Date("2026-09-24T20:03:43.290Z");
const RUN_END_UTC = new Date("2026-09-24T20:08:47.736Z");
const MARGIN_MS = 5 * 60 * 1000;
const QUERY_START_UTC = new Date(RUN_START_UTC.getTime() - MARGIN_MS);
const QUERY_END_UTC = new Date(RUN_END_UTC.getTime() + MARGIN_MS);

interface ExecResult {
  stdout: string | null;
  stderr: string | null;
  error: string | null;
}

function runText(
  command: string,
  args: readonly string[],
  timeout = 60_000,
): ExecResult {
  try {
    const stdout = execFileSync(command, [...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout,
      maxBuffer: 64 * 1024 * 1024,
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
  timeout = 60_000,
): ExecResult {
  const encoded = Buffer.from(source, "utf16le").toString("base64");
  return runText(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
    timeout,
  );
}

function decodeBase64Json(raw: string | null): {
  decodedText: string | null;
  parsed: unknown;
  error: string | null;
} {
  if (!raw) {
    return { decodedText: null, parsed: null, error: "EMPTY_STDOUT" };
  }

  const candidate = raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .at(-1);

  if (!candidate) {
    return { decodedText: null, parsed: null, error: "NO_NONEMPTY_STDOUT_LINE" };
  }

  try {
    const decodedText = Buffer.from(candidate, "base64").toString("utf8");
    const parsed = JSON.parse(decodedText) as unknown;
    return { decodedText, parsed, error: null };
  } catch (error) {
    return {
      decodedText: null,
      parsed: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

const ps = [
  "$ErrorActionPreference='Stop'",
  "$ProgressPreference='SilentlyContinue'",
  "$root=Join-Path $env:LOCALAPPDATA 'Ollama'",
  `$windowStart=[datetimeoffset]::Parse('${QUERY_START_UTC.toISOString()}')`,
  `$windowEnd=[datetimeoffset]::Parse('${QUERY_END_UTC.toISOString()}')`,
  "$results=@()",
  "if(Test-Path $root){",
  "  foreach($file in @(Get-ChildItem $root -File -Filter '*.log' -ErrorAction SilentlyContinue)){",
  "    $matchedLines=@()",
  "    $windowLines=@()",
  "    $tail=@()",
  "    try {",
  "      $tail=@(Get-Content $file.FullName -Tail 4000 -ErrorAction Stop)",
  "      foreach($line in $tail){",
  "        $text=[string]$line",
  "        $interesting=[regex]::IsMatch($text,'(?i)(error|panic|fatal|cuda|out of memory|oom|connection|eof|failed|exit|killed|runner|crash|signal|closed|status=5[0-9][0-9]|generate|http|server)')",
  "        if($interesting){",
  "          if($text.Length -gt 1800){$text=$text.Substring(0,1800)}",
  "          $matchedLines += $text",
  "        }",
  "        $m=[regex]::Match([string]$line,'time=(?<ts>[^ ]+)')",
  "        if($m.Success){",
  "          $dto=[datetimeoffset]::MinValue",
  "          if([datetimeoffset]::TryParse($m.Groups['ts'].Value,[ref]$dto)){",
  "            $utc=$dto.ToUniversalTime()",
  "            if($utc -ge $windowStart -and $utc -le $windowEnd){",
  "              $windowText=[string]$line",
  "              if($windowText.Length -gt 1800){$windowText=$windowText.Substring(0,1800)}",
  "              $windowLines += $windowText",
  "            }",
  "          }",
  "        }",
  "      }",
  "    } catch {",
  "      $matchedLines += ('READ_ERROR:' + $_.Exception.Message)",
  "    }",
  "    $results += [pscustomobject]@{",
  "      Name=[string]$file.Name;",
  "      FullName=[string]$file.FullName;",
  "      LastWriteTimeUtc=$file.LastWriteTimeUtc.ToString('o');",
  "      Length=[int64]$file.Length;",
  "      TailLineCount=[int]$tail.Count;",
  "      WindowLineCount=[int]$windowLines.Count;",
  "      WindowLines=@($windowLines | Select-Object -Last 240);",
  "      InterestingLineCount=[int]$matchedLines.Count;",
  "      InterestingTailLines=@($matchedLines | Select-Object -Last 240)",
  "    }",
  "  }",
  "}",
  "$flat=@($results)",
  "$payload=[pscustomobject]@{Root=[string]$root;LogFileCount=[int]$flat.Count;Logs=$flat}",
  "$json=$payload | ConvertTo-Json -Depth 8 -Compress",
  "$bytes=[Text.Encoding]::UTF8.GetBytes($json)",
  "$b64=[Convert]::ToBase64String($bytes)",
  "[Console]::Out.WriteLine($b64)"
].join("\n");

const execution = runPowerShellEncoded(ps, 75_000);
const decoded = decodeBase64Json(execution.stdout);

console.log(JSON.stringify({
  format:
    "OROTITAN_GATE18_PHASE_C_C4_STMICRO_OLLAMA_LOG_FORENSICS_V0.2",
  gate: 18,
  phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
  stage: "C4_RUNTIME_FAILURE_FORENSICS",
  status:
    execution.stdout !== null && decoded.error === null
      ? "OLLAMA_LOG_MEASUREMENTS_COLLECTED_REVIEW_REQUIRED"
      : "BLOCKED_OLLAMA_LOG_COLLECTION_OR_DECODE_FAILED",
  mode: "LOCAL_READ_ONLY_NO_INFERENCE",
  sourceRun: {
    resultId:
      "G18-PHASEC-C4-STM-QWEN4B-CONTEXT16384-OUTPUT1024-TIMEOUT420-RESULT-001",
    runtimeError: "fetch failed",
    configuredClientTimeoutMs: 420000,
    observedWallClockMs: 304446,
  },
  queryWindow: {
    runStartUtc: RUN_START_UTC.toISOString(),
    runEndUtc: RUN_END_UTC.toISOString(),
    queriedStartUtc: QUERY_START_UTC.toISOString(),
    queriedEndUtc: QUERY_END_UTC.toISOString(),
  },
  transport: {
    powershellProcessSucceeded: execution.stdout !== null,
    stderr: execution.stderr,
    processError: execution.error,
    stdoutPresent: execution.stdout !== null,
    stdoutLength: execution.stdout?.length ?? 0,
    base64DecodeAndJsonParseSucceeded: decoded.error === null,
    decodeError: decoded.error,
  },
  ollamaLogs: decoded.parsed,
  interpretationBoundary: {
    ollamaRunnerCrashConcluded: false,
    oomConcluded: false,
    cudaFailureConcluded: false,
    localConnectionFailureMechanismConcluded: false,
    provider5xxConcluded: false,
    fetchFailureRootCauseConcluded: false,
    modelCapabilityFailureConcluded: false,
    semanticFailureConcluded: false,
    retryAuthorized: false,
    nextAction:
      "Human review of decoded time-window Ollama log evidence before any retry or runtime remediation.",
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
