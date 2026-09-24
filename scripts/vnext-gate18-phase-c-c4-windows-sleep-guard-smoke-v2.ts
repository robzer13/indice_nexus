import { execFileSync, spawn } from "node:child_process";

const READY_PREFIX = "OROTITAN_SLEEP_GUARD_READY:";
const RELEASE_PREFIX = "OROTITAN_SLEEP_GUARD_RELEASED:";
const HOLD_MS = 5_000;

function runText(
  command: string,
  args: readonly string[],
  timeout = 15_000,
): { stdout: string | null; error: string | null } {
  try {
    const stdout = execFileSync(command, [...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout,
      maxBuffer: 4 * 1024 * 1024,
    }).trim();
    return { stdout, error: null };
  } catch (error) {
    return {
      stdout: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

async function waitUntil(
  predicate: () => boolean,
  timeoutMs: number,
): Promise<boolean> {
  const startedAt = Date.now();
  while (!predicate() && Date.now() - startedAt < timeoutMs) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return predicate();
}

async function main(): Promise<void> {
  if (process.platform !== "win32") {
    console.log(JSON.stringify({
      format: "OROTITAN_GATE18_PHASE_C_C4_WINDOWS_SLEEP_GUARD_SMOKE_V0.2",
      status: "BLOCKED_WINDOWS_REQUIRED",
      mode: "PROCESS_SCOPED_POWER_REQUEST_ONLY_NO_INFERENCE",
      safety: {
        modelInferenceExecuted: false,
        persistentPowerPlanMutation: false,
      },
    }, null, 2));
    return;
  }

  const before = runText("powercfg.exe", ["/requests"]);

  const helperSource = [
    "$ErrorActionPreference='Stop'",
    "$signature=@'",
    "using System;",
    "using System.Runtime.InteropServices;",
    "public static class OroTitanExecutionState {",
    "  [DllImport(\"kernel32.dll\", SetLastError = true)]",
    "  public static extern uint SetThreadExecutionState(uint esFlags);",
    "}",
    "'@",
    "Add-Type -TypeDefinition $signature",
    "$ES_CONTINUOUS = [Convert]::ToUInt32('80000000', 16)",
    "$ES_SYSTEM_REQUIRED = [uint32]1",
    "$flags = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)",
    "$acquire = [OroTitanExecutionState]::SetThreadExecutionState($flags)",
    "if($acquire -eq 0){ throw 'SET_THREAD_EXECUTION_STATE_ACQUIRE_FAILED' }",
    `Write-Output "${READY_PREFIX}$acquire"`,
    "[Console]::Out.Flush()",
    `Start-Sleep -Milliseconds ${HOLD_MS}`,
    "$release = [OroTitanExecutionState]::SetThreadExecutionState($ES_CONTINUOUS)",
    "if($release -eq 0){ throw 'SET_THREAD_EXECUTION_STATE_RELEASE_FAILED' }",
    `Write-Output "${RELEASE_PREFIX}$release"`,
    "[Console]::Out.Flush()",
  ].join("\n");

  const helper = spawn(
    "powershell.exe",
    ["-NoProfile", "-Command", helperSource],
    {
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  let helperStdout = "";
  let helperStderr = "";
  let readyValue: string | null = null;
  let releaseValue: string | null = null;

  helper.stdout.setEncoding("utf8");
  helper.stderr.setEncoding("utf8");

  helper.stdout.on("data", (chunk: string) => {
    helperStdout += chunk;
    for (const line of helperStdout.split(/\r?\n/)) {
      if (line.startsWith(READY_PREFIX)) {
        readyValue = line.slice(READY_PREFIX.length).trim();
      }
      if (line.startsWith(RELEASE_PREFIX)) {
        releaseValue = line.slice(RELEASE_PREFIX.length).trim();
      }
    }
  });

  helper.stderr.on("data", (chunk: string) => {
    helperStderr += chunk;
  });

  await waitUntil(
    () => readyValue !== null || helper.exitCode !== null,
    10_000,
  );

  const during = runText("powercfg.exe", ["/requests"]);

  await waitUntil(
    () => helper.exitCode !== null,
    15_000,
  );

  let forcedTermination = false;
  if (helper.exitCode === null && helper.pid) {
    forcedTermination = true;
    runText("taskkill.exe", ["/PID", String(helper.pid), "/T", "/F"]);
    await waitUntil(() => helper.exitCode !== null, 5_000);
  }

  const after = runText("powercfg.exe", ["/requests"]);

  const apiAcquireSucceeded = readyValue !== null;
  const apiReleaseSucceeded = releaseValue !== null;
  const helperExitedCleanly = helper.exitCode === 0;
  const pass =
    apiAcquireSucceeded &&
    apiReleaseSucceeded &&
    helperExitedCleanly &&
    !forcedTermination;

  console.log(JSON.stringify({
    format: "OROTITAN_GATE18_PHASE_C_C4_WINDOWS_SLEEP_GUARD_SMOKE_V0.2",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C4_RUNTIME_ENVIRONMENT_REMEDIATION",
    status: pass
      ? "PASS_API_GUARD_ACQUIRED_AND_RELEASED"
      : "REVIEW_REQUIRED",
    mode: "PROCESS_SCOPED_POWER_REQUEST_ONLY_NO_INFERENCE",
    helper: {
      readyObserved: apiAcquireSucceeded,
      releasedObserved: apiReleaseSucceeded,
      setThreadExecutionStateAcquirePriorFlagsRaw: readyValue,
      setThreadExecutionStateReleasePriorFlagsRaw: releaseValue,
      pid: helper.pid ?? null,
      stderr: helperStderr.trim() || null,
      exitCode: helper.exitCode,
      forcedTermination,
    },
    powercfgRequests: {
      verificationRequiredForPass: false,
      before,
      during,
      after,
      elevationRequiredObserved:
        [before, during, after].some((item) =>
          item.error?.toLowerCase().includes("privil") ?? false
        ),
    },
    interpretationBoundary: {
      idleSleepPreventionMechanismInvoked: apiAcquireSucceeded,
      processScopedGuardReleaseObserved: apiReleaseSucceeded,
      windowsApiGuardSmokePassed: pass,
      powercfgTelemetryRequiredForPass: false,
      lidOrManualSleepPreventionConcluded: false,
      actualLongDurationSleepPreventionProven: false,
      c4InferenceAuthorized: false,
      nextAction:
        "Human review of direct SetThreadExecutionState acquire/release proof before integrating the guard into any inference runner.",
    },
    safety: {
      networkAccessRequested: false,
      ollamaApiCalled: false,
      modelLoaded: false,
      modelInferenceExecuted: false,
      persistentPowerPlanMutation: false,
      displayRequiredFlagUsed: false,
      awayModeFlagUsed: false,
      productionMutation: false,
      publicationAuthority: false,
    },
  }, null, 2));
}

void main();
