import { execFileSync, spawn } from "node:child_process";

const READY_PREFIX = "OROTITAN_SLEEP_GUARD_READY:";
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

async function main(): Promise<void> {
  if (process.platform !== "win32") {
    console.log(JSON.stringify({
      format: "OROTITAN_GATE18_PHASE_C_C4_WINDOWS_SLEEP_GUARD_SMOKE_V0.1",
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
    "$ES_CONTINUOUS = [uint32]0x80000000",
    "$ES_SYSTEM_REQUIRED = [uint32]0x00000001",
    "$flags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED",
    "$result = [OroTitanExecutionState]::SetThreadExecutionState($flags)",
    "if($result -eq 0){ throw 'SET_THREAD_EXECUTION_STATE_FAILED' }",
    `Write-Output "${READY_PREFIX}$result"`,
    "[Console]::Out.Flush()",
    "while($true){ Start-Sleep -Seconds 1 }",
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

  helper.stdout.setEncoding("utf8");
  helper.stderr.setEncoding("utf8");
  helper.stdout.on("data", (chunk: string) => {
    helperStdout += chunk;
    const line = helperStdout
      .split(/\r?\n/)
      .find((item) => item.startsWith(READY_PREFIX));
    if (line) readyValue = line.slice(READY_PREFIX.length).trim();
  });
  helper.stderr.on("data", (chunk: string) => {
    helperStderr += chunk;
  });

  const startedAt = Date.now();
  while (readyValue === null && Date.now() - startedAt < 10_000) {
    if (helper.exitCode !== null) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  const during = runText("powercfg.exe", ["/requests"]);

  await new Promise((resolve) => setTimeout(resolve, HOLD_MS));

  let terminatedCleanly = false;
  try {
    terminatedCleanly = helper.kill();
  } catch {
    terminatedCleanly = false;
  }

  await new Promise<void>((resolve) => {
    if (helper.exitCode !== null) {
      resolve();
      return;
    }
    const timer = setTimeout(() => resolve(), 5_000);
    helper.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });

  if (helper.exitCode === null && helper.pid) {
    runText("taskkill.exe", ["/PID", String(helper.pid), "/T", "/F"]);
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  const after = runText("powercfg.exe", ["/requests"]);

  console.log(JSON.stringify({
    format: "OROTITAN_GATE18_PHASE_C_C4_WINDOWS_SLEEP_GUARD_SMOKE_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C4_RUNTIME_ENVIRONMENT_REMEDIATION",
    status:
      readyValue !== null && helper.exitCode !== null
        ? "PASS_PROCESS_SCOPED_GUARD_STARTED_AND_RELEASED"
        : "REVIEW_REQUIRED",
    mode: "PROCESS_SCOPED_POWER_REQUEST_ONLY_NO_INFERENCE",
    helper: {
      readyObserved: readyValue !== null,
      setThreadExecutionStatePriorFlagsRaw: readyValue,
      pid: helper.pid ?? null,
      stderr: helperStderr.trim() || null,
      terminatedCleanly,
      exitCode: helper.exitCode,
    },
    powercfgRequests: {
      before,
      during,
      after,
    },
    interpretationBoundary: {
      idleSleepPreventionMechanismInvoked: readyValue !== null,
      processScopedGuardReleaseObserved: helper.exitCode !== null,
      lidOrManualSleepPreventionConcluded: false,
      actualSleepPreventionProven: false,
      c4InferenceAuthorized: false,
      nextAction:
        "Human review of process-scoped execution-state guard behavior before integrating it into any inference runner.",
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
