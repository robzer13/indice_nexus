import { spawn } from "node:child_process";

const READY_PREFIX = "OROTITAN_SLEEP_GUARD_READY:";
const RELEASE_PREFIX = "OROTITAN_SLEEP_GUARD_RELEASED:";

export interface WindowsSleepGuardTelemetry {
  readyObserved: boolean;
  releasedObserved: boolean;
  acquirePriorFlagsRaw: string | null;
  releasePriorFlagsRaw: string | null;
  helperPid: number | null;
  helperExitCode: number | null;
  helperStderr: string | null;
  forcedTermination: boolean;
}

export interface WindowsSleepGuardHandle {
  telemetry: WindowsSleepGuardTelemetry;
  release: () => Promise<WindowsSleepGuardTelemetry>;
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

export async function acquireWindowsSystemRequiredGuard(): Promise<WindowsSleepGuardHandle> {
  if (process.platform !== "win32") {
    throw new Error("VNEXT_WINDOWS_SLEEP_GUARD_WINDOWS_REQUIRED");
  }

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
    "$null = [Console]::In.ReadLine()",
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
      stdio: ["pipe", "pipe", "pipe"],
    },
  );

  let stdout = "";
  let stderr = "";
  let readyValue: string | null = null;
  let releaseValue: string | null = null;
  let released = false;

  helper.stdout.setEncoding("utf8");
  helper.stderr.setEncoding("utf8");

  helper.stdout.on("data", (chunk: string) => {
    stdout += chunk;
    for (const line of stdout.split(/\r?\n/)) {
      if (line.startsWith(READY_PREFIX)) {
        readyValue = line.slice(READY_PREFIX.length).trim();
      }
      if (line.startsWith(RELEASE_PREFIX)) {
        releaseValue = line.slice(RELEASE_PREFIX.length).trim();
      }
    }
  });

  helper.stderr.on("data", (chunk: string) => {
    stderr += chunk;
  });

  const ready = await waitUntil(
    () => readyValue !== null || helper.exitCode !== null,
    10_000,
  );

  if (!ready || readyValue === null) {
    if (helper.pid && helper.exitCode === null) {
      helper.kill();
    }
    throw new Error(
      `VNEXT_WINDOWS_SLEEP_GUARD_ACQUIRE_FAILED:${stderr.trim() || "NO_READY_MARKER"}`,
    );
  }

  const telemetry: WindowsSleepGuardTelemetry = {
    readyObserved: true,
    releasedObserved: false,
    acquirePriorFlagsRaw: readyValue,
    releasePriorFlagsRaw: null,
    helperPid: helper.pid ?? null,
    helperExitCode: helper.exitCode,
    helperStderr: null,
    forcedTermination: false,
  };

  return {
    telemetry,
    release: async () => {
      if (released) {
        return telemetry;
      }
      released = true;

      helper.stdin.write("\n");
      helper.stdin.end();

      await waitUntil(
        () => helper.exitCode !== null,
        10_000,
      );

      if (helper.exitCode === null && helper.pid) {
        telemetry.forcedTermination = true;
        helper.kill();
        await waitUntil(() => helper.exitCode !== null, 5_000);
      }

      telemetry.releasedObserved = releaseValue !== null;
      telemetry.releasePriorFlagsRaw = releaseValue;
      telemetry.helperExitCode = helper.exitCode;
      telemetry.helperStderr = stderr.trim() || null;

      if (
        !telemetry.releasedObserved ||
        telemetry.helperExitCode !== 0 ||
        telemetry.forcedTermination
      ) {
        throw new Error(
          `VNEXT_WINDOWS_SLEEP_GUARD_RELEASE_FAILED:${telemetry.helperStderr || "NO_RELEASE_MARKER"}`,
        );
      }

      return telemetry;
    },
  };
}
