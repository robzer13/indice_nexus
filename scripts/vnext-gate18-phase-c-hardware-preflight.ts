import { execFileSync } from "node:child_process";
import os from "node:os";

interface CommandProbe {
  command: string;
  available: boolean;
  path: string | null;
}

interface NvidiaGpu {
  name: string;
  memoryTotalMiB: number | null;
  driverVersion: string | null;
}

interface WindowsAdapter {
  name: string | null;
  adapterRamBytes: number | null;
  driverVersion: string | null;
}

function runText(
  command: string,
  args: readonly string[],
): string | null {
  try {
    return execFileSync(command, [...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 10_000,
      maxBuffer: 2 * 1024 * 1024,
    }).trim();
  } catch {
    return null;
  }
}

function commandPath(command: string): string | null {
  const locator = process.platform === "win32" ? "where.exe" : "which";
  const output = runText(locator, [command]);
  if (!output) {
    return null;
  }
  return output.split(/\r?\n/)[0]?.trim() || null;
}

function probeCommand(command: string): CommandProbe {
  const path = commandPath(command);
  return {
    command,
    available: path !== null,
    path,
  };
}

function parseNvidiaSmi(): NvidiaGpu[] {
  const output = runText("nvidia-smi", [
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
  ]);
  if (!output) {
    return [];
  }

  return output
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [nameRaw, memoryRaw, driverRaw] = line.split(",").map((part) => part.trim());
      const memory = Number(memoryRaw);
      return {
        name: nameRaw || "UNKNOWN",
        memoryTotalMiB: Number.isFinite(memory) ? memory : null,
        driverVersion: driverRaw || null,
      };
    });
}

function parseWindowsAdapters(): WindowsAdapter[] {
  if (process.platform !== "win32") {
    return [];
  }

  const ps = commandPath("powershell.exe") ?? commandPath("pwsh.exe");
  if (!ps) {
    return [];
  }

  const command = [
    "Get-CimInstance Win32_VideoController",
    "| Select-Object Name,AdapterRAM,DriverVersion",
    "| ConvertTo-Json -Compress",
  ].join(" ");

  const output = runText(ps, ["-NoProfile", "-Command", command]);
  if (!output) {
    return [];
  }

  try {
    const parsed = JSON.parse(output) as unknown;
    const rows = Array.isArray(parsed) ? parsed : [parsed];
    return rows
      .filter(
        (row): row is Record<string, unknown> =>
          row !== null && typeof row === "object" && !Array.isArray(row),
      )
      .map((row) => {
        const rawRam = row.AdapterRAM;
        const ram =
          typeof rawRam === "number"
            ? rawRam
            : typeof rawRam === "string"
              ? Number(rawRam)
              : NaN;

        return {
          name: typeof row.Name === "string" ? row.Name : null,
          adapterRamBytes: Number.isFinite(ram) ? ram : null,
          driverVersion:
            typeof row.DriverVersion === "string"
              ? row.DriverVersion
              : null,
        };
      });
  } catch {
    return [];
  }
}

function gib(bytes: number): number {
  return Math.round((bytes / 1024 ** 3) * 100) / 100;
}

const cpus = os.cpus();
const nvidia = parseNvidiaSmi();
const windowsAdapters = parseWindowsAdapters();

const runtimeCommands = [
  "ollama",
  "lms",
  "llama-server",
  "llama-cli",
  "python",
  "python3",
  "docker",
].map(probeCommand);

const payload = {
  format: "OROTITAN_GATE18_PHASE_C_C0_HARDWARE_PREFLIGHT_V0.1",
  gate: 18,
  phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
  stage: "C0_HARDWARE_AND_RUNTIME_QUALIFICATION",
  mode: "LOCAL_PREFLIGHT_ONLY",
  externalModelApiCall: false,
  modelInferenceExecuted: false,
  modelDownloadExecuted: false,
  productionMutation: false,
  publicationAuthority: false,
  system: {
    platform: process.platform,
    release: os.release(),
    arch: process.arch,
    hostnameRedacted: true,
    cpuModel: cpus[0]?.model ?? null,
    logicalCpuCount: cpus.length,
    totalRamGiB: gib(os.totalmem()),
    freeRamGiBAtProbe: gib(os.freemem()),
  },
  gpu: {
    nvidiaSmiAvailable: commandPath("nvidia-smi") !== null,
    nvidia,
    windowsAdapterFallback: windowsAdapters,
    note:
      "Windows AdapterRAM may be incomplete on some drivers; prefer nvidia-smi VRAM when available.",
  },
  localRuntimeCommands: runtimeCommands,
  interpretationBoundary: {
    candidateAdmitted: false,
    runtimeAdmitted: false,
    hardwareFitFrozen: false,
    nextAction:
      "Human review of this receipt before any model download or inference.",
  },
};

console.log(JSON.stringify(payload, null, 2));
