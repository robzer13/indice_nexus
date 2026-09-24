import { execFileSync } from "node:child_process";

const EXPECTED_GPU = "NVIDIA GeForce RTX 3050 Laptop GPU";
const CURRENT_DRIVER = "532.03";
const TARGET_DRIVER = "617.14";
const MINIMUM_DRIVER = "551.61";

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

function powershellJson(script: string): unknown {
  const output = runText(
    "powershell.exe",
    [
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy",
      "Bypass",
      "-Command",
      script,
    ],
    30_000,
  );

  if (!output) {
    return null;
  }

  try {
    return JSON.parse(output);
  } catch {
    return output;
  }
}

function versionParts(value: string | null): number[] {
  if (!value) return [];
  const match = value.match(/\d+(?:\.\d+)+/);
  return match
    ? match[0].split(".").map((part) => Number(part))
    : [];
}

function compareVersions(
  left: string | null,
  right: string,
): number | null {
  const a = versionParts(left);
  const b = versionParts(right);
  if (a.length === 0 || b.length === 0) return null;

  const length = Math.max(a.length, b.length);
  for (let index = 0; index < length; index += 1) {
    const av = a[index] ?? 0;
    const bv = b[index] ?? 0;
    if (av > bv) return 1;
    if (av < bv) return -1;
  }
  return 0;
}

function normalizeArray<T>(value: T | T[] | null): T[] {
  if (value === null) return [];
  return Array.isArray(value) ? value : [value];
}

async function main() {
  const computerSystem = powershellJson(
    `Get-CimInstance Win32_ComputerSystem |
      Select-Object Manufacturer,Model,SystemType,PCSystemType |
      ConvertTo-Json -Compress`,
  );

  const operatingSystem = powershellJson(
    `Get-CimInstance Win32_OperatingSystem |
      Select-Object Caption,Version,BuildNumber,OSArchitecture |
      ConvertTo-Json -Compress`,
  );

  const videoControllers = normalizeArray(
    powershellJson(
      `Get-CimInstance Win32_VideoController |
        Select-Object Name,PNPDeviceID,DriverVersion,AdapterRAM,VideoProcessor,Status |
        ConvertTo-Json -Compress`,
    ),
  );

  const signedDrivers = normalizeArray(
    powershellJson(
      `Get-CimInstance Win32_PnPSignedDriver |
        Where-Object {
          $_.DeviceName -like '*NVIDIA*' -or
          $_.DeviceClass -eq 'DISPLAY'
        } |
        Select-Object DeviceName,DeviceID,DriverVersion,DriverDate,InfName,Manufacturer,Signer |
        ConvertTo-Json -Compress`,
    ),
  );

  const pendingReboot = powershellJson(
    `$pending = [ordered]@{
      ComponentBasedServicing = Test-Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Component Based Servicing\\RebootPending'
      WindowsUpdate = Test-Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\WindowsUpdate\\Auto Update\\RebootRequired'
      PendingFileRenameOperations = $null -ne (Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager' -Name PendingFileRenameOperations -ErrorAction SilentlyContinue).PendingFileRenameOperations
    }
    [pscustomobject]$pending | ConvertTo-Json -Compress`,
  );

  const nvidiaSmi = runText("nvidia-smi", [
    "--query-gpu=name,driver_version,pci.bus_id,memory.total",
    "--format=csv,noheader,nounits",
  ]);

  const nvidiaRow =
    nvidiaSmi?.split(/\r?\n/)[0]?.trim() ?? null;
  const nvidiaFields =
    nvidiaRow?.split(",").map((part) => part.trim()) ?? [];

  const observedGpuName = nvidiaFields[0] ?? null;
  const observedDriver = nvidiaFields[1] ?? null;

  const exactGpuName =
    observedGpuName === EXPECTED_GPU ||
    videoControllers.some(
      (item: any) => item?.Name === EXPECTED_GPU,
    );

  const belowMinimum =
    compareVersions(observedDriver, MINIMUM_DRIVER) === -1;
  const belowTarget =
    compareVersions(observedDriver, TARGET_DRIVER) === -1;

  const osCaption =
    operatingSystem &&
    typeof operatingSystem === "object" &&
    !Array.isArray(operatingSystem)
      ? String((operatingSystem as any).Caption ?? "")
      : "";

  const windowsSupported =
    /Windows 10|Windows 11/i.test(osCaption);

  const status =
    exactGpuName && windowsSupported && belowTarget
      ? "PASS_HARDWARE_IDENTIFIED_REVIEW_OEM"
      : "BLOCKED";

  const payload = {
    format:
      "OROTITAN_GATE18_PHASE_C_NVIDIA_DRIVER_UPGRADE_PREFLIGHT_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C3_RUNTIME_REMEDIATION",
    status,
    mode: "LOCAL_HARDWARE_IDENTIFICATION_ONLY",
    target: {
      gpu: EXPECTED_GPU,
      currentKnownDriver: CURRENT_DRIVER,
      minimumQualificationDriver: MINIMUM_DRIVER,
      targetDriver: TARGET_DRIVER,
      targetCertification: "WHQL",
    },
    system: {
      computerSystem,
      operatingSystem,
      pendingReboot,
    },
    gpu: {
      nvidiaSmiRaw: nvidiaRow,
      observedGpuName,
      observedDriverVersion: observedDriver,
      pciBusId: nvidiaFields[2] ?? null,
      memoryTotalMiB:
        nvidiaFields[3] !== undefined
          ? Number(nvidiaFields[3])
          : null,
      videoControllers,
      signedDrivers,
    },
    validation: {
      exactGpuName,
      windowsSupported,
      belowMinimumQualification: belowMinimum,
      belowTargetDriver: belowTarget,
    },
    safety: {
      networkAccessRequested: false,
      driverDownloaded: false,
      driverInstalled: false,
      currentDriverUninstalled: false,
      dduExecuted: false,
      biosModified: false,
      modelInferenceExecuted: false,
      productionMutation: false,
      publicationAuthority: false,
    },
    nextAction:
      status === "PASS_HARDWARE_IDENTIFIED_REVIEW_OEM"
        ? "Review system manufacturer/model and exact PNPDeviceID against OEM/NVIDIA support before downloading the driver."
        : "Do not download or install a driver until the blocked hardware/OS condition is resolved.",
  };

  console.log(JSON.stringify(payload, null, 2));

  if (status === "BLOCKED") {
    process.exitCode = 1;
  }
}

void main().catch((error: unknown) => {
  const message =
    error instanceof Error ? error.message : String(error);

  console.error(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_NVIDIA_DRIVER_UPGRADE_PREFLIGHT_V0.1",
        status: "FAIL",
        mode: "LOCAL_HARDWARE_IDENTIFICATION_ONLY",
        driverDownloaded: false,
        driverInstalled: false,
        modelInferenceExecuted: false,
        productionMutation: false,
        publicationAuthority: false,
        error: message,
      },
      null,
      2,
    ),
  );

  process.exitCode = 1;
});
