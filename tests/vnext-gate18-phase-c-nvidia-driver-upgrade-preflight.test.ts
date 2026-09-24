import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-nvidia-driver-upgrade-preflight.ts",
  "utf8",
);

test("NVIDIA driver preflight cannot download or install drivers", () => {
  assert.match(source, /LOCAL_HARDWARE_IDENTIFICATION_ONLY/);
  assert.match(source, /driverDownloaded: false/);
  assert.match(source, /driverInstalled: false/);
  assert.match(source, /currentDriverUninstalled: false/);
  assert.match(source, /dduExecuted: false/);
  assert.doesNotMatch(source, /Invoke-WebRequest|Start-BitsTransfer|curl\.exe|wget/i);
  assert.doesNotMatch(source, /pnputil.*add-driver/i);
});

test("NVIDIA driver preflight pins the expected GPU and driver target", () => {
  assert.match(source, /NVIDIA GeForce RTX 3050 Laptop GPU/);
  assert.match(source, /TARGET_DRIVER = "617\.14"/);
  assert.match(source, /MINIMUM_DRIVER = "551\.61"/);
});

test("NVIDIA driver preflight captures OEM and device identity", () => {
  assert.match(source, /Win32_ComputerSystem/);
  assert.match(source, /Manufacturer,Model/);
  assert.match(source, /Win32_VideoController/);
  assert.match(source, /PNPDeviceID/);
  assert.match(source, /Win32_PnPSignedDriver/);
});

test("NVIDIA driver preflight checks reboot state without modifying it", () => {
  assert.match(source, /RebootPending/);
  assert.match(source, /RebootRequired/);
  assert.match(source, /PendingFileRenameOperations/);
});

test("NVIDIA driver preflight cannot authorize model inference", () => {
  assert.match(source, /modelInferenceExecuted: false/);
  assert.match(source, /productionMutation: false/);
  assert.match(source, /publicationAuthority: false/);
});
