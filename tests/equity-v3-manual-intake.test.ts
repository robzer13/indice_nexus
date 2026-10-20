import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  IntakeError, MANUAL_INTAKE_BUCKET, MANUAL_INTAKE_PREFIX,
  MANUAL_INTAKE_MAX_BYTES, intakeObjectPath, intakePrefix, isRunId,
  parseIntakeObject, validateIntakeBytes,
} from "../lib/orotitan-equity/v3/manual-intake-v3-1";

const RUN = "48b9426b-4d0d-4858-9d87-efcd13ac415d";
const bytes = (s: string) => new TextEncoder().encode(s);

test("intake content is retained only in an isolated quarantine namespace", () => {
  const file = validateIntakeBytes("final_research.json", bytes('{"evidence":"unverified"}'));
  const path = intakeObjectPath(RUN, "RESEARCH", file);
  assert.equal(isRunId(RUN), true);
  assert.equal(file.sizeBytes, 25);
  assert.equal(file.contentType, "application/json");
  assert.match(file.sha256, /^[0-9a-f]{64}$/);
  assert.equal(MANUAL_INTAKE_BUCKET, "orotitan-text-artifacts-v1");
  assert.equal(MANUAL_INTAKE_PREFIX, "manual-intake/v1");
  assert.equal(intakePrefix(RUN, "RESEARCH"), `manual-intake/v1/${RUN}/RESEARCH`);
  assert.equal(path, `manual-intake/v1/${RUN}/RESEARCH/${file.sha256}__final_research.json`);
  const listing = parseIntakeObject({name: path.split("/").at(-1)!, metadata:{size:file.sizeBytes}}, "RESEARCH");
  assert.deepEqual(listing, {filename:"final_research.json",sha256:file.sha256,sizeBytes:file.sizeBytes,
    stage:"RESEARCH",status:"RECEIVED_NOT_VERIFIED"});
});

test("intake rejects unsafe inputs, unsupported files and invalid Registry scopes", () => {
  function rejects(fn: () => unknown, code: string) {
    assert.throws(fn, (error: unknown) => error instanceof IntakeError && error.code === code);
  }
  rejects(() => intakePrefix("../BESI", "RESEARCH"), "INTAKE_IDENTITY_INVALID");
  rejects(() => intakePrefix(RUN, "VALUATION"), "INTAKE_IDENTITY_INVALID");
  rejects(() => validateIntakeBytes("results.pdf", bytes("data")), "INTAKE_FILETYPE_NOT_ALLOWED");
  rejects(() => validateIntakeBytes("results.json", bytes("[]")), "INTAKE_JSON_INVALID");
  rejects(() => validateIntakeBytes("results.json", bytes("{no}")), "INTAKE_JSON_INVALID");
  rejects(() => validateIntakeBytes("notes.md", new Uint8Array([0xff])), "INTAKE_NOT_UTF8_TEXT");
  rejects(() => validateIntakeBytes("notes.txt", bytes("abc\0xyz")), "INTAKE_NOT_UTF8_TEXT");
  rejects(() => validateIntakeBytes("notes.txt", new Uint8Array()), "INTAKE_FILE_SIZE_INVALID");
  rejects(() => validateIntakeBytes("notes.txt", new Uint8Array(MANUAL_INTAKE_MAX_BYTES + 1)), "INTAKE_FILE_SIZE_INVALID");
  rejects(() => validateIntakeBytes("x".repeat(200) + ".md", bytes("abc")), "INTAKE_FILENAME_INVALID");
  assert.equal(parseIntakeObject({name:"fake__input.json"},"RESEARCH"),null);
});

test("intake sanitizes filenames without allowing path escapes", () => {
  const file = validateIntakeBytes("../../hello world!.md", bytes("hello"));
  assert.equal(file.fileName,"hello_world_.md");
  assert.match(intakeObjectPath(RUN,"RESEARCH",file), /RESEARCH\/[0-9a-f]{64}__hello_world_\.md$/);
});

test("manual intake route requires admin, origin and production flag and never calls Registry writers", () => {
  const route = readFileSync(new URL("../app/api/orotitan/manual-intake/route.ts",import.meta.url),"utf8");
  const component = readFileSync(new URL("../components/admin/orotitan-manual-intake.tsx",import.meta.url),"utf8");
  const page = readFileSync(new URL("../app/admin/orotitan/page.tsx",import.meta.url),"utf8");
  assert.match(route,/requireAdminSession/);
  assert.match(route,/assertProductionUrl/);
  assert.match(route,/OROTITAN_MANUAL_INTAKE_ENABLED/);
  assert.match(route,/request\.headers\.get\("origin"\)/);
  assert.match(route,/upsert: false/);
  assert.match(route,/findRun\(client, runId\)/);
  assert.doesNotMatch(route,/\.rpc\s*\(|\.insert\s*\(|\.update\s*\(|\.delete\s*\(/);
  assert.doesNotMatch(route,/register_manifest|attest_orotitan|start_orotitan_stage|finalize_orotitan_stage/);
  assert.match(component,/RECEIVED_NOT_VERIFIED/);
  assert.match(component,/Déposer sans valider/);
  assert.match(page,/OroTitanManualIntake/);
});
