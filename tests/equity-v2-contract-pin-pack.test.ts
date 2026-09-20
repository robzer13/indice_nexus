import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { resolveContractPin, sha256Hex, type ContractPin } from "../lib/orotitan-equity/v1/contract-pin-resolver";
import { computeContractSetSha256 } from "../lib/orotitan-equity/v1/stage-manifest";

const REPOSITORY = "robzer13/indice_nexus";
const V2_SOURCE_COMMIT = "86b227a75275cf4aaec6ef61ea2a27e87e2bfec7";
const V1_SOURCE_COMMIT = "8aba7cee6a9b38204785c16976e65e9010f5d959";
const DCF_SOURCE_COMMIT = "003660a6b5984402d2a0cb8bc99fba6cdda9470b";
const V2_SET_SHA = "d933717b9da01e8565a3e8116ff77582ffa7ded109649ec370a0f7839eecc71a";
const V2_KEYS = ["process", "pilotage", "research_stage", "deep_dive_stage", "integration_stage", "integration_spec", "screener_schema"];
const V1_REUSED_KEYS = ["analysis_standard", "master_prompt", "investment_policy", "execution_patch", "i2", "i3b"];

async function fsFetcher({ path }: { repository: string; path: string; commitSha: string }): Promise<Uint8Array> {
  return readFile(path);
}

async function loadPack(path: string): Promise<{ format: string; version: string; repository: string; source_commit_sha: string; contract_set_sha256: string; contract_pins: Record<string, ContractPin> }> {
  return JSON.parse(await readFile(path, "utf8"));
}

test("Contract Pin Pack V2.0.1 contains exactly the frozen 14 logical Registry pins", async () => {
  const pack = await loadPack("contracts/orotitan-equity/v2/contract-pin-pack-v2/OROTITAN_CONTRACT_PIN_PACK_V2.json");
  assert.equal(pack.format, "OROTITAN_CONTRACT_PIN_PACK_V2");
  assert.equal(pack.version, "2.0.1");
  assert.equal(pack.repository, REPOSITORY);
  assert.equal(pack.source_commit_sha, DCF_SOURCE_COMMIT);
  assert.deepEqual(
    Object.keys(pack.contract_pins).sort(),
    [
      "analysis_standard", "dcf_timing", "deep_dive_stage", "execution_patch", "i2", "i3b",
      "integration_spec", "integration_stage", "investment_policy", "master_prompt",
      "pilotage", "process", "research_stage", "screener_schema",
    ],
  );
});

test("Contract Pin Pack V2 reconciles deterministically to the Registry contract-set SHA", async () => {
  const pack = await loadPack("contracts/orotitan-equity/v2/contract-pin-pack-v2/OROTITAN_CONTRACT_PIN_PACK_V2.json");
  assert.equal(computeContractSetSha256(pack.contract_pins), V2_SET_SHA);
  assert.equal(pack.contract_set_sha256, V2_SET_SHA);
});

test("V2 successor pins resolve from the exact green immutable source commit", async () => {
  const pack = await loadPack("contracts/orotitan-equity/v2/contract-pin-pack-v2/OROTITAN_CONTRACT_PIN_PACK_V2.json");
  for (const key of V2_KEYS) {
    const pin = pack.contract_pins[key];
    assert.equal(pin.locator.repository, REPOSITORY);
    assert.equal(pin.locator.commit_sha, V2_SOURCE_COMMIT, `${key} source commit mismatch`);
    const bytes = await resolveContractPin(pin, fsFetcher);
    assert.equal(sha256Hex(bytes), pin.content_sha256, `${key} canonical hash mismatch`);
  }
});

test("DCF timing authority resolves byte-for-byte from its immutable freeze commit", async () => {
  const pack = await loadPack("contracts/orotitan-equity/v2/contract-pin-pack-v2/OROTITAN_CONTRACT_PIN_PACK_V2.json");
  const pin = pack.contract_pins.dcf_timing;
  assert.equal(pin.locator.repository, REPOSITORY);
  assert.equal(pin.locator.commit_sha, DCF_SOURCE_COMMIT);
  const bytes = await resolveContractPin(pin, fsFetcher);
  assert.equal(sha256Hex(bytes), pin.content_sha256);
  assert.equal(pin.content_sha256, "e4f316fd15f9e1e07683dcc1fe1dbb89c0bcdd938724cf118ef24d213dada771");
});

test("unchanged analytical authorities are reused byte-for-byte under their original V1 immutable pins", async () => {
  const v2 = await loadPack("contracts/orotitan-equity/v2/contract-pin-pack-v2/OROTITAN_CONTRACT_PIN_PACK_V2.json");
  const v1 = await loadPack("contracts/orotitan-equity/v1/contract-pin-pack-v1/OROTITAN_CONTRACT_PIN_PACK_V1.json");
  for (const key of V1_REUSED_KEYS) {
    assert.deepEqual(v2.contract_pins[key], v1.contract_pins[key], `${key} must be exact V1 reuse`);
    assert.equal(v2.contract_pins[key].locator.commit_sha, V1_SOURCE_COMMIT);
    const bytes = await resolveContractPin(v2.contract_pins[key], fsFetcher);
    assert.equal(sha256Hex(bytes), v2.contract_pins[key].content_sha256);
  }
});

test("all 14 V2 run authorities resolve byte-for-byte and no V1 run pin object is mutated", async () => {
  const v2 = await loadPack("contracts/orotitan-equity/v2/contract-pin-pack-v2/OROTITAN_CONTRACT_PIN_PACK_V2.json");
  const v1 = await loadPack("contracts/orotitan-equity/v1/contract-pin-pack-v1/OROTITAN_CONTRACT_PIN_PACK_V1.json");
  for (const [key, pin] of Object.entries(v2.contract_pins)) {
    const bytes = await resolveContractPin(pin, fsFetcher);
    assert.equal(sha256Hex(bytes), pin.content_sha256, `${key} canonical hash mismatch`);
  }
  assert.equal(v1.contract_set_sha256, "34b009f05715bab482dbc00b02194b3714e9b2f8151872144677bb6db19f3c63");
  assert.equal(v1.source_commit_sha, V1_SOURCE_COMMIT);
});
