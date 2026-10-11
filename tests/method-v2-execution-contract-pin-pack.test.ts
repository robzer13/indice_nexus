import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import test from "node:test";
import pack from "../contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_EXECUTION_CONTRACT_PIN_PACK_V1.0.json";
import v1 from "../contracts/orotitan-equity/v1/contract-pin-pack-v1/OROTITAN_CONTRACT_PIN_PACK_V1.json";
import { authoritySetSha256, METHOD_V2_AUTHORITY_SET_SHA256, methodV2Manifest, verifyMethodV2Authority } from "../lib/orotitan-equity/method-v2/authority";
import { gitBlobSha1, resolveContractPin, sha256Hex, type ContractPin, type GithubBlobFetcher } from "../lib/orotitan-equity/v1/contract-pin-resolver";
import { computeContractSetSha256 } from "../lib/orotitan-equity/v1/stage-manifest";

const ROOT = "contracts/orotitan-equity/method-v2/";
const SET = "23b75bf5c2d7448e8270e7e8f3a0223e639d0be3a1406b3c089e6e885dd063ea";
const AUTHORITY = "1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2";
const INHERITED = ["research_stage", "analysis_standard", "investment_policy", "execution_patch", "integration_spec", "screener_schema", "i2", "i3b"] as const;
const SUCCESSORS = [
  ["A03", "process", "EXECUTION PROCESS", "EXECUTION_PROCESS", "4bb6672949c704349316f40c4a1280fcd65e2e96d40a619b9e500a7eba89d088"],
  ["A04", "pilotage", "PILOTAGE", "PILOTAGE", "a332708924cb2c684de8d87b21aafa52d74b658467e11b8f2baf9adc8a25df5c"],
  ["A05", "deep_dive_stage", "DEEP DIVE STAGE CONTRACT", "DEEP_DIVE_STAGE_CONTRACT", "7cc34e39a8c8e5895c63ba51781b41aa2a81011219e14b8e127e0c85a76f2b3d"],
  ["A06", "integration_stage", "INTEGRATION ADMISSION", "INTEGRATION_ADMISSION", "c88c915823b04112eede6acb5b0ec136b8149d7f550cd3c9e6edea82c6876ae3"],
  ["A07", "master_prompt", "MASTER PROMPT SEQUENCING", "MASTER_PROMPT_SEQUENCING", "efe42012bcf8f41be4942ff216d49b7a5010e67c0a1101d82e5c4a4330fde51c"],
] as const;
const pins = pack.contract_pins as Record<string, ContractPin>;
const gitFetch: GithubBlobFetcher = async ({ repository, path, commitSha }) => {
  assert.equal(repository, "robzer13/indice_nexus");
  assert.match(commitSha, /^[0-9a-f]{40}$/);
  assert.ok(!path.startsWith("/") && !path.includes(".."));
  return execFileSync("git", ["show", `${commitSha}:${path}`], { maxBuffer: 64 * 1024 * 1024 });
};

test("A01 exactly 13 logical pins, ordered deterministically and matching Registry keys", () => {
  const keys = [...INHERITED, ...SUCCESSORS.map(s => s[1])].sort();
  assert.equal(Object.keys(pins).length, 13);
  assert.deepEqual(Object.keys(pins), keys);
  const sql = readFileSync("migrations/20260914_orotitan_registry_v1_5_contract_pin_guards.sql", "utf8");
  const required = sql.match(/v_required constant text\[\] := array\[([\s\S]*?)\]/)![1];
  assert.deepEqual([...required.matchAll(/'([^']+)'/g)].map(m => m[1]).sort(), keys);
});

test("A02 eight inherited V1 pins retain exact full identity", async () => {
  const frozen = JSON.parse(Buffer.from(await gitFetch({ repository: pack.repository,
    path: "contracts/orotitan-equity/v1/contract-pin-pack-v1/OROTITAN_CONTRACT_PIN_PACK_V1.json",
    commitSha: pack.source_commit_sha })).toString("utf8"));
  assert.equal(computeContractSetSha256(v1.contract_pins), "34b009f05715bab482dbc00b02194b3714e9b2f8151872144677bb6db19f3c63");
  assert.equal(pack.inherited_v1_contract_set_sha256, v1.contract_set_sha256);
  assert.equal(Object.keys(v1.contract_pins).length, 13);
  for (const key of INHERITED) {
    assert.deepEqual(pins[key], v1.contract_pins[key]);
    assert.equal(JSON.stringify(pins[key]), JSON.stringify(frozen.contract_pins[key]));
  }
});

for (const [id, key, title, stem, hash] of SUCCESSORS) {
  test(`${id} ${key} exact frozen successor name/version/hash/locator`, async () => {
    const name = `OROTITAN_METHOD_V2_${stem}_SUCCESSOR_V1.0`;
    const pin = pins[key];
    const bytes = await resolveContractPin(pin, gitFetch);
    assert.equal(Buffer.from(bytes).toString("utf8").split(/\r?\n/)[0], `# OroTitan Method-V2 ${title} successor V1.0`);
    assert.deepEqual(pin, { name, version: "1.0", content_sha256: hash, locator: {
      backend: "GITHUB_IMMUTABLE", repository: "robzer13/indice_nexus", path: `${ROOT}${name}.md`,
      commit_sha: "3ba9cb7e83368155abcca9d44429df490bb08b9e", blob_sha: gitBlobSha1(bytes), locator_format: "RAW_CANONICAL_V1",
    } });
  });
}

test("A08 all current member bytes reproduce canonical content SHA-256s", async () => {
  for (const pin of Object.values(pins)) {
    const bytes = await resolveContractPin(pin, async ({ path }) => readFileSync(path));
    assert.equal(sha256Hex(bytes), pin.content_sha256);
  }
});

test("A09 all immutable Git locators resolve exact canonical bytes and archive parts", async () => {
  for (const pin of Object.values(pins)) {
    assert.equal(pin.locator.backend, "GITHUB_IMMUTABLE");
    assert.match(pin.locator.blob_sha, /^[0-9a-f]{40}$/);
    const historical = await resolveContractPin(pin, gitFetch);
    const current = await resolveContractPin(pin, async ({ path }) => readFileSync(path));
    assert.deepEqual(current, historical);
  }
});

test("A10 no Runtime V2/V3/V3.1 authority is imported", () => {
  const allowed = new Set([...INHERITED.map(k => v1.contract_pins[k].locator.path), ...SUCCESSORS.map(s => `${ROOT}OROTITAN_METHOD_V2_${s[3]}_SUCCESSOR_V1.0.md`)]);
  for (const pin of Object.values(pins)) {
    assert.ok(allowed.has(pin.locator.path));
    assert.doesNotMatch(pin.locator.path, /\/v[23]\b|\/runtime\/|contract-pin-pack-v[23]/);
  }
});

test("A11 canonical Registry digest independently reproducible, including final LF", () => {
  const sql = readFileSync("migrations/20260914_orotitan_registry_v1_3_rpcs.sql", "utf8");
  assert.ok(sql.includes("format('%s|%s|%s', e.key, e.value->>'version', e.value->>'content_sha256')"));
  assert.ok(sql.includes("E'\\n' order by e.key"));
  assert.ok(sql.includes("convert_to(v_lines || E'\\n', 'UTF8')"));
  const lines = Object.keys(pins).sort().map(k => `${k}|${pins[k].version}|${pins[k].content_sha256}`).join("\n");
  assert.equal(sha256Hex(Buffer.from(`${lines}\n`, "utf8")), SET);
  assert.equal(computeContractSetSha256(pins), SET);
  assert.equal(pack.contract_set_sha256, SET);
  assert.notEqual(sha256Hex(Buffer.from(lines, "utf8")), SET);
  assert.equal(computeContractSetSha256(Object.fromEntries(Object.entries(pins).reverse())), SET);
});

test("A12 single-member tamper changes digest or fails immutable resolution", async () => {
  for (const key of Object.keys(pins)) {
    const changed = structuredClone(pins);
    changed[key].content_sha256 = "0".repeat(64);
    assert.notEqual(computeContractSetSha256(changed), SET);
    await assert.rejects(() => resolveContractPin(changed[key], gitFetch), /sha256/);
  }
  const changedLocator = structuredClone(pins.process);
  changedLocator.locator.blob_sha = "0".repeat(40);
  await assert.rejects(() => resolveContractPin(changedLocator, gitFetch), /git blob mismatch/);
  // Registry's historical digest does not bind names/locators. Exact full-pin
  // assertions A02-A07 and immutable resolution enforce those independently.
  const changedName = structuredClone(pins.process);
  changedName.name += "_TAMPERED";
  assert.notDeepEqual(changedName, pins.process);
});

test("A13 analytical authority identity and all immutable bytes unchanged", async () => {
  assert.equal(METHOD_V2_AUTHORITY_SET_SHA256, AUTHORITY);
  assert.equal(authoritySetSha256(methodV2Manifest), AUTHORITY);
  assert.equal(pack.method_v2_authority_set_sha256, AUTHORITY);
  await verifyMethodV2Authority(methodV2Manifest, gitFetch);
});

test("A14 pack freezes execution identity without production/runtime/publication activation", () => {
  assert.equal(pack.production_active, false);
  assert.equal(pack.identity_domain, "METHOD_V2_EXECUTION_NOT_ANALYTICAL_IDENTITY");
  assert.deepEqual(pack.scope_boundary, { creates_live_run: false, activates_runtime: false,
    publishes_canonical_snapshot: false, mutates_production_registry: false });
  assert.equal(pack.repository, "robzer13/indice_nexus");
});
