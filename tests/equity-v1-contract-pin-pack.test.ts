import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { gitBlobSha1, resolveContractPin, sha256Hex, type ContractPin } from "../lib/orotitan-equity/v1/contract-pin-resolver";
import { computeContractSetSha256 } from "../lib/orotitan-equity/v1/stage-manifest";

const COMMIT = "a".repeat(40);
const REPOSITORY = "robzer13/indice_nexus";

const sourceAuthorities = {
  investment_policy: "contracts/orotitan-equity/v1/OROTITAN_INVESTMENT_POLICY_V1.0.0.md",
  execution_patch: "contracts/orotitan-equity/v1/OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1.md",
  integration_spec: "contracts/orotitan-equity/v1/04_INTEGRATION_SPEC_V1_PATCHED.md",
  screener_schema: "contracts/orotitan-equity/v1/04_SCREENER_SCHEMA_V1_PATCHED.json",
  i2: "docs/orotitan-equity/I2_CANONICAL_COMPUTATION.md",
  i3b: "docs/orotitan-equity/I3B_VALIDATED_SNAPSHOT_WRITER.md",
} as const;

const knownBlobShas = {
  investment_policy: "fd0121bbb14e35370ddd700d9e9845eaf1ec9f75",
  execution_patch: "449bb43b664eae9d8c9fd98440e5511c6fbf4be3",
  integration_spec: "e41aee33c714128acacc1342af0949d4d38b6442",
  screener_schema: "22e13b5fb058371eca863613a5f1ac8e6582da00",
  i2: "93633adec3409144fcb03d5ec7b17ecfe22ff736",
  i3b: "65f5c0c013362fdd7ba64a8975a8d25b74d408d6",
} as const;

const multipart = {
  analysis_standard: {
    indexPath: "contracts/orotitan-equity/v1/contract-pin-pack-v1/archives/analysis_standard/01_ANALYSIS_STANDARD_V1.archive.json",
    indexBlob: "4702be68ca6f08138add2b10773140572fdd1c71",
    canonicalSha256: "9283a0df395d4596c93cf6cfad9644ce114b343ee80e6e0a81e8f94f15d1e3df",
  },
  master_prompt: {
    indexPath: "contracts/orotitan-equity/v1/contract-pin-pack-v1/archives/master_prompt/02_OROTITAN_MASTER_PROMPT_V1_PATCHED.archive.json",
    indexBlob: "d63b60bc0c2974b2fcdb8fd4751a5b84d1f3c01a",
    canonicalSha256: "d3b44f0645f766d504a0b937736520048a054e533df59366c90554e5345a0dab",
  },
  deep_dive_stage: {
    indexPath: "contracts/orotitan-equity/v1/contract-pin-pack-v1/archives/deep_dive_stage/OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V1_FREEZE_V1.0.archive.json",
    indexBlob: "aabad6d1aff5f5e08fca091df0c257b1e3834e96",
    canonicalSha256: "21586e572739e2564781ce2814758d3d557c151fee5866d8c75cb0394f9b82c5",
  },
} as const;

async function fsFetcher({ path }: { repository: string; path: string; commitSha: string }): Promise<Uint8Array> {
  return readFile(path);
}

test("source authorities have the exact Git blob identities expected on the locked baseline", async () => {
  const report: Record<string, { path: string; blob_sha: string; content_sha256: string; size_bytes: number }> = {};
  for (const [key, path] of Object.entries(sourceAuthorities) as Array<[keyof typeof sourceAuthorities, string]>) {
    const bytes = await readFile(path);
    const blobSha = gitBlobSha1(bytes);
    assert.equal(blobSha, knownBlobShas[key], `${key} Git blob drift`);
    report[key] = { path, blob_sha: blobSha, content_sha256: sha256Hex(bytes), size_bytes: bytes.byteLength };
  }
  assert.equal(report.integration_spec.content_sha256, "cac78e505d354a124f5fdddb726baf31ecf7f0b6b90653fe578a5c5cca9a8238");
  assert.equal(report.screener_schema.content_sha256, "bf407ca217553521586ba5f6002180ff6522700b4671986079ea6ed577604ede");
  console.log(`OROTITAN_PIN_SOURCE_HASHES=${JSON.stringify(report)}`);
});

test("multipart resolver reconstructs and verifies the three archived authorities byte-for-byte", async () => {
  for (const [key, entry] of Object.entries(multipart)) {
    const pin: ContractPin = {
      name: key,
      version: "1.0",
      content_sha256: entry.canonicalSha256,
      locator: {
        backend: "GITHUB_IMMUTABLE",
        repository: REPOSITORY,
        path: entry.indexPath,
        commit_sha: COMMIT,
        blob_sha: entry.indexBlob,
        locator_format: "OROTITAN_MULTIPART_GZIP_V1",
      },
    };
    const bytes = await resolveContractPin(pin, fsFetcher);
    assert.equal(sha256Hex(bytes), entry.canonicalSha256);
  }
});

test("resolver fails closed on a pin hash mismatch", async () => {
  const entry = multipart.analysis_standard;
  const pin: ContractPin = {
    name: "analysis_standard",
    version: "1.0",
    content_sha256: "0".repeat(64),
    locator: {
      backend: "GITHUB_IMMUTABLE",
      repository: REPOSITORY,
      path: entry.indexPath,
      commit_sha: COMMIT,
      blob_sha: entry.indexBlob,
      locator_format: "OROTITAN_MULTIPART_GZIP_V1",
    },
  };
  await assert.rejects(() => resolveContractPin(pin, fsFetcher), /canonical sha256 does not match pin/);
});

test("resolver fails closed on a Git blob mismatch", async () => {
  const path = sourceAuthorities.investment_policy;
  const bytes = await readFile(path);
  const pin: ContractPin = {
    name: "investment_policy",
    version: "1.0.0",
    content_sha256: sha256Hex(bytes),
    locator: {
      backend: "GITHUB_IMMUTABLE",
      repository: REPOSITORY,
      path,
      commit_sha: COMMIT,
      blob_sha: "0".repeat(40),
      locator_format: "RAW_CANONICAL_V1",
    },
  };
  await assert.rejects(() => resolveContractPin(pin, fsFetcher), /git blob mismatch/);
});

test("final Contract Pin Pack V1 contains exactly 13 pins and reconciles to the Registry contract-set hash", async () => {
  const raw = await readFile("contracts/orotitan-equity/v1/contract-pin-pack-v1/OROTITAN_CONTRACT_PIN_PACK_V1.json", "utf8");
  const pack = JSON.parse(raw) as {
    format: string;
    version: string;
    repository: string;
    source_commit_sha: string;
    contract_set_sha256: string;
    contract_pins: Record<string, ContractPin>;
  };

  assert.equal(pack.format, "OROTITAN_CONTRACT_PIN_PACK_V1");
  assert.equal(pack.version, "1.0");
  assert.equal(pack.repository, REPOSITORY);
  assert.equal(pack.source_commit_sha, "8aba7cee6a9b38204785c16976e65e9010f5d959");
  assert.deepEqual(
    Object.keys(pack.contract_pins).sort(),
    [
      "analysis_standard", "deep_dive_stage", "execution_patch", "i2", "i3b",
      "integration_spec", "integration_stage", "investment_policy", "master_prompt",
      "pilotage", "process", "research_stage", "screener_schema",
    ],
  );
  assert.equal(
    computeContractSetSha256(pack.contract_pins),
    "34b009f05715bab482dbc00b02194b3714e9b2f8151872144677bb6db19f3c63",
  );
  assert.equal(pack.contract_set_sha256, "34b009f05715bab482dbc00b02194b3714e9b2f8151872144677bb6db19f3c63");
  for (const pin of Object.values(pack.contract_pins)) {
    assert.equal(pin.locator.repository, REPOSITORY);
    assert.equal(pin.locator.commit_sha, pack.source_commit_sha);
  }
});

test("final Contract Pin Pack V1 resolves all 13 authorities byte-for-byte", async () => {
  const raw = await readFile("contracts/orotitan-equity/v1/contract-pin-pack-v1/OROTITAN_CONTRACT_PIN_PACK_V1.json", "utf8");
  const pack = JSON.parse(raw) as { contract_pins: Record<string, ContractPin> };

  for (const [logicalName, pin] of Object.entries(pack.contract_pins)) {
    const bytes = await resolveContractPin(pin, fsFetcher);
    assert.equal(sha256Hex(bytes), pin.content_sha256, `${logicalName} canonical hash mismatch`);
  }
});
