import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";

const authorities = {
  process: {
    path: "contracts/orotitan-equity/v2/OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0.md",
    blob_sha: "b69c61383fbc12780df0b727552db5dd780f4661",
  },
  pilotage: {
    path: "contracts/orotitan-equity/v2/OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V2_FREEZE_V2.0.md",
    blob_sha: "4e605c2c14d5eaab80c84fd8173e69b963fe1963",
  },
  research_stage: {
    path: "contracts/orotitan-equity/v2/OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0.md",
    blob_sha: "4b49d4ab670717d844ff0ed3c925a2004c043dfa",
  },
  deep_dive_stage: {
    path: "contracts/orotitan-equity/v2/OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2_FREEZE_V2.0.md",
    blob_sha: "ef100df9d2efbd27605e6df801b48caeadd73275",
  },
  integration_stage: {
    path: "contracts/orotitan-equity/v2/OROTITAN_INTEGRATION_STAGE_CONTRACT_V2_FREEZE_V2.0.md",
    blob_sha: "7cd29e1535bfacc125da3583e8ea5c108a35351e",
  },
  integration_spec: {
    path: "contracts/orotitan-equity/v2/04_INTEGRATION_SPEC_V2.md",
    blob_sha: "383fe1247555cc15ab7fa4217ec2210b128e2e58",
  },
  screener_schema: {
    path: "contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V2.json",
    blob_sha: "bfd346d7ab3f2c9836f9a4a710d7a8b171fd6be0",
  },
} as const;

function gitBlobSha(bytes: Buffer): string {
  const header = Buffer.from(`blob ${bytes.length}\0`, "utf8");
  return createHash("sha1").update(header).update(bytes).digest("hex");
}

test("V2 top-level successor authorities have exact Git blob identities and emit SHA-256", () => {
  const report: Record<string, { path: string; blob_sha: string; content_sha256: string; size_bytes: number }> = {};
  for (const [key, authority] of Object.entries(authorities)) {
    const bytes = readFileSync(authority.path);
    const actualBlob = gitBlobSha(bytes);
    assert.equal(actualBlob, authority.blob_sha, `${key} Git blob drift`);
    report[key] = {
      path: authority.path,
      blob_sha: actualBlob,
      content_sha256: createHash("sha256").update(bytes).digest("hex"),
      size_bytes: bytes.length,
    };
  }
  console.log(`OROTITAN_V2_PIN_SOURCE_HASHES=${JSON.stringify(report)}`);
});
