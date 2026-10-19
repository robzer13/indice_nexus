import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  RESEARCH_FINAL_REQUIRED_ARTIFACT_TYPES,
  projectV31ResearchFinalForV1Schema,
} from "../lib/orotitan-equity/v3/pilotage-research-final-artifacts-v3-1";

const v2 = [
  "RESEARCH_SOURCE_MANIFEST",
  "EVIDENCE_LEDGER",
  "CONFLICT_LEDGER",
  "MATERIAL_RESEARCH_HYPOTHESIS_REGISTER",
  "RESEARCH_GAP_REGISTER",
  "INPUT_SUFFICIENCY_ASSESSMENT",
  "CLASSIFICATION_INPUT_RECORD",
];
const v1 = [
  "RESEARCH_SOURCE_MANIFEST",
  "EVIDENCE_LEDGER",
  "CONFLICT_LEDGER",
  "MATERIAL_RESEARCH_HYPOTHESIS_REGISTER",
  "RESEARCH_GAP_REGISTER",
  "DD_INPUT_SUFFICIENCY_RECORD",
  "ANALYSIS_INPUT_LOCK",
];
const manifest = () => ({
  manifest_kind: "FINAL",
  stage: "RESEARCH",
  process_version: "3.1",
  pilotage_contract_version: "3.1",
  stage_contract: {
    name: "OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0",
    version: "2.0",
  },
  output_artifacts: v2.map((artifact_type, index) => ({
    artifact_id: "11111111-1111-4111-8111-"+String(index+1).padStart(12,"0"),
    version: 1,
    authority_class: "AUTHORITATIVE_STAGE_OUTPUT",
    content_sha256: "a".repeat(64),
    artifact_type,
  })),
});
test("D11 exact seven Research output types match pinned V2 stage contract, not V1 aliases", () => {
  const text = readFileSync(
    new URL("../contracts/orotitan-equity/v2/OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0.md", import.meta.url), "utf8");
  assert.deepEqual(RESEARCH_FINAL_REQUIRED_ARTIFACT_TYPES, v2);
  for (const type of v2) assert.match(text, new RegExp("\\b"+type+"\\b"));
  for (const type of v1.slice(-2)) assert.ok(!RESEARCH_FINAL_REQUIRED_ARTIFACT_TYPES.includes(type as never));
});
test("D11 Research V3.1 FINAL projects labels for V1 schema only, preserving original bytes and artifact proof identities", () => {
  const source = manifest();
  const before = JSON.stringify(source);
  const projected = projectV31ResearchFinalForV1Schema(source);
  assert.notEqual(projected, source);
  assert.deepEqual((projected.output_artifacts as typeof source.output_artifacts).map(x => x.artifact_type), v1);
  assert.deepEqual((projected.output_artifacts as typeof source.output_artifacts).map(x => x.artifact_id), source.output_artifacts.map(x => x.artifact_id));
  assert.equal(JSON.stringify(source), before);
});
test("D11 CHECKPOINT, V1 and wrong stage are never coerced into positive FINAL", () => {
  for (const diff of [
    {manifest_kind:"CHECKPOINT"},
    {process_version:"1.0"},
    {stage:"DEEP_DIVE"},
    {stage_contract:{name:"UNAUTHORIZED",version:"2.0"}},
  ]) {
    const input = {...manifest(),...diff};
    assert.equal(projectV31ResearchFinalForV1Schema(input),input);
  }
});
test("D11 SQL successor proposal is version-pinned, non-installed and preserves V1 names", () => {
  const sql=readFileSync(new URL("../docs/orotitan-equity/proposals/D11_V31_RESEARCH_FINAL_REGISTRY_V2_SQL_PROPOSED.sql",import.meta.url),"utf8");
  assert.match(sql,/PROPOSED \/ NOT INSTALLED/);
  assert.match(sql,/OROTITAN_RESEARCH_STAGE_CONTRACT_V1_FREEZE_V1\.0/);
  assert.match(sql,/OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2\.0/);
  assert.match(sql,/unknown Research stage-contract authority/);
  for (const type of [...v2.slice(-2),...v1.slice(-2)]) assert.ok(sql.includes(type));
  assert.doesNotMatch(sql,/\bDROP\s+(?:TABLE|FUNCTION)\b/i);
});
