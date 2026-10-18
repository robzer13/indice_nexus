import assert from "node:assert/strict";
import test from "node:test";
import { verifyD6ProposedProvenance, D6ProvenanceError } from
  "../lib/orotitan-equity/v3/pilotage-provenance-d6-v3-1";
import type { CanonicalContractInput } from "../lib/orotitan-equity/v1/contract";
import type { VerifiedContent } from "../lib/orotitan-equity/v3/pilotage-typed-gates-v3-1";

const run = "11111111-1111-4111-8111-111111111111";
const date = "2026-10-16";
const contractHash = "a".repeat(64);
function canonical(): CanonicalContractInput {
  return {
    dimensions: {
      MOAT: 80, RUNWAY: 80, RETURN_QUALITY: 80, CASH_ECONOMICS: 80,
      CAPITAL_ALLOCATION: 80, MANAGEMENT_GOVERNANCE: 80, RESILIENCE_RISK: 80,
    },
    evidence: { moat: "STRONGLY_SUPPORTED", runway: "STRONGLY_SUPPORTED" },
    businessResearchStatus: "CERTIFIED", investmentConclusionStatus: "CERTIFIED",
    scorePermission: "ALLOWED", mosStatus: "ROBUST", valuationReliability: "HIGH",
    primaryExpectedReturnDeltaPercentagePoints: 2,
    normalizedExpectedReturnDeltaPercentagePoints: -2,
    eliteGates: {
      researchFullyCertified: "PASS", moatElite: "PASS", runwayElite: "PASS",
      returnQualityElite: "PASS", cashEconomicsElite: "PASS",
      capitalAllocationElite: "PASS", managementGovernanceElite: "PASS",
      resilienceElite: "PASS", valuationElite: "PASS", materialWeakLink: "PASS",
    },
  };
}
function artifact(stage: "DEEP_DIVE" | "RESEARCH", type: string, num: number,
  v: unknown, mediaType = "application/json"): VerifiedContent {
  return {
    proof: {
      runId: run, stageCode: stage, artifactType: type,
      artifactId: "77777777-7777-4777-8777-" + num.toString(16).padStart(12, "0"),
      version: 1, sha256: num.toString(16).padStart(64, "0"),
      available: true, bytesVerified: true,
    },
    mediaType, bytes: Buffer.from(mediaType === "application/json" ? JSON.stringify(v) : String(v)),
  };
}
function ref(x: VerifiedContent) {
  return {
    artifact_id: x.proof.artifactId, version: x.proof.version,
    content_sha256: x.proof.sha256, stage_code: x.proof.stageCode,
    artifact_type: x.proof.artifactType,
  };
}
function source(type: string, num: number, data: Record<string, unknown>, research: object) {
  const o: Record<string, unknown> = {
    run_id: run, data_cutoff: date, stage_revision: 3, canonical: data,
  };
  if (type === "ANALYTICAL_BLOCK_OUTPUTS") o.research_evidence_ref = research;
  return artifact("DEEP_DIVE", type, num, o);
}
function leaves(v: unknown, prefix = ""): Array<{path:string;value:unknown}> {
  if (typeof v !== "object" || v === null || Array.isArray(v)) return [{path:prefix,value:v}];
  return Object.entries(v).filter(([key]) => key !== "deterministic")
    .flatMap(([key,value]) => leaves(value, prefix + "/" + key));
}
type Fixture = {
  fullPrecision: VerifiedContent;
  sources: VerifiedContent[];
  input: CanonicalContractInput;
  document: Record<string, unknown>;
};
function fixture(): Fixture {
  const input = canonical();
  const research = artifact("RESEARCH", "EVIDENCE_LEDGER", 10, "ledger", "text/plain");
  const fundamental = source("ANALYTICAL_BLOCK_OUTPUTS", 11, {
    dimensions: input.dimensions, evidence: input.evidence,
  }, ref(research));
  const certification = source("CERTIFICATION_ARTIFACT", 12, {
    businessResearchStatus: input.businessResearchStatus,
    investmentConclusionStatus: input.investmentConclusionStatus,
    scorePermission: input.scorePermission,
  }, ref(research));
  const valuation = source("VALUATION_ARTIFACT", 13, {
    mosStatus: input.mosStatus, valuationReliability: input.valuationReliability,
    primaryExpectedReturnDeltaPercentagePoints: input.primaryExpectedReturnDeltaPercentagePoints,
    normalizedExpectedReturnDeltaPercentagePoints: input.normalizedExpectedReturnDeltaPercentagePoints,
  }, ref(research));
  const terminal = source("OROTITAN_TERMINAL_GATE_ARTIFACT", 14, {
    eliteGates: input.eliteGates,
  }, ref(research));
  const byRoot: Record<string, {source: VerifiedContent, role: string}> = {};
  for (const key of ["dimensions","evidence"]) byRoot[key] = {source:fundamental, role:"FUNDAMENTAL_JUDGMENT"};
  for (const key of ["businessResearchStatus","investmentConclusionStatus","scorePermission"])
    byRoot[key] = {source:certification,role:"CERTIFICATION_JUDGMENT"};
  for (const key of ["mosStatus","valuationReliability","primaryExpectedReturnDeltaPercentagePoints","normalizedExpectedReturnDeltaPercentagePoints"])
    byRoot[key] = {source:valuation,role:"VALUATION_JUDGMENT"};
  byRoot.eliteGates = {source:terminal,role:"TERMINAL_GATE_JUDGMENT"};
  const self = artifact("DEEP_DIVE", "SCORING_INPUTS_FULL_PRECISION", 20, {});
  const document: Record<string, unknown> = {
    schema_version: "D6_PROPOSED_0.1.0",
    run_id: run, data_cutoff: date, stage_revision: 3,
    contract_set_sha256: contractHash,
    artifact_identity: {artifact_id:self.proof.artifactId, version:1},
    research_evidence_ref: ref(research),
    canonical_contract_input: input,
    bindings: leaves(input).map(x => {
      const b = byRoot[x.path.split("/")[1]];
      return {
        input_pointer: x.path,
        source_ref: ref(b.source),
        source_pointer: "/canonical" + x.path,
        authority_role: b.role,
      };
    }),
  };
  self.bytes = Buffer.from(JSON.stringify(document));
  return {fullPrecision:self,sources:[research,fundamental,certification,valuation,terminal],
    input, document};
}
function update(f: Fixture) { f.fullPrecision.bytes = Buffer.from(JSON.stringify(f.document)); }
function runCheck(f: Fixture) {
  return verifyD6ProposedProvenance({
    fullPrecision:f.fullPrecision,sourceArtifacts:f.sources,
    expectedRunId:run,expectedDataCutoff:date,expectedStageRevision:3,
    expectedContractSetSha256:contractHash,candidateCanonicalInput:f.input,
  });
}
function fails(f: Fixture, code: string) {
  assert.throws(() => runCheck(f),
    (e:unknown) => e instanceof D6ProvenanceError && e.code === code, code);
}
test("D6 proposal verifies each full precision leaf and exact independent source", () => {
  const f=fixture(),o=runCheck(f);
  assert.equal(o.exactSourceBindingsVerified,true);
  assert.equal(o.validatedLeafCount,(f.document.bindings as unknown[]).length);
  assert.equal(o.methodologyFreezeApproved,false);
  assert.equal(o.analyticalCertificationProven,false);
  assert.equal(o.i2Pass,false);
  assert.equal(o.i3bPass,false);
  assert.equal(o.publicationAuthorized,false);
});
test("D6 blocks a missing binding", () => {
  const f=fixture(); (f.document.bindings as unknown[]).pop(); update(f);
  fails(f,"D6_LINEAGE_INCOMPLETE_OR_EXTRA");
});
test("D6 blocks duplicated binding in place of a distinct input", () => {
  const f=fixture(),a=f.document.bindings as Array<Record<string,unknown>>;
  a[1]=structuredClone(a[0]);update(f);
  fails(f,"D6_LINEAGE_DUPLICATE_OR_UNKNOWN_INPUT");
});
test("D6 blocks independently incorrect source value", () => {
  const f=fixture(),s=f.sources[1];
  const json=JSON.parse(Buffer.from(s.bytes).toString("utf8")) as Record<string,unknown>;
  ((json.canonical as Record<string,unknown>).dimensions as Record<string,unknown>).MOAT=75;
  s.bytes=Buffer.from(JSON.stringify(json));
  fails(f,"D6_LINEAGE_SOURCE_VALUE_MISMATCH");
});
test("D6 refuses spoofed source artifact fingerprint", () => {
  const f=fixture(); f.sources[1].proof.sha256="b".repeat(64);
  fails(f,"D6_LINEAGE_SOURCE_NOT_BYTE_VERIFIED");
});
test("D6 refuses a self-declared certification role for fundamentals", () => {
  const f=fixture(),a=f.document.bindings as Array<Record<string,unknown>>;
  a[0].authority_role="CERTIFICATION_JUDGMENT";update(f);
  fails(f,"D6_LINEAGE_WRONG_AUTHORITY_ROLE");
});
test("D6 refuses unverified Research evidence", () => {
  const f=fixture();f.sources[0].proof.bytesVerified=false;
  fails(f,"D6_RESEARCH_EVIDENCE_NOT_BYTE_VERIFIED");
});
test("D6 requires Fundamentals source to backlink Research evidence", () => {
  const f=fixture(),s=f.sources[1];
  const json=JSON.parse(Buffer.from(s.bytes).toString("utf8")) as Record<string,unknown>;
  delete json.research_evidence_ref; s.bytes=Buffer.from(JSON.stringify(json));
  fails(f,"D6_FUNDAMENTALS_RESEARCH_EVIDENCE_BACKLINK_MISSING");
});
test("D6 blocks wrong cutoff and missing freeze pin", () => {
  const f=fixture();f.document.contract_set_sha256="b".repeat(64);update(f);
  fails(f,"D6_PROVENANCE_SCOPE_OR_CONTRACT_MISMATCH");
});
test("D6 blocks malformed structure even when claims say PASS", () => {
  const f=fixture();f.document.i2_pass=true;update(f);
  fails(f,"D6_PROPOSED_SCHEMA_INVALID");
});
test("D6 blocks fake JSON pointer paths", () => {
  const f=fixture(),a=f.document.bindings as Array<Record<string,unknown>>;
  a[0].source_pointer="/canonical/dimensions/DOES_NOT_EXIST";update(f);
  fails(f,"D6_SOURCE_POINTER_NOT_FOUND");
});
