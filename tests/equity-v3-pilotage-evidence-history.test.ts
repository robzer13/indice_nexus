import assert from "node:assert/strict";
import test from "node:test";
import {
  verifyHistoricalPrecheckD5, verifyI2LineageD5, D5EvidenceError,
  type D5Dossier, type D5SnapshotRow,
} from "../lib/orotitan-equity/v3/pilotage-evidence-history-v3-1";
import type { CanonicalContractInput } from "../lib/orotitan-equity/v1/contract";
import type { VerifiedContent } from "../lib/orotitan-equity/v3/pilotage-typed-gates-v3-1";

const ids = {
  run: "11111111-1111-4111-8111-111111111111",
  issuer: "22222222-2222-4222-8222-222222222222",
  security: "33333333-3333-4333-8333-333333333333",
  dossier: "44444444-4444-4444-8444-444444444444",
  baseline: "55555555-5555-4555-8555-555555555555",
  candidate: "66666666-6666-4666-8666-666666666666",
  artifact: "77777777-7777-4777-8777-777777777777",
};
const date = "2026-10-16";
const sha = "a".repeat(64);

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
function verifiedFullPrecision(value: unknown): VerifiedContent {
  return {
    proof: {
      artifactId: ids.artifact, runId: ids.run, stageCode: "DEEP_DIVE",
      artifactType: "SCORING_INPUTS_FULL_PRECISION",
      version: 3, sha256: sha, available: true, bytesVerified: true,
    },
    bytes: Buffer.from(JSON.stringify(value)), mediaType: "application/json",
  };
}
function checkLineage(value: unknown, candidate = canonical()) {
  return verifyI2LineageD5({
    candidateCanonicalInput: candidate,
    fullPrecision: verifiedFullPrecision(value),
    expectedRunId: ids.run, expectedDataCutoff: date,
    expectedArtifactId: ids.artifact, expectedVersion: 3, expectedSha256: sha,
  });
}
function exactRecord() {
  return { run_id: ids.run, data_cutoff: date, canonical_contract_input: canonical() };
}
function fails(fn: () => unknown, code: string) {
  assert.throws(fn, (e: unknown) => e instanceof D5EvidenceError && e.code === code, code);
}
test("D5 exact full precision I2 replay attests equality but never I2 admission", () => {
  const result = checkLineage(exactRecord());
  assert.equal(result.fullPrecisionInputsMatch, true);
  assert.equal(result.deterministicOutputsMatch, true);
  assert.equal(result.i2EndToEndPass, false);
  assert.equal(result.authoritySchemaFrozen, false);
  assert.equal(result.publicationAuthorized, false);
});
test("D5 rejects a different upstream unrounded number", () => {
  const r = exactRecord();
  r.canonical_contract_input.primaryExpectedReturnDeltaPercentagePoints = 2.00000000001;
  fails(() => checkLineage(r), "D5_CANDIDATE_I2_INPUTS_DIFFER_FROM_UPSTREAM");
});
test("D5 refuses missing upstream contract input rather than fabricating provenance", () => {
  fails(() => checkLineage({run_id: ids.run, data_cutoff: date}),
    "D5_UPSTREAM_I2_INPUT_PROJECTION_MISSING");
});
test("D5 rejects foreign run and cutoff even when input content matches", () => {
  const r = exactRecord();
  fails(() => checkLineage({...r,run_id: ids.baseline}), "D5_FULL_PRECISION_SCOPE_MISMATCH");
  fails(() => checkLineage({...r,data_cutoff: "2026-10-15"}), "D5_FULL_PRECISION_SCOPE_MISMATCH");
});
test("D5 rejects invalid scoring dimension and a forged artifact ref", () => {
  const r = exactRecord();
  r.canonical_contract_input.dimensions.MOAT = 300;
  fails(() => checkLineage(r), "D5_UPSTREAM_I2_INPUT_INVALID");
  const a = verifiedFullPrecision(exactRecord());
  a.proof.sha256 = "b".repeat(64);
  fails(() => verifyI2LineageD5({
    candidateCanonicalInput: canonical(), fullPrecision: a,
    expectedRunId: ids.run, expectedDataCutoff: date,
    expectedArtifactId: ids.artifact, expectedVersion: 3, expectedSha256: sha,
  }), "D5_UPSTREAM_FULL_PRECISION_REF_NOT_VERIFIED");
});
function candidate() {
  return {
    snapshot_id: ids.candidate, issuer_id: ids.issuer, security_id: ids.security,
    report_id: "research-2026-10-16", execution_mode: "ANALYZE",
    data_lock: { data_cutoff: date, calculation_date: date },
    versions: {
      report_version: "r1", method_version: "m1",
      calculation_version: "c1", evidence_ledger_version: "e1",
    },
  };
}
function snapshotRow(id: string, payload: ReturnType<typeof candidate>): D5SnapshotRow {
  return {
    snapshot_id: id, dossier_id: ids.dossier, issuer_id: ids.issuer,
    security_id: ids.security, report_id: payload.report_id,
    execution_mode: payload.execution_mode, data_cutoff: date,
    calculation_date: date,
    report_version: "r1", method_version: "m1",
    calculation_version: "c1", evidence_ledger_version: "e1",
    contract_version: "04_SCREENER_SCHEMA_V2", schema_version: "2.0.0",
    canonical_payload: payload,
  };
}
function fixture() {
  const dossier: D5Dossier = {
    dossier_id: ids.dossier, issuer_id: ids.issuer,
    current_snapshot_id: ids.baseline, active: true,
  };
  const payload = candidate();
  return {
    dossier, payload,
    baseline: snapshotRow(ids.baseline, {...payload, snapshot_id: ids.baseline}),
    existingCandidate: null as D5SnapshotRow | null,
  };
}
function history(f = fixture(), expectedBaselineId: string | null = ids.baseline) {
  return verifyHistoricalPrecheckD5({
    observed: { dossier: f.dossier, baseline: f.baseline,
      existingCandidate: f.existingCandidate },
    candidate: f.payload, runId: ids.run, dossierId: ids.dossier,
    issuerId: ids.issuer, securityId: ids.security, expectedBaselineId,
    expectedWriterContractVersion: "04_SCREENER_SCHEMA_V2",
    expectedWriterSchemaVersion: "2.0.0",
  });
}
test("historical first-write precheck observes exact baseline but never CAS proof", () => {
  const r = history();
  assert.equal(r.mode, "NEW_SNAPSHOT_PRECHECK");
  assert.equal(r.observedCurrentSnapshotId, ids.baseline);
  assert.equal(r.transactionCasValidated, false);
  assert.equal(r.historyTransitionPass, false);
  assert.equal(r.publicationAuthorized, false);
});
test("stale dossier pointer prevents fresh candidate", () => {
  const f = fixture();
  f.dossier.current_snapshot_id = ids.candidate;
  fails(() => history(f), "D5_HISTORY_POINTER_CHANGED");
});
test("baseline not belonging to dossier is rejected", () => {
  const f = fixture();
  f.baseline.dossier_id = ids.issuer;
  fails(() => history(f), "D5_HISTORY_BASELINE_NOT_OWNED_BY_DOSSIER");
});
test("candidate collision with incompatible immutable payload fails", () => {
  const f = fixture();
  f.existingCandidate = snapshotRow(ids.candidate, {
    ...f.payload, report_id: "malicious-replay",
  });
  f.dossier.current_snapshot_id = ids.candidate;
  fails(() => history(f), "D5_HISTORY_EXISTING_SNAPSHOT_CONFLICT");
});
test("idempotent historical replay still never implies publication rights", () => {
  const f = fixture();
  f.existingCandidate = snapshotRow(ids.candidate, f.payload);
  f.dossier.current_snapshot_id = ids.candidate;
  const r = history(f);
  assert.equal(r.mode, "IDENTICAL_REPLAY_PRECHECK");
  assert.equal(r.transactionCasValidated, false);
  assert.equal(r.historyTransitionPass, false);
});
test("idempotent replay rejects a pointer no longer on the candidate", () => {
  const f = fixture();
  f.existingCandidate = snapshotRow(ids.candidate, f.payload);
  f.dossier.current_snapshot_id = ids.baseline;
  fails(() => history(f), "D5_HISTORY_IDEMPOTENT_REPLAY_POINTER_MISMATCH");
});
test("first-ever dossier with null pointer can be prechecked without inventing a baseline", () => {
  const f = fixture();
  f.dossier.current_snapshot_id = null;
  f.baseline = null as unknown as D5SnapshotRow;
  const result = history(f, null);
  assert.equal(result.mode, "NEW_SNAPSHOT_PRECHECK");
});

test("idempotent replay requires pinned writer version authority", () => {
  const f = fixture();
  f.existingCandidate = snapshotRow(ids.candidate, f.payload);
  f.dossier.current_snapshot_id = ids.candidate;
  fails(() => verifyHistoricalPrecheckD5({
    observed: { dossier: f.dossier, baseline: f.baseline, existingCandidate: f.existingCandidate },
    candidate: f.payload, runId: ids.run, dossierId: ids.dossier,
    issuerId: ids.issuer, securityId: ids.security,
    expectedBaselineId: ids.baseline,
  }), "D5_HISTORY_WRITER_VERSION_NOT_AUTHORITATIVE");
});
test("idempotent replay rejects incorrect physical schema pair", () => {
  const f = fixture();
  f.existingCandidate = snapshotRow(ids.candidate, f.payload);
  f.existingCandidate.schema_version = "3.1.0";
  f.dossier.current_snapshot_id = ids.candidate;
  fails(() => history(f), "D5_HISTORY_WRITER_VERSION_CONFLICT");
});
