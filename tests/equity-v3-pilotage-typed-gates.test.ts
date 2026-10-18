import assert from "node:assert/strict";
import test from "node:test";
import {
  verifyDeepDiveTypedLocks, TypedGateError, assessIntegrationReportsReadOnly,
  type VerifiedContent,
} from "../lib/orotitan-equity/v3/pilotage-typed-gates-v3-1";
import type { ArtifactProof, ManifestProof } from "../lib/orotitan-equity/v3/pilotage-adapter-v3-1";

const RUN = "11111111-1111-4111-8111-111111111111";
const SHA = "a".repeat(64);
const DATE = "2026-09-22";
function proof(stage: "RESEARCH" | "DEEP_DIVE", name: string, number: number): ArtifactProof {
  return {
    runId: RUN, stageCode: stage, artifactType: name,
    artifactId: "77777777-7777-4777-8777-" + number.toString(16).padStart(12, "0"),
    version: 1, sha256: SHA, available: true, bytesVerified: true,
  };
}
function ref(p: ArtifactProof) {
  return { artifact_id: p.artifactId, version: p.version, content_sha256: p.sha256 };
}
function textArtifact(stage: "RESEARCH" | "DEEP_DIVE", name: string, id: number): VerifiedContent {
  return { proof: proof(stage, name, id), bytes: Buffer.from(name), mediaType: "text/plain" };
}
function jsonArtifact(stage: "RESEARCH" | "DEEP_DIVE", name: string, id: number,
  data: Record<string, unknown>): VerifiedContent {
  return { proof: proof(stage, name, id), bytes: Buffer.from(JSON.stringify(data)), mediaType: "application/json" };
}
function overwrite(c: VerifiedContent, data: Record<string, unknown>): void {
  c.bytes = Buffer.from(JSON.stringify(data));
}
function read(c: VerifiedContent): Record<string, unknown> {
  return JSON.parse(Buffer.from(c.bytes).toString("utf8")) as Record<string, unknown>;
}
function fixture(includeValuation = false) {
  const researchFinal = { ...proof("RESEARCH", "RESEARCH_STAGE_MANIFEST", 1),
    kind: "FINAL", stageRevision: 1, gatesVerifiedAgainstBytes: true } as ManifestProof;
  const research = [textArtifact("RESEARCH", "EVIDENCE_LEDGER", 2)];
  const taxonomy = textArtifact("DEEP_DIVE", "CONTROLLED_TAXONOMY_RECORD", 3);
  const description = textArtifact("DEEP_DIVE", "BUSINESS_DESCRIPTION_SHORT_ARTIFACT", 4);
  const scoring = textArtifact("DEEP_DIVE", "SCORING_INPUTS", 5);
  const redTeam = textArtifact("DEEP_DIVE", "FUNDAMENTAL_RED_TEAM_RECORD", 6);
  const fundamentalsProof = proof("DEEP_DIVE", "FUNDAMENTALS_LOCK", 10);
  const lock = {
    lock_type: "FUNDAMENTALS_LOCK",
    artifact: ref(fundamentalsProof), run_id: RUN, data_cutoff: DATE,
    stage_revision: 1, research_final_manifest: ref(researchFinal),
    evidence_lineage: [ref(research[0].proof)], red_team_status: "COMPLETE",
    taxonomy_record: ref(taxonomy.proof),
    business_description_artifact: ref(description.proof),
    scoring_inputs_artifact: ref(scoring.proof),
    limitations: [], invalidation_triggers: [], ready_for_valuation: "YES",
  };
  const fundamentals = { proof: fundamentalsProof, mediaType: "application/json",
    bytes: Buffer.from(JSON.stringify(lock)) };
  const current: VerifiedContent[] = [taxonomy, description, scoring, redTeam, fundamentals];
  let valuation: VerifiedContent | null = null;
  let timing: VerifiedContent | null = null;
  if (includeValuation) {
    const calculation = textArtifact("DEEP_DIVE", "CALCULATION_LEDGER", 20);
    const assumptions = textArtifact("DEEP_DIVE", "MATERIAL_ASSUMPTION_REGISTER", 21);
    const fullPrecision = textArtifact("DEEP_DIVE", "SCORING_INPUTS_FULL_PRECISION", 22);
    timing = jsonArtifact("DEEP_DIVE", "VALUATION_ARTIFACT", 23, {
      valuation_date_alignment: {
        dataCutoff: DATE, valuationDate: DATE, timeOrigin: DATE,
        hasNumericReferencePrice: true, referencePriceDate: DATE,
        referencePriceStalenessResolved: true, economicShareCountDate: DATE,
        shareCountAsOfDate: DATE, economicShareCountRepresentation: "EXACT",
        denominatorBridgeResolved: true, cashFlowBasis: "FCFF",
        evToEquityBridgeDate: DATE, evBridgeRollforwardResolved: true,
        perShareOutputDate: DATE, marketCapShareCountDate: DATE,
        marketCapShareCountResolved: true,
      },
    });
    const valProof = proof("DEEP_DIVE", "VALUATION_LOCK", 30);
    const valData = {
      lock_type: "VALUATION_LOCK", artifact: ref(valProof),
      run_id: RUN, data_cutoff: DATE, stage_revision: 1,
      fundamentals_lock: ref(fundamentalsProof),
      evidence_lineage: [ref(research[0].proof)],
      calculation_ledger: ref(calculation.proof),
      assumption_register: ref(assumptions.proof),
      valuation_artifact: ref(timing.proof),
      full_precision_inputs_artifact: ref(fullPrecision.proof),
      ready_for_certification: "YES",
    };
    valuation = { proof: valProof, mediaType: "application/json",
      bytes: Buffer.from(JSON.stringify(valData)) };
    current.push(calculation, assumptions, fullPrecision, timing, valuation);
  }
  const evaluate = (kind: "FINAL" | "CHECKPOINT" = "CHECKPOINT") =>
    verifyDeepDiveTypedLocks({
      runId: RUN, dataCutoff: DATE, stageRevision: 1, manifestKind: kind,
      researchFinalManifest: researchFinal, currentOutputs: current, researchOutputs: research,
    });
  return { researchFinal, research, current, fundamentals, valuation, timing, evaluate };
}
function fail(fn: () => unknown, code: string) {
  assert.throws(fn, (e: unknown) => e instanceof TypedGateError && e.code === code, code);
}

test("Verified Fundamentals lock and Red Team record derive only the permitted lock gates", () => {
  const f = fixture();
  const got = f.evaluate();
  assert.equal(got.gates.READY_FOR_VALUATION, true);
  assert.equal(got.gates.FUNDAMENTAL_RED_TEAM_COMPLETE, true);
  assert.equal(got.gates.DENOMINATOR_ADMISSION_VALID, undefined);
  assert.equal(got.gates.DCF_BASIS_ADMITTED, undefined);
  assert.equal(got.gates.EV_BRIDGE_GOVERNED, undefined);
  assert.equal(got.gates.READY_FOR_CERTIFICATION, undefined);
});
test("wrong self identity fails", () => {
  const f = fixture();
  const raw = read(f.fundamentals);
  raw.artifact = { ...raw.artifact as object, version: 2 };
  overwrite(f.fundamentals, raw);
  fail(f.evaluate, "FUNDAMENTALS_SELF_REF_INVALID");
});
test("missing Research FINAL lineage fails", () => {
  const f = fixture();
  const raw = read(f.fundamentals);
  raw.research_final_manifest = { ...raw.research_final_manifest as object, version: 2 };
  overwrite(f.fundamentals, raw);
  fail(f.evaluate, "FUNDAMENTALS_RESEARCH_MANIFEST_NOT_CURRENT");
});
test("forged evidence hash fails even if UUID and version are valid", () => {
  const f = fixture();
  const raw = read(f.fundamentals);
  const line = raw.evidence_lineage as Array<Record<string, unknown>>;
  raw.evidence_lineage = [{ ...line[0], content_sha256: "f".repeat(64) }];
  overwrite(f.fundamentals, raw);
  fail(f.evaluate, "FUNDAMENTALS_EVIDENCE_REF_INVALID");
});
test("missing mandatory Red Team record fails", () => {
  const f = fixture();
  f.current.splice(f.current.findIndex(x => x.proof.artifactType === "FUNDAMENTAL_RED_TEAM_RECORD"), 1);
  fail(f.evaluate, "RED_TEAM_RECORD_NOT_VERIFIED");
});
test("not ready Fundamentals lock fails despite artifact existence", () => {
  const f = fixture();
  const raw = read(f.fundamentals);
  raw.ready_for_valuation = "NO";
  overwrite(f.fundamentals, raw);
  fail(f.evaluate, "FUNDAMENTALS_NOT_READY_FOR_VALUATION");
});
test("incomplete Red Team fails despite lock YES", () => {
  const f = fixture();
  const raw = read(f.fundamentals);
  raw.red_team_status = "INCOMPLETE";
  overwrite(f.fundamentals, raw);
  fail(f.evaluate, "FUNDAMENTALS_RED_TEAM_NOT_COMPLETE");
});
test("FINAL Deep Dive cannot omit Valuation lock", () => {
  fail(() => fixture().evaluate("FINAL"), "FINAL_WITHOUT_VALUATION_LOCK");
});
test("Valuation lock verifies exact Fundamentals lock linkage", () => {
  const f = fixture(true);
  const got = f.evaluate();
  assert.equal(got.lockRefs.valuation?.artifact_id, f.valuation?.proof.artifactId);
  assert.equal(got.observedDateAlignment, "ADMITTED");
  assert.equal(got.gates.VALUATION_DATE_ALIGNMENT_VALID, true);
  assert.equal(got.gates.READY_FOR_CERTIFICATION, undefined);
});
test("Valuation lock cannot consume a different Fundamentals version", () => {
  const f = fixture(true);
  const raw = read(f.valuation!);
  raw.fundamentals_lock = { ...raw.fundamentals_lock as object, version: 2 };
  overwrite(f.valuation!, raw);
  fail(f.evaluate, "VALUATION_FUNDAMENTALS_LINK_MISMATCH");
});
test("Valuation must own exact cutoff and stage revision", () => {
  const f = fixture(true);
  const raw = read(f.valuation!);
  raw.data_cutoff = "2026-09-23";
  overwrite(f.valuation!, raw);
  fail(f.evaluate, "VALUATION_LOCK_SCOPE_OR_REVISION_MISMATCH");
});
test("Valuation date mismatch is never silently accepted", () => {
  const f = fixture(true);
  const raw = read(f.timing!);
  const time = raw.valuation_date_alignment as Record<string, unknown>;
  time.valuationDate = "2026-09-21";
  overwrite(f.timing!, raw);
  assert.throws(f.evaluate, /VALUATION_DATE_DATA_CUTOFF_MISMATCH/);
});
test("negative or unknown denominator blocks timing gate", () => {
  const f = fixture(true);
  const raw = read(f.timing!);
  const time = raw.valuation_date_alignment as Record<string, unknown>;
  time.economicShareCountRepresentation = "UNKNOWN";
  overwrite(f.timing!, raw);
  const got = f.evaluate();
  assert.equal(got.observedDateAlignment, "BLOCKED");
  assert.equal(got.gates.VALUATION_DATE_ALIGNMENT_VALID, undefined);
});
test("Final Deep Dive content must verify both locks", () => {
  const f = fixture(true);
  const result = f.evaluate("FINAL");
  assert.ok(result.lockRefs.fundamentals);
  assert.ok(result.lockRefs.valuation);
  assert.equal(result.gates.READY_FOR_CERTIFICATION, undefined);
});
test("I2 and I3-B PASS reports are not independent validation", () => {
  const x = assessIntegrationReportsReadOnly();
  assert.equal(x.i2Recomputed, false);
  assert.equal(x.i3bComposedValidated, false);
});
