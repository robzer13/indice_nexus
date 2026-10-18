/**
 * D3: exact byte-backed lock gates for Pilotage V3.1.
 * No report status is accepted as proof of I2 or I3-B recomputation.
 * Denominator provenance and EV-bridge economic proofs remain separate.
 */
import {
  validateFundamentalsLock, validateValuationLock,
  type FundamentalsLock, type ValuationLock,
} from "../v2/deep-dive-runtime";
import type { ArtifactRef } from "../v2/handoff";
import {
  resolveValuationDateAlignment, type ValuationDateAlignmentInput,
} from "./valuation-date-alignment";
import type { ArtifactProof, Gate, ManifestProof } from "./pilotage-adapter-v3-1";

export interface VerifiedContent {
  proof: ArtifactProof;
  bytes: Uint8Array;
  mediaType: string;
}
export interface TypedStageCheck {
  gates: Partial<Record<Gate, boolean>>;
  lockRefs: { fundamentals: ArtifactRef | null; valuation: ArtifactRef | null };
  observedDateAlignment: "ADMITTED" | "BLOCKED" | "NOT_VERIFIED";
  i2Recomputed: false;
  i3bComposedValidated: false;
}
export class TypedGateError extends Error {
  constructor(readonly code: string) { super(code); }
}
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SHA = /^[0-9a-f]{64}$/;
function need(v: unknown, code: string): asserts v {
  if (!v) throw new TypedGateError(code);
}
function obj(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}
function json(c: VerifiedContent, code: string): Record<string, unknown> {
  need(c.mediaType === "application/json", code + "_MEDIA_TYPE");
  try {
    const v: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(c.bytes));
    need(obj(v), code + "_NOT_OBJECT");
    return v;
  } catch (e) {
    if (e instanceof TypedGateError) throw e;
    throw new TypedGateError(code + "_INVALID_JSON");
  }
}
function validRef(ref: unknown): ref is Required<ArtifactRef> {
  return obj(ref) && typeof ref.artifact_id === "string" && UUID.test(ref.artifact_id) &&
    Number.isSafeInteger(ref.version) && (ref.version as number) > 0 &&
    typeof ref.content_sha256 === "string" && SHA.test(ref.content_sha256);
}
function refOf(p: ArtifactProof): Required<ArtifactRef> {
  return { artifact_id: p.artifactId, version: p.version, content_sha256: p.sha256 };
}
function same(ref: ArtifactRef, p: ArtifactProof): boolean {
  return ref.artifact_id === p.artifactId && ref.version === p.version &&
    ref.content_sha256 === p.sha256;
}
function requireRef(ref: unknown, verified: readonly VerifiedContent[], code: string): void {
  need(validRef(ref) && verified.filter(x => same(ref, x.proof)).length === 1, code);
}
function requireSelf(ref: unknown, p: ArtifactProof, code: string): void {
  need(validRef(ref) && same(ref, p), code);
}
function one(verified: readonly VerifiedContent[], artifactType: string): VerifiedContent | null {
  const a = verified.filter(x => x.proof.artifactType === artifactType);
  need(a.length <= 1, "DUPLICATE_VERIFIED_" + artifactType);
  return a[0] ?? null;
}
function verifyLineage(v: Record<string, unknown>, self: ArtifactProof,
  verified: readonly VerifiedContent[], fields: readonly string[], code: string): void {
  requireSelf(v.artifact, self, code + "_SELF_REF_INVALID");
  for (const f of fields) requireRef(v[f], verified, code + "_" + f + "_REF_INVALID");
  need(Array.isArray(v.evidence_lineage) && v.evidence_lineage.length > 0,
    code + "_EVIDENCE_LINEAGE_MISSING");
  for (const x of v.evidence_lineage) requireRef(x, verified, code + "_EVIDENCE_REF_INVALID");
}

export function verifyDeepDiveTypedLocks(input: {
  runId: string;
  dataCutoff: string;
  stageRevision: number;
  manifestKind: "CHECKPOINT" | "FINAL";
  researchFinalManifest: ManifestProof;
  currentOutputs: readonly VerifiedContent[];
  researchOutputs: readonly VerifiedContent[];
}): TypedStageCheck {
  const result: TypedStageCheck = {
    gates: {}, lockRefs: { fundamentals: null, valuation: null },
    observedDateAlignment: "NOT_VERIFIED", i2Recomputed: false, i3bComposedValidated: false,
  };
  const current = input.currentOutputs;
  const all = [...current, ...input.researchOutputs];
  const fund = one(current, "FUNDAMENTALS_LOCK");
  need(fund, "FUNDAMENTALS_LOCK_NOT_VERIFIED_IN_CURRENT_BUNDLE");
  const f = json(fund, "FUNDAMENTALS_LOCK");
  need(f.run_id === input.runId && f.data_cutoff === input.dataCutoff &&
    f.stage_revision === input.stageRevision, "FUNDAMENTALS_LOCK_SCOPE_OR_REVISION_MISMATCH");
  requireSelf(f.research_final_manifest, input.researchFinalManifest,
    "FUNDAMENTALS_RESEARCH_MANIFEST_NOT_CURRENT");
  verifyLineage(f, fund.proof, all,
    ["taxonomy_record", "business_description_artifact", "scoring_inputs_artifact"], "FUNDAMENTALS");
  need(f.red_team_status === "COMPLETE", "FUNDAMENTALS_RED_TEAM_NOT_COMPLETE");
  need(Array.isArray(f.limitations) && Array.isArray(f.invalidation_triggers),
    "FUNDAMENTALS_LOCK_LIMITATIONS_INVALID");
  const errors = validateFundamentalsLock(f as unknown as FundamentalsLock, {
    runId: input.runId, dataCutoff: input.dataCutoff,
    researchFinalManifest: refOf(input.researchFinalManifest),
  });
  need(errors.length === 0, "FUNDAMENTALS_LOCK_CONTRACT_INVALID");
  need(f.ready_for_valuation === "YES", "FUNDAMENTALS_NOT_READY_FOR_VALUATION");
  need(one(current, "FUNDAMENTAL_RED_TEAM_RECORD"), "RED_TEAM_RECORD_NOT_VERIFIED");
  result.gates.READY_FOR_VALUATION = true;
  result.gates.FUNDAMENTAL_RED_TEAM_COMPLETE = true;
  result.lockRefs.fundamentals = refOf(fund.proof);

  const val = one(current, "VALUATION_LOCK");
  if (!val) {
    need(input.manifestKind === "CHECKPOINT", "FINAL_WITHOUT_VALUATION_LOCK");
    return result;
  }
  const v = json(val, "VALUATION_LOCK");
  need(v.run_id === input.runId && v.data_cutoff === input.dataCutoff &&
    v.stage_revision === input.stageRevision, "VALUATION_LOCK_SCOPE_OR_REVISION_MISMATCH");
  requireSelf(v.fundamentals_lock, fund.proof, "VALUATION_FUNDAMENTALS_LINK_MISMATCH");
  verifyLineage(v, val.proof, all, [
    "calculation_ledger", "assumption_register", "valuation_artifact",
    "full_precision_inputs_artifact",
  ], "VALUATION");
  const errorsVal = validateValuationLock(v as unknown as ValuationLock, {
    runId: input.runId, dataCutoff: input.dataCutoff, fundamentalsLock: refOf(fund.proof),
  });
  need(errorsVal.length === 0, "VALUATION_LOCK_CONTRACT_INVALID");
  need(v.ready_for_certification === "YES", "VALUATION_NOT_READY_FOR_CERTIFICATION");
  result.lockRefs.valuation = refOf(val.proof);

  // Deterministic date consistency, only when typed inputs are actually
  // persisted inside a byte-verified VALUATION_ARTIFACT. It does not establish
  // real-world correctness of share-count movement or cash-flow provenance.
  const valArtifact = one(current, "VALUATION_ARTIFACT");
  if (valArtifact && valArtifact.mediaType === "application/json") {
    const parsed = json(valArtifact, "VALUATION_ARTIFACT");
    const raw = parsed.valuation_date_alignment;
    if (obj(raw)) {
      const fields = [
        "dataCutoff", "valuationDate", "timeOrigin", "hasNumericReferencePrice",
        "referencePriceDate", "referencePriceStalenessResolved",
        "economicShareCountDate", "shareCountAsOfDate",
        "economicShareCountRepresentation", "denominatorBridgeResolved",
        "cashFlowBasis", "evToEquityBridgeDate", "evBridgeRollforwardResolved",
        "perShareOutputDate", "marketCapShareCountDate", "marketCapShareCountResolved",
      ];
      need(Object.keys(raw).length === fields.length &&
        fields.every(k => Object.prototype.hasOwnProperty.call(raw, k)),
        "VALUATION_DATE_INPUT_SHAPE_INVALID");
      const booleans = ["hasNumericReferencePrice", "referencePriceStalenessResolved",
        "denominatorBridgeResolved", "evBridgeRollforwardResolved", "marketCapShareCountResolved"];
      need(booleans.every(k => typeof raw[k] === "boolean"), "VALUATION_DATE_BOOLEAN_FIELD_INVALID");
      const dates = ["dataCutoff", "valuationDate", "timeOrigin",
        "economicShareCountDate", "shareCountAsOfDate", "perShareOutputDate"];
      need(dates.every(k => typeof raw[k] === "string" && /^\d{4}-\d{2}-\d{2}$/.test(raw[k] as string)),
        "VALUATION_DATE_REQUIRED_DATE_INVALID");
      const nullableDates = ["referencePriceDate", "evToEquityBridgeDate", "marketCapShareCountDate"];
      need(nullableDates.every(k => raw[k] === null || (typeof raw[k] === "string" &&
        /^\d{4}-\d{2}-\d{2}$/.test(raw[k] as string))), "VALUATION_DATE_OPTIONAL_DATE_INVALID");
      need(["EXACT", "BOUNDED", "UNKNOWN"].includes(raw.economicShareCountRepresentation as string),
        "VALUATION_DATE_REPRESENTATION_INVALID");
      need(["FCFF", "FCFE", "OWNER_EARNINGS"].includes(raw.cashFlowBasis as string),
        "VALUATION_DATE_CASH_FLOW_BASIS_INVALID");
      need(raw.dataCutoff === input.dataCutoff, "VALUATION_DATE_CUTOFF_MISMATCH");
      const date = resolveValuationDateAlignment(raw as unknown as ValuationDateAlignmentInput);
      result.observedDateAlignment = date.valuationAdmission && date.blockers.length === 0 ?
        "ADMITTED" : "BLOCKED";
      if (result.observedDateAlignment === "ADMITTED") {
        result.gates.VALUATION_DATE_ALIGNMENT_VALID = true;
      }
    }
  }
  // Not enough authority here to set denominator/EV economic proofs,
  // READY_FOR_CERTIFICATION, I2_PASS, I3B_PASS or publication admission.
  return result;
}

/** Reports have no frozen standalone typed PASS attestation schema. */
export function assessIntegrationReportsReadOnly(): {
  i2Recomputed: false;
  i3bComposedValidated: false;
} {
  return { i2Recomputed: false, i3bComposedValidated: false };
}
