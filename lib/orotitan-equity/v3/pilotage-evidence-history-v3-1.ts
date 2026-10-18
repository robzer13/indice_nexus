/**
 * OroTitan D5 — independent, read-only historical pointer and scoring lineage
 * checks. Neither check is an authorization of an I3-B writer transaction.
 */
import { isDeepStrictEqual } from "node:util";
import { validateCanonicalContract, computeCanonicalSnapshot,
  type CanonicalContractInput } from "../v1/contract";
import type { VerifiedContent } from "./pilotage-typed-gates-v3-1";

export class D5EvidenceError extends Error {
  constructor(readonly code: string) { super(code); }
}
function requireTrue(v: unknown, code: string): asserts v {
  if (v !== true) throw new D5EvidenceError(code);
}
function obj(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SHA = /^[a-f0-9]{64}$/;
function samePointer(a: string | null, b: string | null): boolean {
  return a === b;
}
function valueAt(raw: unknown, path: string[]): unknown {
  let current = raw;
  for (const p of path) {
    if (!obj(current) || !Object.prototype.hasOwnProperty.call(current, p)) return undefined;
    current = current[p];
  }
  return current;
}
export interface D5Dossier {
  dossier_id: string;
  issuer_id: string;
  current_snapshot_id: string | null;
  active: boolean;
}
export interface D5SnapshotRow {
  snapshot_id: string;
  dossier_id: string;
  issuer_id: string;
  security_id: string;
  report_id: string;
  execution_mode: string;
  data_cutoff: string;
  calculation_date: string;
  report_version: string;
  method_version: string;
  calculation_version: string;
  evidence_ledger_version: string;
  contract_version: string;
  schema_version: string;
  canonical_payload: unknown;
}
export interface D5HistoryObservation {
  dossier: D5Dossier;
  baseline: D5SnapshotRow | null;
  existingCandidate: D5SnapshotRow | null;
}
export interface D5HistoryResult {
  mode: "NEW_SNAPSHOT_PRECHECK" | "IDENTICAL_REPLAY_PRECHECK";
  observedCurrentSnapshotId: string | null;
  expectedCurrentSnapshotId: string | null;
  candidateSnapshotId: string;
  historicalRowsConsistent: true;
  transactionCasValidated: false;
  historyTransitionPass: false;
  publicationAuthorized: false;
}
export interface D5I2LineageResult {
  fullPrecisionInputsMatch: true;
  upstreamArtifactVerified: true;
  deterministicOutputsMatch: true;
  authoritySchemaFrozen: false;
  upstreamCertificationVerified: false;
  i2EndToEndPass: false;
  publicationAuthorized: false;
}
/**
 * Check exact scalar/range scoring values against the full-precision
 * input record referenced by the current valuation lock.
 *
 * The canonical_contract_input wrapper is a D5 proposed typed extraction,
 * not a frozen OroTitan artifact schema. Matching does not establish that
 * the input record was analytically certified or that all source lines agree.
 */
export function verifyI2LineageD5(input: {
  candidateCanonicalInput: CanonicalContractInput;
  fullPrecision: VerifiedContent;
  expectedRunId: string;
  expectedDataCutoff: string;
  expectedArtifactId: string;
  expectedVersion: number;
  expectedSha256: string;
}): D5I2LineageResult {
  const a = input.fullPrecision;
  requireTrue(a.proof.bytesVerified === true && a.proof.available === true &&
    a.proof.runId === input.expectedRunId && a.proof.stageCode === "DEEP_DIVE" &&
    a.proof.artifactType === "SCORING_INPUTS_FULL_PRECISION" &&
    a.proof.artifactId === input.expectedArtifactId &&
    a.proof.version === input.expectedVersion &&
    SHA.test(input.expectedSha256) && a.proof.sha256 === input.expectedSha256,
    "D5_UPSTREAM_FULL_PRECISION_REF_NOT_VERIFIED");
  requireTrue(a.mediaType === "application/json", "D5_FULL_PRECISION_NOT_JSON");
  let raw: unknown;
  try { raw = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(a.bytes)); }
  catch { throw new D5EvidenceError("D5_FULL_PRECISION_JSON_INVALID"); }
  requireTrue(obj(raw), "D5_FULL_PRECISION_NOT_OBJECT");
  requireTrue(raw.run_id === input.expectedRunId && raw.data_cutoff === input.expectedDataCutoff,
    "D5_FULL_PRECISION_SCOPE_MISMATCH");
  requireTrue(obj(raw.canonical_contract_input),
    "D5_UPSTREAM_I2_INPUT_PROJECTION_MISSING");
  const parsed = validateCanonicalContract(raw.canonical_contract_input);
  requireTrue(parsed.success, "D5_UPSTREAM_I2_INPUT_INVALID");
  // Compare all exact fields (including optional deterministic assertions)
  // with no display rounding, hidden tolerances or scalarized ranges.
  requireTrue(isDeepStrictEqual(parsed.data, input.candidateCanonicalInput),
    "D5_CANDIDATE_I2_INPUTS_DIFFER_FROM_UPSTREAM");
  const aComputed = computeCanonicalSnapshot(parsed.data);
  const bComputed = computeCanonicalSnapshot(input.candidateCanonicalInput);
  requireTrue(isDeepStrictEqual(aComputed, bComputed),
    "D5_RECOMPUTATION_DIFFERENT_FROM_UPSTREAM");
  return {
    fullPrecisionInputsMatch: true,
    upstreamArtifactVerified: true,
    deterministicOutputsMatch: true,
    authoritySchemaFrozen: false,
    upstreamCertificationVerified: false,
    i2EndToEndPass: false,
    publicationAuthorized: false,
  };
}
/**
 * Independent precheck of physical writer's dossier/pointer/snapshot conditions.
 * This runs before a potential transaction, not inside SELECT ... FOR UPDATE.
 */
export function verifyHistoricalPrecheckD5(input: {
  observed: D5HistoryObservation;
  candidate: Record<string, unknown>;
  runId: string;
  dossierId: string;
  issuerId: string;
  securityId: string;
  expectedBaselineId: string | null;
}): D5HistoryResult {
  requireTrue(UUID.test(input.runId) && UUID.test(input.dossierId) &&
    UUID.test(input.issuerId) && UUID.test(input.securityId) &&
    (input.expectedBaselineId === null || UUID.test(input.expectedBaselineId)),
    "D5_HISTORY_EXPECTED_ID_INVALID");
  const c = input.candidate;
  const snapshotId = c.snapshot_id;
  requireTrue(typeof snapshotId === "string" && UUID.test(snapshotId),
    "D5_CANDIDATE_SNAPSHOT_ID_INVALID");
  requireTrue(c.issuer_id === input.issuerId && c.security_id === input.securityId,
    "D5_HISTORY_CANDIDATE_IDENTITY_MISMATCH");
  const d = input.observed.dossier;
  requireTrue(d.dossier_id === input.dossierId && d.issuer_id === input.issuerId &&
    d.active === true, "D5_HISTORY_DOSSIER_IDENTITY_INVALID");
  requireTrue(samePointer(d.current_snapshot_id, input.expectedBaselineId),
    "D5_HISTORY_POINTER_CHANGED");
  const baseline = input.observed.baseline;
  if (input.expectedBaselineId !== null) {
    requireTrue(baseline !== null && baseline.snapshot_id === input.expectedBaselineId &&
      baseline.dossier_id === input.dossierId && baseline.issuer_id === input.issuerId,
      "D5_HISTORY_BASELINE_NOT_OWNED_BY_DOSSIER");
    requireTrue(baseline.security_id === input.securityId,
      "D5_HISTORY_BASELINE_SECURITY_MISMATCH");
  } else requireTrue(baseline === null, "D5_HISTORY_UNEXPECTED_BASELINE");
  const existing = input.observed.existingCandidate;
  if (existing === null) {
    requireTrue(snapshotId !== input.expectedBaselineId,
      "D5_HISTORY_CANDIDATE_EQUALS_BASELINE_WITHOUT_ROW");
    return {
      mode: "NEW_SNAPSHOT_PRECHECK",
      observedCurrentSnapshotId: d.current_snapshot_id,
      expectedCurrentSnapshotId: input.expectedBaselineId,
      candidateSnapshotId: snapshotId,
      historicalRowsConsistent: true,
      transactionCasValidated: false,
      historyTransitionPass: false,
      publicationAuthorized: false,
    };
  }
  requireTrue(existing.snapshot_id === snapshotId,
    "D5_HISTORY_RETURNED_CANDIDATE_ID_MISMATCH");
  // The frozen V1/V2 I3-B writers reject any snapshot_id reuse with changed
  // payload, identity, version or pointer state, even for the same dossier.
  requireTrue(existing.dossier_id === input.dossierId &&
    existing.issuer_id === input.issuerId && existing.security_id === input.securityId &&
    existing.report_id === c.report_id &&
    existing.execution_mode === c.execution_mode &&
    existing.data_cutoff === valueAt(c, ["data_lock", "data_cutoff"]) &&
    existing.calculation_date === valueAt(c, ["data_lock", "calculation_date"]) &&
    existing.report_version === valueAt(c, ["versions", "report_version"]) &&
    existing.method_version === valueAt(c, ["versions", "method_version"]) &&
    existing.calculation_version === valueAt(c, ["versions", "calculation_version"]) &&
    existing.evidence_ledger_version === valueAt(c, ["versions", "evidence_ledger_version"]) &&
    isDeepStrictEqual(existing.canonical_payload, c),
    "D5_HISTORY_EXISTING_SNAPSHOT_CONFLICT");
  requireTrue(d.current_snapshot_id === snapshotId,
    "D5_HISTORY_IDEMPOTENT_REPLAY_POINTER_MISMATCH");
  return {
    mode: "IDENTICAL_REPLAY_PRECHECK",
    observedCurrentSnapshotId: d.current_snapshot_id,
    expectedCurrentSnapshotId: input.expectedBaselineId,
    candidateSnapshotId: snapshotId,
    historicalRowsConsistent: true,
    transactionCasValidated: false,
    historyTransitionPass: false,
    publicationAuthorized: false,
  };
}
