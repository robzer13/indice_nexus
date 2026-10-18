/**
 * D4 — read-only independent I2 recomputation and V3.1 integration-candidate
 * admission checks. A valid candidate is NOT a historical CAS authorization.
 *
 * All input bytes MUST have been cryptographically verified by D2 and linked
 * through the active Integration FINAL manifest. No report's text PASS is used.
 */
import Ajv2020 from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { readFileSync } from "node:fs";
import { computeCanonicalSnapshot } from "../v1/contract";
import { validateV2ResearchSnapshotForPersistence } from "../v2/research-snapshot-schema";
import type { VerifiedContent } from "./pilotage-typed-gates-v3-1";
import { resolveValuationDateAlignment, type ValuationDateAlignmentInput } from "./valuation-date-alignment";

const schema = JSON.parse(readFileSync(
  new URL("../../../contracts/orotitan-equity/v3/04_SCREENER_SCHEMA_V3_1.json", import.meta.url),
  "utf8",
)) as Record<string, unknown>;
const ajv = new Ajv2020({ strict: false, allErrors: true });
addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);
const validateV31 = ajv.compile(schema);
const ISO = /^\d{4}-\d{2}-\d{2}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function object(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}
export class IntegrationD4Error extends Error {
  constructor(readonly code: string) { super(code); }
}
function requireTruth(v: unknown, code: string): asserts v {
  if (!v) throw new IntegrationD4Error(code);
}
function parseCandidate(c: VerifiedContent): Record<string, unknown> {
  requireTruth(c.proof.available && c.proof.bytesVerified &&
    c.proof.stageCode === "INTEGRATION" &&
    c.proof.artifactType === "CANONICAL_SNAPSHOT_CANDIDATE" &&
    c.mediaType === "application/json", "D4_CANDIDATE_NOT_BYTE_VERIFIED");
  try {
    const x: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(c.bytes));
    requireTruth(object(x), "D4_CANDIDATE_NOT_OBJECT");
    return x;
  } catch (e) {
    if (e instanceof IntegrationD4Error) throw e;
    throw new IntegrationD4Error("D4_CANDIDATE_INVALID_JSON");
  }
}
function requireDate(v: unknown, code: string): asserts v is string {
  requireTruth(typeof v === "string" && ISO.test(v) &&
    !Number.isNaN(Date.parse(v + "T00:00:00Z")), code);
}
function asObject(v: unknown, code: string): Record<string, unknown> {
  requireTruth(object(v), code);
  return v;
}
function alignBoolean(v: unknown): v is boolean { return typeof v === "boolean"; }
function v3CrossFields(candidate: Record<string, unknown>, expectedCutoff: string,
  upstreamTiming: ValuationDateAlignmentInput | null): {
  denominatorStructurallyAdmitted: boolean;
  timingStructurallyAdmitted: boolean;
  upstreamTimingReconciled: boolean;
} {
  const method = asObject(candidate.v3_methodology, "D4_V3_METHOD_MISSING");
  const share = asObject(method.economic_share_count, "D4_DENOMINATOR_MISSING");
  const timing = asObject(method.valuation_timing, "D4_TIMING_MISSING");
  const lock = asObject(candidate.data_lock, "D4_DATA_LOCK_MISSING");
  requireDate(expectedCutoff, "D4_EXPECTED_CUTOFF_INVALID");
  requireTruth(lock.data_cutoff === expectedCutoff, "D4_DATA_CUTOFF_MISMATCH");
  for (const field of ["valuation_date", "time_origin", "economic_share_count_date",
    "share_count_as_of_date", "per_share_output_date"]) {
    requireTruth(timing[field] === expectedCutoff, "D4_TIMING_TUPLE_MISMATCH_" + field);
  }
  requireTruth(share.share_count_as_of_date === expectedCutoff, "D4_DENOMINATOR_DATE_MISMATCH");
  requireDate(timing.reference_price_date, "D4_REFERENCE_DATE_INVALID");
  requireTruth(timing.reference_price_date <= expectedCutoff, "D4_REFERENCE_PRICE_FUTURE");
  requireTruth(lock.reference_price_date === timing.reference_price_date,
    "D4_REFERENCE_PRICE_DATE_NOT_TRACEABLE");
  if (typeof lock.reference_price === "number") {
    requireTruth(Number.isFinite(lock.reference_price) && lock.reference_price > 0,
      "D4_REFERENCE_PRICE_INVALID");
    requireTruth(timing.market_cap_date === timing.reference_price_date ||
      timing.market_cap_date === "NOT_ASSESSABLE", "D4_MARKET_CAP_DATE_MISMATCH");
  }
  const rep = share.representation;
  requireTruth(rep === "EXACT" || rep === "BOUNDED", "D4_UNKNOWN_DENOMINATOR_CANNOT_FINALIZE");
  requireTruth(share.valuation_admission === "YES", "D4_DENOMINATOR_NOT_ADMITTED");
  if (rep === "EXACT") {
    requireTruth(Number.isSafeInteger(share.exact_count) && (share.exact_count as number) > 0 &&
      share.propagation_status === "SCALAR_EXACT", "D4_EXACT_DENOMINATOR_INVALID");
  } else {
    requireTruth(Number.isSafeInteger(share.lower_bound) &&
      Number.isSafeInteger(share.upper_bound) &&
      (share.lower_bound as number) > 0 &&
      (share.lower_bound as number) <= (share.upper_bound as number) &&
      share.bound_completeness === "COMPLETE" &&
      share.unbounded_movement_classes === 0 &&
      share.propagation_status === "RANGE_PROPAGATED", "D4_BOUNDED_DENOMINATOR_INVALID");
  }
  let upstreamTimingReconciled = false;
  if (upstreamTiming) {
    // D3 must provide this from the exact, SHA-verified VALUATION_ARTIFACT,
    // never from a report / client request.
    requireTruth(upstreamTiming.dataCutoff === expectedCutoff &&
      upstreamTiming.valuationDate === timing.valuation_date &&
      upstreamTiming.timeOrigin === timing.time_origin &&
      upstreamTiming.economicShareCountDate === timing.economic_share_count_date &&
      upstreamTiming.shareCountAsOfDate === timing.share_count_as_of_date &&
      upstreamTiming.perShareOutputDate === timing.per_share_output_date &&
      upstreamTiming.referencePriceDate === timing.reference_price_date,
      "D4_UPSTREAM_TIMING_CONFLICT");
    requireTruth(upstreamTiming.economicShareCountRepresentation === share.representation,
      "D4_UPSTREAM_DENOMINATOR_REPRESENTATION_CONFLICT");
    const b = upstreamTiming.cashFlowBasis;
    requireTruth((b === "FCFF" && timing.ev_to_equity_bridge_date === expectedCutoff) ||
      ((b === "FCFE" || b === "OWNER_EARNINGS") &&
        timing.ev_to_equity_bridge_date === "NOT_APPLICABLE"),
      "D4_DCF_BASIS_EV_BRIDGE_MISMATCH");
    if (upstreamTiming.marketCapShareCountResolved &&
      upstreamTiming.hasNumericReferencePrice) {
      requireTruth(timing.market_cap_date === timing.reference_price_date,
        "D4_MARKET_CAP_NOT_SAME_DATE");
    }
    requireTruth(alignBoolean(upstreamTiming.denominatorBridgeResolved) &&
      alignBoolean(upstreamTiming.evBridgeRollforwardResolved) &&
      alignBoolean(upstreamTiming.referencePriceStalenessResolved) &&
      alignBoolean(upstreamTiming.marketCapShareCountResolved) &&
      alignBoolean(upstreamTiming.hasNumericReferencePrice),
      "D4_UPSTREAM_TIMING_FLAGS_INVALID");
    const resolved = resolveValuationDateAlignment(upstreamTiming);
    requireTruth(resolved.valuationAdmission && resolved.blockers.length === 0,
      "D4_UPSTREAM_VALUATION_ADMISSION_BLOCKED");
    upstreamTimingReconciled = true;
  }
  return {
    denominatorStructurallyAdmitted: true,
    timingStructurallyAdmitted: true,
    upstreamTimingReconciled,
  };
}
export interface D4CandidateCheck {
  /** I2 source-core validator and deterministic scores recomputed from candidate. */
  i2CandidateRecomputed: true;
  /** V2 product + V3.1 JSON schema + temporal/denominator structure passed. */
  v31CandidateSchemaPassed: true;
  /** Independent match to an already byte-verified Deep Dive timing record. */
  upstreamTimingReconciled: boolean;
  /** No historic pointer/CAS or analytical evidence replay was validated here. */
  historyCasValidated: false;
  /** A candidate result is NOT the end-to-end I2 or I3-B admission gate. */
  i2Pass: false;
  i3bPass: false;
  canonicalCandidateAdmitted: false;
  publicationAuthorized: false;
  computed: ReturnType<typeof computeCanonicalSnapshot>;
}

/**
 * Recomputes frozen I2 formulas and V3.1 composed payload semantics.
 * Does NOT dispatch a writer, publish, or permit an Integration FINAL gate.
 */
export function verifyIntegrationCandidateD4(input: {
  candidate: VerifiedContent;
  dossierId: string;
  issuerId: string;
  securityId: string;
  dataCutoff: string;
  upstreamTiming: ValuationDateAlignmentInput | null;
}): D4CandidateCheck {
  requireTruth(UUID.test(input.dossierId) && UUID.test(input.issuerId) &&
    UUID.test(input.securityId), "D4_RUN_IDENTITY_INVALID");
  const candidate = parseCandidate(input.candidate);
  requireTruth(candidate.issuer_id === input.issuerId &&
    candidate.security_id === input.securityId, "D4_CANDIDATE_IDENTITY_MISMATCH");
  requireTruth(candidate.execution_mode === "ANALYZE" ||
    candidate.execution_mode === "REFRESH" ||
    candidate.execution_mode === "DISCOVER + ANALYZE", "D4_NON_ANALYTICAL_CANDIDATE");
  requireTruth(validateV31(candidate), "D4_V31_OVERLAY_SCHEMA_INVALID");
  const v2 = { ...candidate };
  delete v2.v3_methodology;
  const validated = validateV2ResearchSnapshotForPersistence(v2, input.dossierId);
  requireTruth(validated.ok, "D4_I2_OR_V2_RECONCILIATION_FAILED");
  // The V2 validator independently recalculates every persisted deterministic
  // score and terminal predicate through the frozen V1 I2 computation engine.
  const computed = computeCanonicalSnapshot(validated.canonicalContract as Parameters<typeof computeCanonicalSnapshot>[0]);
  const verified = v3CrossFields(candidate, input.dataCutoff, input.upstreamTiming);
  return {
    i2CandidateRecomputed: true,
    v31CandidateSchemaPassed: true,
    upstreamTimingReconciled: verified.upstreamTimingReconciled,
    historyCasValidated: false, i2Pass: false, i3bPass: false,
    canonicalCandidateAdmitted: false, publicationAuthorized: false, computed,
  };
}
