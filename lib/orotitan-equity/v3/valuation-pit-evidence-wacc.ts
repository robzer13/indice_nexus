/**
 * Generic, side-effect-free PIT evidence and WACC preflight.
 *
 * Public application code: NO issuer facts, no trusted source-byte claims, no
 * Registry mutations, no source fetching, and no authority/gate promotion.
 * The caller MUST provide independently reviewed anchors and trusted provenance.
 */
import { createHash } from "node:crypto";

export type EvidenceClassification =
  | "OBSERVED"
  | "HISTORICAL_PUBLICATION_UNPROVEN"
  | "ANNOUNCED_NOT_SETTLED";

export type PitEvidenceClaim = {
  id: string;
  sourceUrl: string;
  sourceLocator: string;
  economicDate: string;
  publishedAt: string | null;
  classification: EvidenceClassification;
  scheduledSettlementDate?: string;
  value: string;
  unit: string;
};

export type PitEvidencePreflight = {
  status: "LOCAL_TRANSCRIPT_CHECK_ONLY";
  evidenceCount: number;
  allPublicationDatesVerified: boolean;
  originalSourceBytesVerified: false;
  blockers: string[];
};

export class PitEvidenceWaccError extends Error {
  readonly code: string;
  constructor(code: string) {
    super(code);
    this.code = code;
    this.name = "PitEvidenceWaccError";
  }
}

function need(condition: unknown, code: string): asserts condition {
  if (!condition) throw new PitEvidenceWaccError(code);
}

function validDate(s: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) return false;
  const d = new Date(s + "T00:00:00.000Z");
  return Number.isFinite(d.getTime()) && d.toISOString().slice(0, 10) === s;
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    const o = value as Record<string, unknown>;
    const keys = Object.keys(o).sort();
    return Object.fromEntries(keys.map(k => [k, canonical(o[k])]));
  }
  return value;
}

/** Integrity of editorial transcription, NOT the external PDF or HTML. */
export function transcriptSha256(claim: PitEvidenceClaim): string {
  return createHash("sha256").update(JSON.stringify(canonical(claim)), "utf8").digest("hex");
}

/**
 * Trusted anchors and host allowlist must come from a server-owned review,
 * never untrusted request data. A matching hash is not an issuer attestation.
 */
export function preflightPitEvidence(input: {
  cutoff: string;
  claims: readonly PitEvidenceClaim[];
  trustedTranscriptSha256: Readonly<Record<string, string>>;
  allowedHosts: readonly string[];
}): PitEvidencePreflight {
  need(validDate(input.cutoff), "INVALID_CUTOFF_DATE");
  need(input.claims.length > 0 && input.claims.length <= 3000,
    "EVIDENCE_COUNT_INVALID");
  need(input.allowedHosts.length > 0 && input.allowedHosts.every(h =>
    /^[a-z0-9.-]+$/.test(h) && !h.startsWith(".") && !h.endsWith(".")),
  "SOURCE_HOST_ALLOWLIST_INVALID");

  const anchorIds = Object.keys(input.trustedTranscriptSha256);
  need(anchorIds.length === input.claims.length, "TRANSCRIPT_SET_MISMATCH");
  const seen = new Set<string>();
  const blockers = new Set<string>();
  for (const claim of input.claims) {
    need(claim && typeof claim.id === "string" && claim.id.length > 0 && !seen.has(claim.id),
      "DUPLICATE_OR_INVALID_EVIDENCE_ID");
    seen.add(claim.id);
    const pin = input.trustedTranscriptSha256[claim.id];
    need(typeof pin === "string" && /^[0-9a-f]{64}$/.test(pin),
      "TRANSCRIPT_ANCHOR_MISSING_OR_INVALID");
    need(transcriptSha256(claim) === pin, "TRANSCRIPT_BYTES_CHANGED");

    let parsed: URL;
    try { parsed = new URL(claim.sourceUrl); }
    catch { throw new PitEvidenceWaccError("SOURCE_URL_INVALID"); }
    need(parsed.protocol === "https:" && !parsed.username && !parsed.password &&
      !parsed.port && !parsed.hash && input.allowedHosts.includes(parsed.hostname),
      "SOURCE_ORIGIN_NOT_APPROVED");
    need(typeof claim.sourceLocator === "string" && claim.sourceLocator.trim().length >= 8,
      "SOURCE_LOCATOR_MISSING");
    need(typeof claim.unit === "string" && claim.unit.length > 0 &&
      typeof claim.value === "string" && claim.value.length > 0,
      "SOURCE_VALUE_OR_UNIT_INVALID");
    need(validDate(claim.economicDate), "ECONOMIC_DATE_INVALID");
    need(claim.economicDate <= input.cutoff, "ECONOMIC_EVENT_AFTER_CUTOFF");
    need(["OBSERVED", "HISTORICAL_PUBLICATION_UNPROVEN", "ANNOUNCED_NOT_SETTLED"]
      .includes(claim.classification), "EVIDENCE_CLASSIFICATION_INVALID");

    if (claim.publishedAt === null) {
      need(claim.classification === "HISTORICAL_PUBLICATION_UNPROVEN",
        "UNVERIFIED_PUBLICATION_UNDISCLOSED");
      blockers.add("PUBLICATION_TIME_NOT_PROVEN");
    } else {
      need(validDate(claim.publishedAt), "PUBLICATION_DATE_INVALID");
      need(claim.publishedAt <= input.cutoff, "SOURCE_PUBLISHED_AFTER_CUTOFF");
    }
    if (claim.classification === "ANNOUNCED_NOT_SETTLED") {
      need(typeof claim.scheduledSettlementDate === "string" &&
        validDate(claim.scheduledSettlementDate), "SETTLEMENT_DATE_MISSING");
      need(claim.scheduledSettlementDate > input.cutoff,
        "FINANCING_WRONGLY_TREATED_AS_UNSETTLED");
      blockers.add("FINANCING_NOT_SETTLED_AT_CUTOFF");
    } else {
      need(claim.scheduledSettlementDate === undefined,
        "UNEXPECTED_SETTLEMENT_DATE");
    }
  }
  need(seen.size === anchorIds.length && anchorIds.every(id => seen.has(id)),
    "TRANSCRIPT_SET_MISMATCH");

  // Never report original byte authentication: this function does not fetch
  // the source or compare its original bytes to a trusted archival hash.
  blockers.add("ORIGINAL_SOURCE_BYTES_NOT_ATTESTED");
  return {
    status: "LOCAL_TRANSCRIPT_CHECK_ONLY",
    evidenceCount: seen.size,
    allPublicationDatesVerified: !blockers.has("PUBLICATION_TIME_NOT_PROVEN"),
    originalSourceBytesVerified: false,
    blockers: [...blockers].sort(),
  };
}

export type WaccDiagnosticInput = {
  unleveredBeta: number;
  taxRate: number;
  equityWeight: number;
  debtWeight: number;
  riskFreeRatePct: number;
  equityRiskPremiumPct: number;
  marginalDebtCostPct: number;
  cashFlowCurrency: string;
  riskFreeCurrency: string;
  equityRiskPremiumCurrency: string;
  betaMode: "RELEVER" | "EQUITY_BETA";
};

export type WaccDiagnostic = {
  status: "ILLUSTRATIVE_NOT_CERTIFIED";
  leveredBeta: number;
  costOfEquityPct: number;
  waccPct: number;
  certified: false;
  blockers: string[];
};

export function computeDiagnosticWacc(input: WaccDiagnosticInput): WaccDiagnostic {
  const x = input;
  const values = [x.unleveredBeta, x.taxRate, x.equityWeight, x.debtWeight,
    x.riskFreeRatePct, x.equityRiskPremiumPct, x.marginalDebtCostPct];
  need(values.every(Number.isFinite), "WACC_NONFINITE_INPUT");
  need(x.unleveredBeta > 0 && x.unleveredBeta < 10 &&
    x.taxRate >= 0 && x.taxRate < 1 &&
    x.equityWeight > 0 && x.equityWeight <= 1 &&
    x.debtWeight >= 0 && x.debtWeight < 1 &&
    Math.abs(x.equityWeight + x.debtWeight - 1) < 1e-12 &&
    x.equityRiskPremiumPct > 0 && x.marginalDebtCostPct >= 0,
    "WACC_INPUT_DOMAIN_INVALID");
  need(x.betaMode === "RELEVER", "UNLEVERED_BETA_RELEVER_REQUIRED");
  need([x.cashFlowCurrency, x.riskFreeCurrency, x.equityRiskPremiumCurrency]
    .every(c => /^[A-Z]{3}$/.test(c)), "WACC_CURRENCY_CODE_INVALID");

  // Simplified Hamada relevering, assumes debt beta = 0. This assumption is
  // NOT an actual measurement and is not sufficient for a certified WACC.
  const de = x.debtWeight / x.equityWeight;
  const leveredBeta = x.unleveredBeta * (1 + (1 - x.taxRate) * de);
  const costOfEquityPct = x.riskFreeRatePct + leveredBeta * x.equityRiskPremiumPct;
  const waccPct = x.equityWeight * costOfEquityPct +
    x.debtWeight * x.marginalDebtCostPct * (1 - x.taxRate);
  const blockers = ["BETA_DEBT_ASSUMED_ZERO", "CAPITAL_WEIGHTS_AND_COST_OF_DEBT_NOT_SOURCE_VERIFIED"];
  if (x.cashFlowCurrency !== x.riskFreeCurrency ||
      x.cashFlowCurrency !== x.equityRiskPremiumCurrency) {
    blockers.push("DISCOUNT_RATE_CURRENCY_INCONSISTENT");
  }
  return {
    status: "ILLUSTRATIVE_NOT_CERTIFIED", leveredBeta, costOfEquityPct,
    waccPct, certified: false, blockers,
  };
}
