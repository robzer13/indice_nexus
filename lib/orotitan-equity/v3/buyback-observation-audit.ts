/**
 * Non-authoritative audit of disclosed buyback trade observations.
 * Dates and URLs are caller assertions, not admitted original source bytes.
 * Trade volume is NOT the X -> T economic-share transition or a denominator.
 */
export type BuybackTradeObservation = {
  tradeId: string;
  tradeDate: string;
  sharesPurchased: number;
};

export type BuybackIssuerNotice = {
  noticeId: string;
  publicationDate: string;
  sourceUrl: string;
  trades: BuybackTradeObservation[];
};

export type BuybackObservationBatch = {
  dataCutoff: string;
  notices: BuybackIssuerNotice[];
};

export type BuybackObservationAudit = {
  status: "OBSERVED_UNATTESTED";
  dataCutoff: string;
  noticeCount: number;
  tradeRowCount: number;
  observedPurchasedShares: number;
  firstTradeDate: string;
  lastTradeDate: string;
  sourceBytesAdmitted: false;
  movementClassCompleteness: "NOT_EVALUATED";
  economicShareCount: "UNKNOWN";
  denominatorValuationAdmission: "NO";
};

export class BuybackObservationAuditError extends Error {
  readonly code: string;
  constructor(code: string, message: string) {
    super(code + ": " + message);
    this.code = code;
    this.name = "BuybackObservationAuditError";
  }
}

function fail(code: string, message: string): never {
  throw new BuybackObservationAuditError(code, message);
}

function date(value: string, label: string): string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    fail("INVALID_DATE", label + " must be a YYYY-MM-DD civil date");
  }
  const parsed = new Date(value + "T00:00:00Z");
  if (!Number.isFinite(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== value) {
    fail("INVALID_DATE", label + " is not a real calendar date");
  }
  return value;
}

function identifier(value: string, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    fail("MISSING_IDENTIFIER", label + " must be non-empty");
  }
  return value;
}

function validateSourceLocator(value: string): void {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    fail("INVALID_SOURCE_URL", "source locator must be HTTPS");
  }
  if (url.protocol !== "https:" || !url.hostname || url.username || url.password) {
    fail("INVALID_SOURCE_URL", "source locator must be HTTPS without credentials");
  }
}

/**
 * Strictly validates internal batch arithmetic and publication-date cutoff.
 * Never creates a source attestation, complete share-movement ledger or
 * valuation-admissible denominator; invalid rows reject the entire batch.
 */
export function auditUnattestedBuybackObservations(
  input: BuybackObservationBatch,
): BuybackObservationAudit {
  const cutoff = date(input.dataCutoff, "DATA_CUTOFF");
  if (!Array.isArray(input.notices) || input.notices.length === 0) {
    fail("NO_NOTICES", "at least one notice is required");
  }

  const noticeIds = new Set<string>();
  const tradeIds = new Set<string>();
  let total = 0;
  let rowCount = 0;
  let first = "";
  let last = "";

  for (const notice of input.notices) {
    const id = identifier(notice.noticeId, "noticeId");
    if (noticeIds.has(id)) fail("DUPLICATE_NOTICE", "duplicate notice identifier");
    noticeIds.add(id);
    validateSourceLocator(notice.sourceUrl);
    const published = date(notice.publicationDate, "publicationDate");
    if (published > cutoff) fail("POST_CUTOFF_NOTICE", "notice was published after cutoff");
    if (!Array.isArray(notice.trades) || notice.trades.length === 0) {
      fail("EMPTY_NOTICE", "notice requires at least one trade");
    }

    for (const row of notice.trades) {
      const tradeId = identifier(row.tradeId, "tradeId");
      if (tradeIds.has(tradeId)) fail("DUPLICATE_TRADE", "duplicate trade identifier");
      tradeIds.add(tradeId);
      const traded = date(row.tradeDate, "tradeDate");
      if (traded > published) fail("TRADE_AFTER_NOTICE", "trade date is after release");
      if (!Number.isSafeInteger(row.sharesPurchased) || row.sharesPurchased <= 0) {
        fail("INVALID_SHARE_QUANTITY", "positive safe-integer shares required");
      }
      if (!Number.isSafeInteger(total + row.sharesPurchased)) {
        fail("UNSAFE_SHARE_TOTAL", "total exceeds safe-integer precision");
      }
      total += row.sharesPurchased;
      rowCount += 1;
      if (first === "" || traded < first) first = traded;
      if (last === "" || traded > last) last = traded;
    }
  }

  return {
    status: "OBSERVED_UNATTESTED",
    dataCutoff: cutoff,
    noticeCount: input.notices.length,
    tradeRowCount: rowCount,
    observedPurchasedShares: total,
    firstTradeDate: first,
    lastTradeDate: last,
    sourceBytesAdmitted: false,
    movementClassCompleteness: "NOT_EVALUATED",
    economicShareCount: "UNKNOWN",
    denominatorValuationAdmission: "NO",
  };
}
