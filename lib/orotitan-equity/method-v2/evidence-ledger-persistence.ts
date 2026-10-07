export const METHOD_V2_EVIDENCE_LEDGER_FORMAT = "OROTITAN_METHOD_V2_EVIDENCE_LEDGER";
export const METHOD_V2_EVIDENCE_LEDGER_VERSION = "1.0";

export type MethodV2EvidenceLedgerEntry = Record<string, unknown> & {
  EVIDENCE_ID: string;
};

export interface MethodV2EvidenceLedger {
  format: typeof METHOD_V2_EVIDENCE_LEDGER_FORMAT;
  version: typeof METHOD_V2_EVIDENCE_LEDGER_VERSION;
  entries: MethodV2EvidenceLedgerEntry[];
  [key: string]: unknown;
}

function fail(code: string): never {
  throw new Error(code);
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parseMethodV2EvidenceLedger(bytes: Uint8Array): MethodV2EvidenceLedger {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return fail("METHOD_V2_EVIDENCE_LEDGER_INVALID_UTF8");
  }

  let candidate: unknown;
  try {
    candidate = JSON.parse(text);
  } catch {
    return fail("METHOD_V2_EVIDENCE_LEDGER_INVALID_JSON");
  }

  if (!isObject(candidate)) fail("METHOD_V2_EVIDENCE_LEDGER_INVALID_ROOT");
  if (candidate.format !== METHOD_V2_EVIDENCE_LEDGER_FORMAT) {
    fail("METHOD_V2_EVIDENCE_LEDGER_INVALID_FORMAT");
  }
  if (candidate.version !== METHOD_V2_EVIDENCE_LEDGER_VERSION) {
    fail("METHOD_V2_EVIDENCE_LEDGER_INVALID_VERSION");
  }
  if (!Array.isArray(candidate.entries)) fail("METHOD_V2_EVIDENCE_LEDGER_INVALID_ENTRIES");

  const evidenceIds = new Set<string>();
  for (const entry of candidate.entries) {
    if (!isObject(entry)) fail("METHOD_V2_EVIDENCE_LEDGER_INVALID_ENTRY");
    if (!Object.prototype.hasOwnProperty.call(entry, "EVIDENCE_ID")
        || typeof entry.EVIDENCE_ID !== "string"
        || entry.EVIDENCE_ID.length === 0) {
      fail("METHOD_V2_EVIDENCE_LEDGER_INVALID_EVIDENCE_ID");
    }
    if (evidenceIds.has(entry.EVIDENCE_ID)) {
      fail("METHOD_V2_EVIDENCE_LEDGER_DUPLICATE_EVIDENCE_ID");
    }
    evidenceIds.add(entry.EVIDENCE_ID);
  }

  return candidate as MethodV2EvidenceLedger;
}

export function findMethodV2EvidenceEntry(
  bytes: Uint8Array,
  evidenceId: string,
): MethodV2EvidenceLedgerEntry | undefined {
  return parseMethodV2EvidenceLedger(bytes).entries.find(entry => entry.EVIDENCE_ID === evidenceId);
}

export function assertMethodV2EvidenceIdExists(
  bytes: Uint8Array,
  evidenceId: string,
): MethodV2EvidenceLedgerEntry {
  const entry = findMethodV2EvidenceEntry(bytes, evidenceId);
  if (!entry) fail("METHOD_V2_EVIDENCE_LEDGER_EVIDENCE_ID_NOT_FOUND");
  return entry;
}
