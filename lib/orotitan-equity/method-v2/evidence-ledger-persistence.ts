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

class StructuralJsonScanner {
  private index = 0;

  constructor(private readonly text: string) {}

  scan(): void {
    this.skipWhitespace();
    this.scanValue({ root: true });
    this.skipWhitespace();
    if (this.index !== this.text.length) this.invalidJson();
  }

  private invalidJson(): never {
    throw new Error("METHOD_V2_EVIDENCE_LEDGER_INVALID_JSON");
  }

  private skipWhitespace(): void {
    while (this.index < this.text.length && /[\u0009\u000a\u000d\u0020]/.test(this.text[this.index])) {
      this.index += 1;
    }
  }

  private scanValue(context: { root?: boolean; entriesValue?: boolean; directEntry?: boolean } = {}): void {
    this.skipWhitespace();
    const token = this.text[this.index];
    if (token === "{") {
      this.scanObject({ root: context.root, directEntry: context.directEntry });
      return;
    }
    if (token === "[") {
      this.scanArray(context.entriesValue === true);
      return;
    }
    if (token === '"') {
      this.scanString();
      return;
    }
    if (token === "t") return this.scanLiteral("true");
    if (token === "f") return this.scanLiteral("false");
    if (token === "n") return this.scanLiteral("null");
    this.scanNumber();
  }

  private scanObject(context: { root?: boolean; directEntry?: boolean }): void {
    this.index += 1;
    this.skipWhitespace();
    let evidenceIdMemberSeen = false;
    if (this.text[this.index] === "}") {
      this.index += 1;
      return;
    }

    while (this.index < this.text.length) {
      if (this.text[this.index] !== '"') this.invalidJson();
      const key = this.scanString();
      if (context.directEntry && key === "EVIDENCE_ID") {
        if (evidenceIdMemberSeen) {
          fail("METHOD_V2_EVIDENCE_LEDGER_DUPLICATE_EVIDENCE_ID_MEMBER");
        }
        evidenceIdMemberSeen = true;
      }
      this.skipWhitespace();
      if (this.text[this.index] !== ":") this.invalidJson();
      this.index += 1;
      this.scanValue({ entriesValue: context.root && key === "entries" });
      this.skipWhitespace();
      if (this.text[this.index] === "}") {
        this.index += 1;
        return;
      }
      if (this.text[this.index] !== ",") this.invalidJson();
      this.index += 1;
      this.skipWhitespace();
    }
    this.invalidJson();
  }

  private scanArray(entriesArray: boolean): void {
    this.index += 1;
    this.skipWhitespace();
    if (this.text[this.index] === "]") {
      this.index += 1;
      return;
    }

    while (this.index < this.text.length) {
      this.scanValue({ directEntry: entriesArray });
      this.skipWhitespace();
      if (this.text[this.index] === "]") {
        this.index += 1;
        return;
      }
      if (this.text[this.index] !== ",") this.invalidJson();
      this.index += 1;
      this.skipWhitespace();
    }
    this.invalidJson();
  }

  private scanString(): string {
    const start = this.index;
    this.index += 1;
    while (this.index < this.text.length) {
      const character = this.text[this.index];
      if (character === '"') {
        this.index += 1;
        try {
          return JSON.parse(this.text.slice(start, this.index)) as string;
        } catch {
          return this.invalidJson();
        }
      }
      if (character === "\\") {
        this.index += 1;
        const escape = this.text[this.index];
        if (escape === "u") {
          if (!/^[0-9a-fA-F]{4}$/.test(this.text.slice(this.index + 1, this.index + 5))) this.invalidJson();
          this.index += 5;
          continue;
        }
        if (!['"', "\\", "/", "b", "f", "n", "r", "t"].includes(escape)) this.invalidJson();
        this.index += 1;
        continue;
      }
      if (character.charCodeAt(0) <= 0x1f) this.invalidJson();
      this.index += 1;
    }
    return this.invalidJson();
  }

  private scanLiteral(literal: string): void {
    if (this.text.slice(this.index, this.index + literal.length) !== literal) this.invalidJson();
    this.index += literal.length;
  }

  private scanNumber(): void {
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(this.text.slice(this.index));
    if (!match) this.invalidJson();
    this.index += match[0].length;
  }
}

function assertNoDuplicateEvidenceIdMembers(text: string): void {
  new StructuralJsonScanner(text).scan();
}

export function parseMethodV2EvidenceLedger(bytes: Uint8Array): MethodV2EvidenceLedger {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return fail("METHOD_V2_EVIDENCE_LEDGER_INVALID_UTF8");
  }

  assertNoDuplicateEvidenceIdMembers(text);

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
