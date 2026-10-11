import type { ArtifactRef } from "./pre-certification";

export const METHOD_V2_CERTIFICATION_FORMAT = "OROTITAN_METHOD_V2_CERTIFICATION_ARTIFACT";
export const METHOD_V2_CERTIFICATION_VERSION = "1.0";
export type ChallengeLimitation = {
  question_id: string; decision_impact: string; mitigation_or_resolution: string;
};
export type MethodV2CertificationChallengeBinding = {
  question_ledger_ref: ArtifactRef; challenge_report_ref: ArtifactRef;
  fundamentals_lock_ref: ArtifactRef; valuation_lock_ref: ArtifactRef;
  challenge_status: "PASS" | "PASS_WITH_CONCERNS";
  challenge_limitations: ChallengeLimitation[];
};
export type MethodV2CertificationExpectedContext = MethodV2CertificationChallengeBinding & {
  run_id: string; stage_revision: number; data_cutoff: string;
};
export type MethodV2CertificationArtifact = {
  format: typeof METHOD_V2_CERTIFICATION_FORMAT;
  version: typeof METHOD_V2_CERTIFICATION_VERSION;
  run_id: string; stage_revision: number; data_cutoff: string;
  certification: Record<string, unknown>;
  method_v2_challenge_binding: MethodV2CertificationChallengeBinding;
};

function fail(code: string): never { throw new Error(`METHOD_V2_CERTIFICATION_${code}`); }
function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function exactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  return Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key));
}
const refKeys = ["artifact_id", "version", "content_sha256"] as const;
const bindingRefs = ["question_ledger_ref", "challenge_report_ref", "fundamentals_lock_ref", "valuation_lock_ref"] as const;
const limitationKeys = ["question_id", "decision_impact", "mitigation_or_resolution"] as const;
const uuid = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;
const positiveInteger = (value: unknown): boolean => typeof value === "number" && Number.isSafeInteger(value) && value >= 1;
function assertIdentity(value: Record<string, unknown>): void {
  if (typeof value.run_id !== "string" || !uuid.test(value.run_id)) fail("INVALID_RUN_ID");
  if (!positiveInteger(value.stage_revision)) fail("INVALID_STAGE_REVISION");
  if (typeof value.data_cutoff !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value.data_cutoff)) fail("INVALID_DATA_CUTOFF");
  const [year, month, day] = value.data_cutoff.split("-").map(Number);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  if (month < 1 || month > 12 || day < 1 || day > days[month - 1]) fail("INVALID_DATA_CUTOFF");
}
function assertBinding(value: unknown): asserts value is MethodV2CertificationChallengeBinding {
  if (!object(value) || !exactKeys(value, [...bindingRefs, "challenge_status", "challenge_limitations"])) fail("INVALID_BINDING");
  for (const key of bindingRefs) {
    const ref = value[key];
    if (!object(ref) || !exactKeys(ref, refKeys) || typeof ref.artifact_id !== "string" || !uuid.test(ref.artifact_id)
        || !positiveInteger(ref.version) || typeof ref.content_sha256 !== "string" || !/^[0-9a-f]{64}$/.test(ref.content_sha256)) {
      fail("INVALID_ARTIFACT_REF");
    }
  }
  if (value.challenge_status !== "PASS" && value.challenge_status !== "PASS_WITH_CONCERNS") fail("INVALID_CHALLENGE_STATUS");
  if (!Array.isArray(value.challenge_limitations)) fail("INVALID_CHALLENGE_LIMITATIONS");
  const ids = new Set<string>();
  for (const limitation of value.challenge_limitations) {
    if (!object(limitation) || !exactKeys(limitation, limitationKeys)
        || !limitationKeys.every(key => typeof limitation[key] === "string" && limitation[key].length > 0)) fail("INVALID_CHALLENGE_LIMITATION");
    const id = limitation.question_id as string;
    if (ids.has(id)) fail("DUPLICATE_QUESTION_ID");
    ids.add(id);
  }
  if (value.challenge_status === "PASS" && ids.size !== 0) fail("PASS_LIMITATIONS_NOT_EMPTY");
  if (value.challenge_status === "PASS_WITH_CONCERNS" && ids.size === 0) fail("CONCERNS_ABSENT");
}
function assertArtifact(value: unknown): asserts value is MethodV2CertificationArtifact {
  if (!object(value)) fail("INVALID_ROOT");
  if (value.format !== METHOD_V2_CERTIFICATION_FORMAT) fail("INVALID_FORMAT");
  if (value.version !== METHOD_V2_CERTIFICATION_VERSION) fail("INVALID_VERSION");
  if (!exactKeys(value, ["format", "version", "run_id", "stage_revision", "data_cutoff", "certification", "method_v2_challenge_binding"])) fail("INVALID_ROOT_MEMBERS");
  assertIdentity(value);
  if (!object(value.certification)) fail("INVALID_CERTIFICATION_PAYLOAD");
  assertBinding(value.method_v2_challenge_binding);
}

/** Structural scan before JSON.parse: decoded names, including escaped aliases,
 * must be unique in every object. Based on the Evidence Ledger scanner precedent.
 * Certification values are opaque; this checks only unambiguous JSON structure. */
class UniqueMemberScanner {
  private index = 0;
  constructor(private readonly text: string) {}
  private whitespace(): void {
    while (/[\t\n\r ]/.test(this.text[this.index] ?? "")) this.index++;
  }
  scan(): void {
    this.value(); this.whitespace();
    if (this.index !== this.text.length) fail("INVALID_JSON");
  }
  private string(): string {
    const start = this.index++;
    while (this.index < this.text.length) {
      const char = this.text[this.index++];
      if (char === '"') {
        try { return JSON.parse(this.text.slice(start, this.index)) as string; }
        catch { fail("INVALID_JSON"); }
      }
      if (char === "\\") this.index++;
    }
    return fail("INVALID_JSON");
  }
  private value(): void {
    this.whitespace();
    const token = this.text[this.index];
    if (token === '"') { this.string(); return; }
    if (token === "{" || token === "[") {
      this.index++; this.whitespace();
      const end = token === "{" ? "}" : "]";
      const keys = new Set<string>();
      if (this.text[this.index] === end) { this.index++; return; }
      while (this.index < this.text.length) {
        if (token === "{") {
          if (this.text[this.index] !== '"') fail("INVALID_JSON");
          const key = this.string();
          if (keys.has(key)) fail("DUPLICATE_JSON_MEMBER");
          keys.add(key); this.whitespace();
          if (this.text[this.index++] !== ":") fail("INVALID_JSON");
        }
        this.value(); this.whitespace();
        if (this.text[this.index] === end) { this.index++; return; }
        if (this.text[this.index++] !== ",") fail("INVALID_JSON");
        this.whitespace();
      }
      fail("INVALID_JSON");
    }
    const match = /^(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/.exec(this.text.slice(this.index));
    if (!match) fail("INVALID_JSON");
    this.index += match[0].length;
  }
}

export function parseMethodV2CertificationArtifact(bytes: Uint8Array): MethodV2CertificationArtifact {
  let text: string;
  try { text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes); }
  catch { return fail("INVALID_UTF8"); }
  // A BOM is not JSON whitespace and is deliberately rejected.
  new UniqueMemberScanner(text).scan();
  let artifact: unknown;
  try { artifact = JSON.parse(text); }
  catch { return fail("INVALID_JSON"); }
  assertArtifact(artifact);
  return artifact;
}

function deterministicJson(value: unknown): string {
  if (value === null || typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "number" && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return `[${Array.from(value, deterministicJson).join(",")}]`;
  if (object(value) && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null)) {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${deterministicJson(value[key])}`).join(",")}}`;
  }
  return fail("NON_JSON_SERIALIZATION_VALUE");
}

/** Deterministic writer; retains inherited array order and all string values. */
export function serializeMethodV2CertificationArtifact(artifact: MethodV2CertificationArtifact): Uint8Array {
  assertArtifact(artifact);
  const binding = artifact.method_v2_challenge_binding;
  const challenge_limitations = [...binding.challenge_limitations].sort((a, b) =>
    a.question_id < b.question_id ? -1 : a.question_id > b.question_id ? 1 : 0);
  return new TextEncoder().encode(deterministicJson({ ...artifact,
    method_v2_challenge_binding: { ...binding, challenge_limitations } }));
}

/** Trusted context must come from current, independently validated Challenge
 * admission and persisted lineage. This function makes no Certification judgment. */
export function verifyMethodV2CertificationChallengeBinding(
  artifact: MethodV2CertificationArtifact,
  expectedContext: MethodV2CertificationExpectedContext,
): { verified: true } {
  assertArtifact(artifact);
  if (!object(expectedContext)) fail("INVALID_EXPECTED_CONTEXT");
  assertIdentity(expectedContext);
  const { run_id, stage_revision, data_cutoff, ...binding } = expectedContext;
  assertBinding(binding);
  if (artifact.run_id !== run_id || artifact.stage_revision !== stage_revision || artifact.data_cutoff !== data_cutoff) fail("CONTEXT_MISMATCH");
  const actual = artifact.method_v2_challenge_binding;
  for (const key of bindingRefs) {
    if (!refKeys.every(member => actual[key][member] === binding[key][member])) fail("ARTIFACT_REF_MISMATCH");
  }
  if (actual.challenge_status !== binding.challenge_status) fail("CHALLENGE_STATUS_MISMATCH");
  const expected = new Map(binding.challenge_limitations.map(item => [item.question_id, item]));
  if (actual.challenge_limitations.length !== expected.size || !actual.challenge_limitations.every(item => {
    const counterpart = expected.get(item.question_id);
    return counterpart !== undefined && limitationKeys.every(key => item[key] === counterpart[key]);
  })) fail("CHALLENGE_LIMITATIONS_MISMATCH");
  return { verified: true };
}
