import { createHash } from "node:crypto";
import {
  validateV2ResearchSnapshotForPersistenceWithLongThesis,
} from "./research-snapshot-schema-thesis-compat";

type JsonObject = Record<string, unknown>;
type ExactThesis = Readonly<{
  quality_case: string;
  valuation_case: string;
  key_risk: string;
}>;
export type SealedThesisArtifact = Readonly<{
  bytesUtf8: string;
  expectedSha256: string;
  expectedBytes: number;
  expectedVersion: number;
}>;
export type ExactThesisAdmissionInput = Readonly<{
  runId: string;
  dataCutoff: string;
  dossierId: string;
  issuerId: string;
  securityId: string;
  structuredThesis: SealedThesisArtifact;
  deepDiveReport: SealedThesisArtifact;
  canonicalCandidate: unknown;
}>;
export type ExactThesisAdmissionResult =
  | Readonly<{ admittedForValidation: true; sourceThesis: ExactThesis; errors: readonly [] }>
  | Readonly<{ admittedForValidation: false; sourceThesis: null; errors: readonly string[] }>;

function isObject(value: unknown): value is JsonObject {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function reject(...errors: string[]): ExactThesisAdmissionResult {
  return { admittedForValidation: false, sourceThesis: null, errors };
}

function readSealedJson(
  artifact: SealedThesisArtifact,
  expectedType: string,
  runId: string,
  dataCutoff: string,
): { valid: true; body: JsonObject } | { valid: false; reason: string } {
  if (!/^[0-9a-f]{64}$/i.test(artifact.expectedSha256) ||
      !Number.isSafeInteger(artifact.expectedBytes) || artifact.expectedBytes <= 0 ||
      !Number.isSafeInteger(artifact.expectedVersion) || artifact.expectedVersion < 1) {
    return { valid: false, reason: "SEALED_ARTIFACT_LOCATOR_INVALID" };
  }
  const actualSha256 = createHash("sha256").update(artifact.bytesUtf8, "utf8").digest("hex");
  if (actualSha256 !== artifact.expectedSha256.toLowerCase() ||
      Buffer.byteLength(artifact.bytesUtf8, "utf8") !== artifact.expectedBytes) {
    return { valid: false, reason: "SEALED_ARTIFACT_BYTES_OR_HASH_MISMATCH" };
  }
  let body: unknown;
  try {
    body = JSON.parse(artifact.bytesUtf8);
  } catch {
    return { valid: false, reason: "SEALED_ARTIFACT_JSON_INVALID" };
  }
  if (!isObject(body) || body.run_id !== runId || body.data_cutoff !== dataCutoff ||
      body.stage !== "DEEP_DIVE" || body.artifact_type !== expectedType ||
      body.version !== artifact.expectedVersion) {
    return { valid: false, reason: "SEALED_ARTIFACT_IDENTITY_OR_VERSION_MISMATCH" };
  }
  return { valid: true, body };
}

function exactString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

/**
 * Admission gate before any candidate enters composed V2.0.11 validation.
 *
 * This function NEVER constructs, trims, truncates or writes a snapshot.
 * It proves that three exact source strings are identical in the two
 * sealed upstream Deep Dive records AND in the canonical candidate.
 * It does not authorize GO PUBLISH, override a frozen pin, or execute an RPC.
 */
export function verifyCertifiedThesisAndCandidate(
  input: ExactThesisAdmissionInput,
): ExactThesisAdmissionResult {
  const sealed = readSealedJson(input.structuredThesis, "STRUCTURED_INVESTMENT_THESIS_ARTIFACT", input.runId, input.dataCutoff);
  if (!sealed.valid) return reject("STRUCTURED_THESIS_" + sealed.reason);
  const report = readSealedJson(input.deepDiveReport, "DEEP_DIVE_REPORT", input.runId, input.dataCutoff);
  if (!report.valid) return reject("DEEP_DIVE_REPORT_" + report.reason);

  const body = sealed.body;
  if (!isObject(report.body.thesis)) return reject("DEEP_DIVE_REPORT_THESIS_MISSING");
  const candidate = input.canonicalCandidate;
  if (!isObject(candidate) || !isObject(candidate.v2_product) ||
      !isObject(candidate.v2_product.investment_thesis)) {
    return reject("CANONICAL_CANDIDATE_THESIS_MISSING");
  }

  if (!isObject(candidate.data_lock) ||
      candidate.data_lock.data_cutoff !== input.dataCutoff ||
      candidate.issuer_id !== input.issuerId ||
      candidate.security_id !== input.securityId) {
    return reject("CANDIDATE_IDENTITY_OR_DATA_CUTOFF_MISMATCH");
  }

  const candidateThesis = candidate.v2_product.investment_thesis;
  const sourceFields = {
    quality_case: body.QUALITY_CASE,
    valuation_case: body.VALUATION_CASE,
    key_risk: body.KEY_RISK,
  };
  for (const [canonicalKey, sourceValue] of Object.entries(sourceFields)) {
    const sourceKey = canonicalKey.toUpperCase();
    if (!exactString(sourceValue)) return reject("CERTIFIED_THESIS_FIELD_MISSING:" + canonicalKey);
    if (report.body.thesis[sourceKey] !== sourceValue) {
      return reject("CERTIFIED_SOURCE_REPORT_CONFLICT:" + canonicalKey);
    }
    if (candidateThesis[canonicalKey] !== sourceValue) {
      return reject("CANDIDATE_REWRITES_CERTIFIED_THESIS:" + canonicalKey);
    }
  }

  // This is V2.0.11 composed validation only; additional Registry,
  // lineage, cutoff, I2, history, I3-B and publication gates remain.
  const validated = validateV2ResearchSnapshotForPersistenceWithLongThesis(candidate, input.dossierId);
  if (!validated.ok) return reject("V2_COMPOSED_VALIDATION_FAILED:" + validated.stage, ...validated.errors);
  return {
    admittedForValidation: true,
    sourceThesis: sourceFields as ExactThesis,
    errors: [],
  };
}
