import { createHash } from "node:crypto";
import {
  EXACT_SOURCE_SEMANTIC_COMPATIBILITY as Literal,
} from "./exact-source-semantic-compatibility";
import {
  validateV2ResearchSnapshotWithExactCertifiedLiterals,
} from "./research-snapshot-schema-exact-literals";

type RecordObject = Record<string, unknown>;
export type RegistrySealedSource = Readonly<{
  utf8Body: string;
  registeredSha256: string;
  registeredSizeBytes: number;
  registeredArtifactId: string;
  registeredVersion: number;
}>;
export type CertifiedLiteralGateInput = Readonly<{
  runId: string;
  issuerId: string;
  securityId: string;
  dossierId: string;
  dataCutoff: string;
  frozenFundamentalsLock: RegistrySealedSource;
  frozenReadiness: RegistrySealedSource;
  candidate: unknown;
}>;
export type CertifiedLiteralGateResult = Readonly<
  | { ok: true; stage: "EXACT_SOURCE_LITERAL_ADMITTED"; semanticValidator: "PASS"; }
  | { ok: false; stage: "SOURCE_PROVENANCE" | "SEMANTIC_SCHEMA"; errors: readonly string[]; }
>;

function isObject(x: unknown): x is RecordObject {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

function fail(stage: "SOURCE_PROVENANCE" | "SEMANTIC_SCHEMA", code: string): CertifiedLiteralGateResult {
  return { ok: false, stage, errors: [code] };
}

function authenticatedSource(
  input: RegistrySealedSource,
  expected: { runId: string; dataCutoff: string; type: string },
): RecordObject | null {
  if (!/^[0-9a-f]{64}$/.test(input.registeredSha256) ||
      !/^[0-9a-f]{8}-[0-9a-f-]{27,}$/.test(input.registeredArtifactId) ||
      !Number.isSafeInteger(input.registeredSizeBytes) ||
      input.registeredSizeBytes < 1 ||
      !Number.isSafeInteger(input.registeredVersion) ||
      input.registeredVersion < 1) return null;
  const buf = Buffer.from(input.utf8Body, "utf8");
  if (buf.length !== input.registeredSizeBytes ||
      createHash("sha256").update(buf).digest("hex") !== input.registeredSha256) return null;

  let parsed: unknown;
  try { parsed = JSON.parse(input.utf8Body); }
  catch { return null; }
  if (!isObject(parsed) || parsed.run_id !== expected.runId ||
      parsed.data_cutoff !== expected.dataCutoff ||
      parsed.artifact_type !== expected.type ||
      parsed.stage !== "DEEP_DIVE" ||
      parsed.version !== input.registeredVersion) return null;
  if (isObject(parsed.artifact) && parsed.artifact.artifact_id !== input.registeredArtifactId) return null;
  return parsed;
}

/**
 * On a V2.0.12 review path only:
 * - registered artifact hashes/identities MUST be supplied by the Registry
 *   and independently compared to its FINAL manifest before invoking this;
 * - exact frozen source literals MUST match candidate byte-for-byte;
 * - both frozen literal domains are additionally pinned to two safe values;
 * - then the full composed V1/V2 semantic/score validation is invoked.
 *
 * This function is PURE and creates no publication authorization.
 */
export function validateFicoStyleExactLiteralsFromSealedSources(
  input: CertifiedLiteralGateInput,
): CertifiedLiteralGateResult {
  const expected = { runId: input.runId, dataCutoff: input.dataCutoff };
  const fundamental = authenticatedSource(input.frozenFundamentalsLock,
    { ...expected, type: "FUNDAMENTALS_LOCK" });
  const readiness = authenticatedSource(input.frozenReadiness,
    { ...expected, type: "READINESS_NEXT_ACTION_ARTIFACT" });
  if (!fundamental || !readiness) return fail("SOURCE_PROVENANCE", "SEALED_SOURCE_HASH_OR_IDENTITY_MISMATCH");
  if (fundamental.issuer_id !== input.issuerId ||
      fundamental.security_id !== input.securityId ||
      fundamental.dossier_id !== input.dossierId) {
    return fail("SOURCE_PROVENANCE", "FUNDAMENTALS_IDENTITY_MISMATCH");
  }
  const candidate = input.candidate;
  if (!isObject(candidate) ||
      candidate.issuer_id !== input.issuerId ||
      candidate.security_id !== input.securityId ||
      !isObject(candidate.data_lock) ||
      candidate.data_lock.data_cutoff !== input.dataCutoff ||
      !isObject(candidate.l2_research_fundamentals) ||
      !isObject(candidate.l2_research_fundamentals.analytical_metrics) ||
      !isObject(candidate.l4_operational_state) ||
      !isObject(fundamental.canonical_fundamental_verdicts)) {
    return fail("SOURCE_PROVENANCE", "CANDIDATE_LINEAGE_OR_SHAPE_MISMATCH");
  }

  const returnSource = fundamental.canonical_fundamental_verdicts.RETURN_INTERPRETABILITY;
  const refreshSource = readiness.refresh_class;
  if (returnSource !== Literal.sourceReturnInterpretability ||
      refreshSource !== Literal.sourceRefreshClass) {
    return fail("SOURCE_PROVENANCE", "SOURCE_OUTSIDE_ADMITTED_EXACT_LITERAL_SCOPE");
  }
  if (candidate.l2_research_fundamentals.analytical_metrics.return_interpretability !== returnSource ||
      candidate.l4_operational_state.refresh_class !== refreshSource) {
    return fail("SOURCE_PROVENANCE", "CANONICAL_CANDIDATE_COERCED_OR_REWROTE_SOURCE");
  }

  const validated = validateV2ResearchSnapshotWithExactCertifiedLiterals(candidate, input.dossierId);
  if (!validated.ok) return { ok: false, stage: "SEMANTIC_SCHEMA", errors: validated.errors };
  return { ok: true, stage: "EXACT_SOURCE_LITERAL_ADMITTED", semanticValidator: "PASS" };
}
