/**
 * OroTitan D6 PROPOSED full-precision provenance validator.
 *
 * This format is not frozen. Passing it cannot grant I2_PASS, I3B_PASS or
 * publication authority. D2 must first verify every immutable artifact's
 * registered SHA-256, Git blob SHA, size, authority and active manifest.
 */
import { isDeepStrictEqual } from "node:util";
import { readFileSync } from "node:fs";
import Ajv2020 from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { validateCanonicalContract, computeCanonicalSnapshot,
  type CanonicalContractInput } from "../v1/contract";
import type { VerifiedContent } from "./pilotage-typed-gates-v3-1";

const schema = JSON.parse(readFileSync(new URL(
  "../../../docs/orotitan-equity/proposals/OROTITAN_I2_FULL_PRECISION_PROVENANCE_SCHEMA_D6_V0_1_PROPOSED.json",
  import.meta.url,
), "utf8")) as object;
const ajv = new Ajv2020({ strict: false, allErrors: true });
addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);
const validateSchema = ajv.compile(schema);
const JSON_MIME = "application/json";
const roleByRoot: Record<string, {
  authority_role: string;
  artifact_type: string;
  stage_code: "DEEP_DIVE";
}> = {
  dimensions: { authority_role: "FUNDAMENTAL_JUDGMENT", artifact_type: "ANALYTICAL_BLOCK_OUTPUTS", stage_code: "DEEP_DIVE" },
  evidence: { authority_role: "FUNDAMENTAL_JUDGMENT", artifact_type: "ANALYTICAL_BLOCK_OUTPUTS", stage_code: "DEEP_DIVE" },
  businessResearchStatus: { authority_role: "CERTIFICATION_JUDGMENT", artifact_type: "CERTIFICATION_ARTIFACT", stage_code: "DEEP_DIVE" },
  investmentConclusionStatus: { authority_role: "CERTIFICATION_JUDGMENT", artifact_type: "CERTIFICATION_ARTIFACT", stage_code: "DEEP_DIVE" },
  scorePermission: { authority_role: "CERTIFICATION_JUDGMENT", artifact_type: "CERTIFICATION_ARTIFACT", stage_code: "DEEP_DIVE" },
  mosStatus: { authority_role: "VALUATION_JUDGMENT", artifact_type: "VALUATION_ARTIFACT", stage_code: "DEEP_DIVE" },
  valuationReliability: { authority_role: "VALUATION_JUDGMENT", artifact_type: "VALUATION_ARTIFACT", stage_code: "DEEP_DIVE" },
  primaryExpectedReturnDeltaPercentagePoints: { authority_role: "VALUATION_JUDGMENT", artifact_type: "VALUATION_ARTIFACT", stage_code: "DEEP_DIVE" },
  normalizedExpectedReturnDeltaPercentagePoints: { authority_role: "VALUATION_JUDGMENT", artifact_type: "VALUATION_ARTIFACT", stage_code: "DEEP_DIVE" },
  eliteGates: { authority_role: "TERMINAL_GATE_JUDGMENT", artifact_type: "OROTITAN_TERMINAL_GATE_ARTIFACT", stage_code: "DEEP_DIVE" },
};
type JsonObject = Record<string, unknown>;
type Ref = {
  artifact_id: string; version: number; content_sha256: string;
  artifact_type: string; stage_code: "RESEARCH" | "DEEP_DIVE";
};
type Binding = {
  input_pointer: string; source_ref: Ref;
  source_pointer: string; authority_role: string;
};
export class D6ProvenanceError extends Error {
  constructor(readonly code: string) { super(code); }
}
function requireCheck(v: unknown, code: string): asserts v {
  if (!v) throw new D6ProvenanceError(code);
}
function obj(v: unknown): v is JsonObject {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
function decodeJson(c: VerifiedContent, code: string): JsonObject {
  requireCheck(c.proof.bytesVerified && c.proof.available && c.mediaType === JSON_MIME,
    code + "_NOT_VERIFIED_JSON");
  let j: unknown;
  try { j = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(c.bytes)); }
  catch { throw new D6ProvenanceError(code + "_INVALID_UTF8_JSON"); }
  requireCheck(obj(j), code + "_NOT_OBJECT");
  return j;
}
function exactProof(ref: Ref, c: VerifiedContent, runId: string): boolean {
  return c.proof.bytesVerified && c.proof.available &&
    c.proof.runId === runId && c.proof.artifactId === ref.artifact_id &&
    c.proof.version === ref.version && c.proof.sha256 === ref.content_sha256 &&
    c.proof.stageCode === ref.stage_code && c.proof.artifactType === ref.artifact_type;
}
function encoded(segment: string): string {
  return segment.replace(/~/g, "~0").replace(/\//g, "~1");
}
function pointerParts(pointer: string): string[] {
  requireCheck(pointer.startsWith("/") && pointer.length <= 500, "D6_POINTER_INVALID");
  const parts = pointer.slice(1).split("/").map(s => {
    requireCheck(!/~(?![01])/.test(s), "D6_POINTER_BAD_ESCAPE");
    const value = s.replace(/~1/g, "/").replace(/~0/g, "~");
    requireCheck(value !== "" && value !== "__proto__" && value !== "prototype" &&
      value !== "constructor", "D6_POINTER_UNSAFE");
    return value;
  });
  requireCheck(parts.length <= 12, "D6_POINTER_TOO_DEEP");
  return parts;
}
function readPointer(v: unknown, pointer: string): unknown {
  let value = v;
  for (const p of pointerParts(pointer)) {
    requireCheck(obj(value) && Object.prototype.hasOwnProperty.call(value, p),
      "D6_SOURCE_POINTER_NOT_FOUND");
    value = value[p];
  }
  return value;
}
function leafPaths(value: unknown, prefix = ""): Map<string, unknown> {
  const result = new Map<string, unknown>();
  const walk = (v: unknown, path: string, depth: number): void => {
    requireCheck(depth <= 12, "D6_INPUT_TREE_TOO_DEEP");
    if (!obj(v)) {
      requireCheck(v !== undefined && v !== null, "D6_INPUT_LEAF_UNDEFINED");
      result.set(path, v);
      return;
    }
    const entries = Object.entries(v);
    requireCheck(entries.length > 0, "D6_INPUT_EMPTY_OBJECT");
    for (const [k, x] of entries) {
      requireCheck(k !== "__proto__" && k !== "constructor" && k !== "prototype",
        "D6_INPUT_KEY_UNSAFE");
      if (path === "" && k === "deterministic") continue;
      walk(x, path + "/" + encoded(k), depth + 1);
    }
  };
  walk(value, prefix, 0);
  requireCheck(result.size > 0 && result.size <= 120, "D6_INPUT_LEAF_COUNT_INVALID");
  return result;
}
export interface D6ProvenanceResult {
  proposedSchemaValidated: true;
  exactSourceBindingsVerified: true;
  validatedLeafCount: number;
  independentRecomputationMatches: true;
  methodologyFreezeApproved: false;
  analyticalCertificationProven: false;
  i2Pass: false;
  i3bPass: false;
  publicationAuthorized: false;
}
/** Pure, fail-closed D6 validator against D2-verified immutable source bytes. */
export function verifyD6ProposedProvenance(input: {
  fullPrecision: VerifiedContent;
  sourceArtifacts: readonly VerifiedContent[];
  expectedRunId: string;
  expectedDataCutoff: string;
  expectedStageRevision: number;
  expectedContractSetSha256: string;
  candidateCanonicalInput: CanonicalContractInput;
}): D6ProvenanceResult {
  const { fullPrecision: artifact } = input;
  requireCheck(artifact.proof.artifactType === "SCORING_INPUTS_FULL_PRECISION" &&
    artifact.proof.stageCode === "DEEP_DIVE" && artifact.proof.runId === input.expectedRunId,
    "D6_FULL_PRECISION_IDENTITY_INVALID");
  const document = decodeJson(artifact, "D6_FULL_PRECISION");
  requireCheck(validateSchema(document), "D6_PROPOSED_SCHEMA_INVALID");
  requireCheck(document.run_id === input.expectedRunId &&
    document.data_cutoff === input.expectedDataCutoff &&
    document.stage_revision === input.expectedStageRevision &&
    document.contract_set_sha256 === input.expectedContractSetSha256,
    "D6_PROVENANCE_SCOPE_OR_CONTRACT_MISMATCH");
  const own = document.artifact_identity as JsonObject;
  requireCheck(own.artifact_id === artifact.proof.artifactId &&
    own.version === artifact.proof.version, "D6_PROVENANCE_SELF_IDENTITY_MISMATCH");
  const parsed = validateCanonicalContract(document.canonical_contract_input);
  requireCheck(parsed.success, "D6_PROVENANCE_INPUT_INVALID");
  requireCheck(isDeepStrictEqual(parsed.data, input.candidateCanonicalInput),
    "D6_CANDIDATE_UPSTREAM_INPUT_MISMATCH");
  const recomputed = computeCanonicalSnapshot(parsed.data);
  requireCheck(isDeepStrictEqual(recomputed, computeCanonicalSnapshot(input.candidateCanonicalInput)),
    "D6_CANDIDATE_RECOMPUTE_MISMATCH");

  const evidence = document.research_evidence_ref as Ref;
  requireCheck(evidence.stage_code === "RESEARCH" && evidence.artifact_type === "EVIDENCE_LEDGER" &&
    input.sourceArtifacts.filter(c => exactProof(evidence, c, input.expectedRunId)).length === 1,
    "D6_RESEARCH_EVIDENCE_NOT_BYTE_VERIFIED");
  const leaves = leafPaths(parsed.data);
  const rows = document.bindings as Binding[];
  requireCheck(rows.length === leaves.size, "D6_LINEAGE_INCOMPLETE_OR_EXTRA");
  const seen = new Set<string>();
  const sourceCache = new Map<string, JsonObject>();
  let anchoredFundamentals = false;
  for (const row of rows) {
    const canonical = row.input_pointer;
    requireCheck(!seen.has(canonical) && leaves.has(canonical),
      "D6_LINEAGE_DUPLICATE_OR_UNKNOWN_INPUT");
    seen.add(canonical);
    const root = pointerParts(canonical)[0];
    const role = roleByRoot[root];
    requireCheck(role !== undefined && row.authority_role === role.authority_role &&
      row.source_ref.stage_code === role.stage_code &&
      row.source_ref.artifact_type === role.artifact_type,
      "D6_LINEAGE_WRONG_AUTHORITY_ROLE");
    const matching = input.sourceArtifacts.filter(x =>
      exactProof(row.source_ref, x, input.expectedRunId));
    requireCheck(matching.length === 1, "D6_LINEAGE_SOURCE_NOT_BYTE_VERIFIED");
    const source = matching[0];
    const k = source.proof.artifactId + ":" + source.proof.version;
    let body = sourceCache.get(k);
    if (!body) {
      body = decodeJson(source, "D6_LINEAGE_SOURCE");
      requireCheck(body.run_id === input.expectedRunId &&
        body.data_cutoff === input.expectedDataCutoff &&
        body.stage_revision === input.expectedStageRevision,
        "D6_LINEAGE_SOURCE_SCOPE_MISMATCH");
      sourceCache.set(k, body);
    }
    requireCheck(isDeepStrictEqual(readPointer(body, row.source_pointer), leaves.get(canonical)),
      "D6_LINEAGE_SOURCE_VALUE_MISMATCH");
    if (role.authority_role === "FUNDAMENTAL_JUDGMENT") {
      const backlink = body.research_evidence_ref;
      requireCheck(obj(backlink) && isDeepStrictEqual(backlink, evidence),
        "D6_FUNDAMENTALS_RESEARCH_EVIDENCE_BACKLINK_MISSING");
      anchoredFundamentals = true;
    }
  }
  requireCheck(seen.size === leaves.size && anchoredFundamentals,
    "D6_LINEAGE_NOT_COMPLETE");
  return {
    proposedSchemaValidated: true,
    exactSourceBindingsVerified: true,
    validatedLeafCount: leaves.size,
    independentRecomputationMatches: true,
    methodologyFreezeApproved: false,
    analyticalCertificationProven: false,
    i2Pass: false, i3bPass: false, publicationAuthorized: false,
  };
}
