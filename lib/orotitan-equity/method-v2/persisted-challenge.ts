import { METHOD_V2_AUTHORITY_SET_SHA256, canonicalJson } from "./authority";
import { parseMethodV2EvidenceLedger } from "./evidence-ledger-persistence";
import { parseMethodV2CertificationArtifact, verifyMethodV2CertificationChallengeBinding } from "./certification-persistence";
import { admitMethodV2Certification, CHALLENGE_FAMILIES, type ArtifactRef, type ChallengeContext, type FamilyCoverage } from "./pre-certification";
import { METHOD_V2_RUNTIME_BINDING_SHA256, METHOD_V2_CERTIFICATION_PROFILE_SHA256, resolvePersistedMethodV2Artifact,
  type ArtifactExpectation, type PersistedArtifactSource } from "./persisted-artifacts";

export type ChallengeIdentity = {
  context: ChallengeContext; methodologyGeneration: "METHOD_V2"; contractSetSha256: string;
  stageRevision: number; runtimeBindingSha256: string;
  candidateManifest?: ArtifactExpectation; historicalPassing?: true;
  questionLedger: ArtifactExpectation; challengeReport: ArtifactExpectation;
  fundamentalsLock: ArtifactExpectation; valuationLock: ArtifactExpectation;
  certification: ArtifactExpectation;
};
export interface ChallengeRegistry extends PersistedArtifactSource {
  // This identity is loaded from Registry run/stage/artifact lineage, not supplied as PASS claims.
  readChallengeIdentity(runId: string, candidateManifestRef?: ArtifactRef): Promise<ChallengeIdentity>;
  readPriorPassingIdentity(reportRef: ArtifactRef, ledgerRef: ArtifactRef): Promise<ChallengeIdentity>;
  readEvidenceExpectation(ref: ArtifactRef, current: ChallengeIdentity): Promise<ArtifactExpectation>;
}
export type ChallengeProof = {
  run_id: string; stage_revision: number;
  question_ledger_id: string; question_ledger_version: number; question_ledger_sha256: string;
  challenge_report_id: string; challenge_report_version: number; challenge_report_sha256: string;
  fundamentals_lock_id: string; fundamentals_lock_version: number; fundamentals_lock_sha256: string;
  valuation_lock_id: string; valuation_lock_version: number; valuation_lock_sha256: string;
  certification_id: string; certification_version: number; certification_sha256: string;
  certification_profile_sha256: string;
  runtime_binding_sha256: string; validator_identity: "verifyPersistedMethodV2Challenge:1.1";
};
const same = (a: unknown, b: unknown) => canonicalJson(a) === canonicalJson(b);

/** No self-attestation inputs or storage paths. Every admission rereads all required bytes. */
export async function verifyPersistedMethodV2Challenge(registry: ChallengeRegistry, runId: string, candidateManifestRef?: ArtifactRef) {
  const visited = new Set<string>();
  const verify = async (identity: ChallengeIdentity, historical = false): Promise<ReturnType<typeof admitMethodV2Certification>> => {
    const context = identity.context;
    if (identity.methodologyGeneration !== "METHOD_V2" || context.analyticalAuthoritySetSha256 !== METHOD_V2_AUTHORITY_SET_SHA256
      || identity.contractSetSha256 !== "23b75bf5c2d7448e8270e7e8f3a0223e639d0be3a1406b3c089e6e885dd063ea"
      || identity.runtimeBindingSha256 !== METHOD_V2_RUNTIME_BINDING_SHA256
      || !Number.isSafeInteger(identity.stageRevision) || identity.stageRevision < 1) throw new Error("METHOD_V2_CHALLENGE_RUNTIME_IDENTITY_MISMATCH");
    if (identity.historicalPassing && !historical) throw new Error("METHOD_V2_CHALLENGE_CURRENT_LINEAGE_REQUIRED");
    const key = `${context.challenge_report_ref.artifact_id}:${context.challenge_report_ref.version}`;
    if (visited.has(key)) throw new Error("METHOD_V2_CHALLENGE_DELTA_CYCLE");
    visited.add(key);
    if (!identity.certification) throw new Error("METHOD_V2_CERTIFICATION_CURRENT_LINEAGE_REQUIRED");
    const expectedArtifacts = [
      [identity.questionLedger, context.question_ledger_ref, "PRE_CERTIFICATION_QUESTION_LEDGER"],
      [identity.challengeReport, context.challenge_report_ref, "PRE_CERTIFICATION_CHALLENGE_REPORT"],
      [identity.fundamentalsLock, context.fundamentals_lock_ref, "FUNDAMENTALS_LOCK"],
      [identity.valuationLock, context.valuation_lock_ref, "VALUATION_LOCK"],
      [identity.certification, identity.certification.ref, "CERTIFICATION_ARTIFACT"],
    ] as const;
    for (const [expected, ref, kind] of expectedArtifacts) {
      if (!same(expected.ref, ref) || expected.runId !== context.run_id || expected.stageCode !== "DEEP_DIVE"
        || expected.artifactType !== kind || expected.artifactStatus !== "SEALED"
        || !(kind === "CERTIFICATION_ARTIFACT" ? ["AUTHORITATIVE_STAGE_OUTPUT"] : ["AUTHORITATIVE_STAGE_OUTPUT", "CHECKPOINT_STAGE_OUTPUT"]).includes(expected.authorityClass)
        || (kind === "CERTIFICATION_ARTIFACT" && !(historical ? ["AUTHORITATIVE", "SUPERSEDED"] : ["AUTHORITATIVE"]).includes(expected.authorityState))
        || !(historical ? ["AUTHORITATIVE", "CHECKPOINT", "SUPERSEDED"] : ["AUTHORITATIVE", "CHECKPOINT"]).includes(expected.authorityState)) {
        throw new Error("METHOD_V2_CHALLENGE_REGISTRY_LINEAGE_MISMATCH");
      }
    }
    if (identity.candidateManifest) {
      const manifestExpected = identity.candidateManifest;
      if (manifestExpected.runId !== context.run_id || manifestExpected.stageCode !== "DEEP_DIVE"
        || manifestExpected.artifactType !== "DEEP_DIVE_STAGE_MANIFEST" || manifestExpected.authorityClass !== "AUTHORITATIVE_STAGE_OUTPUT"
        || manifestExpected.authorityState !== "AUTHORITATIVE") throw new Error("METHOD_V2_CHALLENGE_CANDIDATE_MISMATCH");
      const manifest = await resolvePersistedMethodV2Artifact(registry, manifestExpected);
      const raw = JSON.parse(Buffer.from(manifest.bytes).toString("utf8")) as {
        run_id: string; stage: string; manifest_id: string; manifest_kind: string; stage_revision: number;
        contract_set_sha256: string; output_artifacts: (ArtifactRef & { artifact_type: string; authority_class: string })[];
      };
      if (raw.run_id !== context.run_id || raw.stage !== "DEEP_DIVE" || raw.manifest_id !== manifestExpected.ref.artifact_id
        || raw.manifest_kind !== "FINAL" || raw.stage_revision !== identity.stageRevision || raw.contract_set_sha256 !== identity.contractSetSha256
        || !Array.isArray(raw.output_artifacts) || expectedArtifacts.some(([expected, ref, kind]) =>
          raw.output_artifacts.filter(a => a.artifact_type === kind && a.authority_class === "AUTHORITATIVE_STAGE_OUTPUT"
            && same({ artifact_id: a.artifact_id, version: a.version, content_sha256: a.content_sha256 }, ref)).length !== 1
          || expected.manifestRef?.artifact_id !== manifestExpected.ref.artifact_id
          || expected.manifestRef.version !== manifestExpected.ref.version)) throw new Error("METHOD_V2_CHALLENGE_CANDIDATE_MISMATCH");
    }
    const [ledger, report, , , certification] = await Promise.all(expectedArtifacts.map(([expected]) => resolvePersistedMethodV2Artifact(registry, expected)));
    // Pure validation owns schema, aggregation, saturation, concerns and cutoff checks.
    const result = admitMethodV2Certification(context, ledger.bytes, report.bytes);
    const rawReport = JSON.parse(Buffer.from(report.bytes).toString("utf8")) as {
      mode: "FULL" | "DELTA"; challenge_status: "PASS" | "PASS_WITH_CONCERNS";
      saturation_record: { family_coverage: Record<typeof CHALLENGE_FAMILIES[number], FamilyCoverage> };
    };
    if (rawReport.mode === "DELTA") {
      if (!context.priorPassing) throw new Error("CHALLENGE_DELTA_PROVENANCE_INVALID");
      const prior = await registry.readPriorPassingIdentity(context.priorPassing.report_ref, context.priorPassing.ledger_ref);
      if (!same(prior.context.challenge_report_ref, context.priorPassing.report_ref)
        || !same(prior.context.question_ledger_ref, context.priorPassing.ledger_ref)
        || prior.context.company !== context.company || prior.context.data_cutoff > context.data_cutoff) {
        throw new Error("METHOD_V2_CHALLENGE_PRIOR_LINEAGE_MISMATCH");
      }
      await verify(prior, true);
    }
    let authoritativeEvidenceLedgerRef: ArtifactRef | undefined;
    for (const family of CHALLENGE_FAMILIES) {
      const coverage = rawReport.saturation_record.family_coverage[family];
      if (coverage.disposition !== "EVIDENCED_NO_MATERIAL_CHALLENGE") continue;
      for (const reference of coverage.evidence_references) {
        const ref = reference.evidence_ledger_ref;
        if (authoritativeEvidenceLedgerRef && (ref.artifact_id !== authoritativeEvidenceLedgerRef.artifact_id
          || ref.version !== authoritativeEvidenceLedgerRef.version
          || ref.content_sha256 !== authoritativeEvidenceLedgerRef.content_sha256)) {
          throw new Error("METHOD_V2_EVIDENCE_LEDGER_AUTHORITY_MISMATCH");
        }
        const expected = await registry.readEvidenceExpectation(reference.evidence_ledger_ref, identity);
        if (!same(expected.ref, reference.evidence_ledger_ref) || expected.artifactType !== "EVIDENCE_LEDGER"
          || expected.artifactStatus !== "SEALED" || expected.authorityClass !== "AUTHORITATIVE_STAGE_OUTPUT"
          || !(historical ? ["AUTHORITATIVE", "SUPERSEDED"] : ["AUTHORITATIVE"]).includes(expected.authorityState)) throw new Error("METHOD_V2_EVIDENCE_REGISTRY_LINEAGE_MISMATCH");
        const resolved = await resolvePersistedMethodV2Artifact(registry, expected);
        const parsed = parseMethodV2EvidenceLedger(resolved.bytes);
        if (!parsed.entries.some(entry => entry.EVIDENCE_ID === reference.evidence_id)) {
          throw new Error("METHOD_V2_EVIDENCE_LEDGER_EVIDENCE_ID_NOT_FOUND");
        }
        authoritativeEvidenceLedgerRef ??= { artifact_id: ref.artifact_id, version: ref.version, content_sha256: ref.content_sha256 };
      }
    }
    // Validate authoritative persisted Certification, including opaque inherited
    // bytes, only against the independently admitted current Challenge context.
    // CONSUMES edges and caller claims cannot stand in for this exact binding.
    verifyMethodV2CertificationChallengeBinding(parseMethodV2CertificationArtifact(certification.bytes), {
      run_id: context.run_id, stage_revision: identity.stageRevision, data_cutoff: context.data_cutoff,
      question_ledger_ref: context.question_ledger_ref, challenge_report_ref: context.challenge_report_ref,
      fundamentals_lock_ref: context.fundamentals_lock_ref, valuation_lock_ref: context.valuation_lock_ref,
      challenge_status: rawReport.challenge_status, challenge_limitations: result.limitations,
    });
    return result;
  };
  const identity = await registry.readChallengeIdentity(runId, candidateManifestRef);
  if (candidateManifestRef && (!identity.candidateManifest || !same(identity.candidateManifest.ref, candidateManifestRef))) throw new Error("METHOD_V2_CHALLENGE_CANDIDATE_MISMATCH");
  if (identity.context.run_id !== runId) throw new Error("METHOD_V2_CHALLENGE_RUN_MISMATCH");
  const admission = await verify(identity);
  const ctx = identity.context;
  const proof: ChallengeProof = {
    run_id: runId, stage_revision: identity.stageRevision,
    question_ledger_id: ctx.question_ledger_ref.artifact_id, question_ledger_version: ctx.question_ledger_ref.version,
    question_ledger_sha256: ctx.question_ledger_ref.content_sha256,
    challenge_report_id: ctx.challenge_report_ref.artifact_id, challenge_report_version: ctx.challenge_report_ref.version,
    challenge_report_sha256: ctx.challenge_report_ref.content_sha256,
    fundamentals_lock_id: ctx.fundamentals_lock_ref.artifact_id, fundamentals_lock_version: ctx.fundamentals_lock_ref.version,
    fundamentals_lock_sha256: ctx.fundamentals_lock_ref.content_sha256,
    valuation_lock_id: ctx.valuation_lock_ref.artifact_id, valuation_lock_version: ctx.valuation_lock_ref.version,
    valuation_lock_sha256: ctx.valuation_lock_ref.content_sha256,
    certification_id: identity.certification.ref.artifact_id, certification_version: identity.certification.ref.version,
    certification_sha256: identity.certification.ref.content_sha256,
    certification_profile_sha256: METHOD_V2_CERTIFICATION_PROFILE_SHA256,
    runtime_binding_sha256: METHOD_V2_RUNTIME_BINDING_SHA256, validator_identity: "verifyPersistedMethodV2Challenge:1.1",
  };
  return { admission, proof };
}

export interface TrustedChallengeProofWriter {
  // Owner execution boundary. This mission grants no API/runtime role proof writes.
  recordVerifiedProof(proof: ChallengeProof): Promise<void>;
}

/** The owner persistence sink receives only output from a successful persisted-byte verification. */
export async function verifyAndRecordMethodV2Challenge(registry: ChallengeRegistry, runId: string, writer: TrustedChallengeProofWriter, candidateManifestRef?: ArtifactRef) {
  const result = await verifyPersistedMethodV2Challenge(registry, runId, candidateManifestRef);
  await writer.recordVerifiedProof(result.proof);
  return result.admission;
}
