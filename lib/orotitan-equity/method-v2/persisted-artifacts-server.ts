import "server-only";
import pack from "../../../contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_EXECUTION_CONTRACT_PIN_PACK_V1.0.json";
import { createServerSupabaseClient } from "../../supabase/server";
import { canonicalJson, METHOD_V2_AUTHORITY_SET_SHA256 } from "./authority";
import type { ArtifactRef, ChallengeContext } from "./pre-certification";
import { METHOD_V2_RUNTIME_BINDING_SHA256, type RegistryArtifact, type ArtifactExpectation } from "./persisted-artifacts";
import { verifyPersistedMethodV2Challenge, type ChallengeIdentity, type ChallengeRegistry, type ChallengeProof } from "./persisted-challenge";

type Client = ReturnType<typeof createServerSupabaseClient>;
const ref = (a: RegistryArtifact): ArtifactRef => ({ artifact_id: a.artifact_id, version: a.version, content_sha256: a.content_sha256 });
const exact = (a: unknown, b: unknown) => canonicalJson(a) === canonicalJson(b);
function fail(code: string): never { throw new Error(code); }

/** Read-only adapter over the existing server Supabase architecture. Never creates a run or records a caller PASS. */
export class SupabaseMethodV2ChallengeRegistry implements ChallengeRegistry {
  constructor(private readonly client: Client) {}
  async readArtifact(artifactId: string, version: number): Promise<RegistryArtifact | null> {
    const { data, error } = await this.client.from("orotitan_artifacts").select("*")
      .eq("artifact_id", artifactId).eq("version", version).maybeSingle();
    if (error) fail("METHOD_V2_REGISTRY_READ_FAILED");
    return data as RegistryArtifact | null;
  }
  async download(bucket: string, exactObjectPath: string): Promise<Uint8Array> {
    const { data, error } = await this.client.storage.from(bucket).download(exactObjectPath);
    if (error || !data) fail("METHOD_V2_STORAGE_DOWNLOAD_FAILED");
    return new Uint8Array(await data.arrayBuffer());
  }
  private expectation(a: RegistryArtifact, historical = false): ArtifactExpectation {
    if (a.authority_class !== "AUTHORITATIVE_STAGE_OUTPUT" && a.authority_class !== "CHECKPOINT_STAGE_OUTPUT") fail("METHOD_V2_ARTIFACT_AUTHORITY_INVALID");
    if (a.authority_state !== "AUTHORITATIVE" && a.authority_state !== "CHECKPOINT" && !(historical && a.authority_state === "SUPERSEDED")) fail("METHOD_V2_ARTIFACT_AUTHORITY_INVALID");
    return { ref: ref(a), runId: a.run_id, stageCode: a.stage_code, artifactType: a.artifact_type,
      authorityClass: a.authority_class, authorityState: a.authority_state, artifactStatus: "SEALED" };
  }
  private async parents(child: RegistryArtifact): Promise<RegistryArtifact[]> {
    const { data, error } = await this.client.from("orotitan_artifact_edges").select("parent_run_id,parent_artifact_id,parent_version")
      .eq("child_run_id", child.run_id).eq("child_artifact_id", child.artifact_id).eq("child_version", child.version).eq("relation_type", "CONSUMES");
    if (error) fail("METHOD_V2_REGISTRY_EDGE_READ_FAILED");
    return Promise.all((data ?? []).map(async edge => {
      const parent = await this.readArtifact(edge.parent_artifact_id, edge.parent_version);
      if (!parent || parent.run_id !== edge.parent_run_id) fail("METHOD_V2_REGISTRY_EDGE_MISMATCH");
      return parent;
    }));
  }
  private async identity(runId: string, ledger?: RegistryArtifact, report?: RegistryArtifact, priorProof?: ChallengeProof, candidateManifestRef?: ArtifactRef): Promise<ChallengeIdentity> {
    const [runResult, stageResult] = await Promise.all([
      this.client.from("orotitan_runs").select("*").eq("run_id", runId).single(),
      this.client.from("orotitan_run_stages").select("*").eq("run_id", runId).eq("stage_code", "DEEP_DIVE").single(),
    ]);
    const r = runResult.data; const stage = stageResult.data;
    if (runResult.error || stageResult.error || !r || !stage || r.methodology_generation !== "METHOD_V2"
      || r.methodology_authority_sha256 !== METHOD_V2_AUTHORITY_SET_SHA256 || r.contract_set_sha256 !== pack.contract_set_sha256
      || !exact(r.contract_pins, pack.contract_pins) || r.runtime_binding_sha256 !== METHOD_V2_RUNTIME_BINDING_SHA256
      || !r.issuer_id || !r.security_id || !r.dossier_id) fail("METHOD_V2_CHALLENGE_RUNTIME_IDENTITY_MISMATCH");
    let candidateManifest: ArtifactExpectation | undefined;
    if (candidateManifestRef) {
      const candidate = await this.readArtifact(candidateManifestRef.artifact_id, candidateManifestRef.version);
      if (!candidate || !exact(ref(candidate), candidateManifestRef) || candidate.run_id !== runId
        || candidate.stage_code !== "DEEP_DIVE" || candidate.artifact_type !== "DEEP_DIVE_STAGE_MANIFEST") fail("METHOD_V2_CHALLENGE_CANDIDATE_MISMATCH");
      candidateManifest = this.expectation(candidate);
    }
    if (!ledger || !report) {
      const { data, error } = await this.client.from("orotitan_artifacts").select("*").eq("run_id", runId)
        .eq("stage_code", "DEEP_DIVE").eq("manifest_artifact_id", candidateManifestRef?.artifact_id ?? stage.active_manifest_artifact_id)
        .eq("manifest_version", candidateManifestRef?.version ?? stage.active_manifest_version)
        .in("artifact_type", ["PRE_CERTIFICATION_QUESTION_LEDGER", "PRE_CERTIFICATION_CHALLENGE_REPORT"]);
      if (error) fail("METHOD_V2_REGISTRY_READ_FAILED");
      const rows = (data ?? []) as RegistryArtifact[];
      const q = rows.filter(a => a.artifact_type === "PRE_CERTIFICATION_QUESTION_LEDGER");
      const ch = rows.filter(a => a.artifact_type === "PRE_CERTIFICATION_CHALLENGE_REPORT");
      if (q.length !== 1 || ch.length !== 1) fail("METHOD_V2_CHALLENGE_CURRENT_LINEAGE_ABSENT");
      ledger = q[0]; report = ch[0];
    }
    if (ledger.run_id !== runId || report.run_id !== runId || ledger.stage_code !== "DEEP_DIVE" || report.stage_code !== "DEEP_DIVE"
      || ledger.artifact_type !== "PRE_CERTIFICATION_QUESTION_LEDGER" || report.artifact_type !== "PRE_CERTIFICATION_CHALLENGE_REPORT") fail("METHOD_V2_CHALLENGE_REGISTRY_LINEAGE_MISMATCH");
    const parents = await this.parents(report);
    const one = (kind: string) => {
      const candidates = parents.filter(a => a.artifact_type === kind && a.run_id === runId && a.stage_code === "DEEP_DIVE");
      if (candidates.length !== 1) fail("METHOD_V2_CHALLENGE_LOCK_LINEAGE_MISMATCH");
      return candidates[0];
    };
    const fundamentals = one("FUNDAMENTALS_LOCK"); const valuation = one("VALUATION_LOCK");
    if (!parents.some(a => exact(ref(a), ref(ledger)))) fail("METHOD_V2_CHALLENGE_LEDGER_EDGE_ABSENT");
    const { data: issuer, error: issuerError } = await this.client.from("issuers").select("display_name").eq("issuer_id", r.issuer_id).single();
    if (issuerError || !issuer) fail("METHOD_V2_ISSUER_READ_FAILED");
    const context: ChallengeContext = { analyticalAuthoritySetSha256: r.methodology_authority_sha256, run_id: runId,
      company: issuer.display_name, data_cutoff: r.data_cutoff, fundamentals_lock_ref: ref(fundamentals), valuation_lock_ref: ref(valuation),
      question_ledger_ref: ref(ledger), challenge_report_ref: ref(report) };
    const priorReports = parents.filter(a => a.artifact_type === "PRE_CERTIFICATION_CHALLENGE_REPORT");
    if (priorReports.length > 1) fail("METHOD_V2_CHALLENGE_PRIOR_LINEAGE_AMBIGUOUS");
    if (priorReports.length === 1) {
      const prior = priorReports[0];
      const { data: proof, error } = await this.client.from("orotitan_method_v2_challenge_proofs").select("*")
        .eq("run_id", prior.run_id).eq("challenge_report_id", prior.artifact_id).eq("challenge_report_version", prior.version).single();
      if (error || !proof || proof.challenge_report_sha256 !== prior.content_sha256) fail("METHOD_V2_CHALLENGE_PRIOR_PROOF_ABSENT");
      const priorLedger = await this.readArtifact(proof.question_ledger_id, proof.question_ledger_version);
      const { data: priorRun, error: priorRunError } = await this.client.from("orotitan_runs").select("issuer_id,security_id,dossier_id,data_cutoff")
        .eq("run_id", prior.run_id).single();
      if (!priorLedger || priorRunError || !priorRun || priorRun.issuer_id !== r.issuer_id || priorRun.security_id !== r.security_id
        || priorRun.dossier_id !== r.dossier_id || priorRun.data_cutoff > r.data_cutoff) fail("METHOD_V2_CHALLENGE_PRIOR_LINEAGE_MISMATCH");
      context.priorPassing = { report_ref: ref(prior), ledger_ref: ref(priorLedger) };
    }
    if (priorProof && (priorProof.question_ledger_sha256 !== ledger.content_sha256 || priorProof.challenge_report_sha256 !== report.content_sha256
      || priorProof.fundamentals_lock_id !== fundamentals.artifact_id || priorProof.fundamentals_lock_version !== fundamentals.version
      || priorProof.fundamentals_lock_sha256 !== fundamentals.content_sha256 || priorProof.valuation_lock_id !== valuation.artifact_id
      || priorProof.valuation_lock_version !== valuation.version || priorProof.valuation_lock_sha256 !== valuation.content_sha256)) fail("METHOD_V2_CHALLENGE_PRIOR_PROOF_MISMATCH");
    const expected = (a: RegistryArtifact) => ({ ...this.expectation(a, !!priorProof),
      ...(candidateManifestRef ? { manifestRef: { artifact_id: candidateManifestRef.artifact_id, version: candidateManifestRef.version } } : {}) });
    return { context, candidateManifest, ...(priorProof ? { historicalPassing: true as const } : {}), methodologyGeneration: "METHOD_V2", contractSetSha256: r.contract_set_sha256,
      runtimeBindingSha256: r.runtime_binding_sha256, stageRevision: priorProof?.stage_revision ?? stage.stage_revision,
      questionLedger: expected(ledger), challengeReport: expected(report),
      fundamentalsLock: expected(fundamentals), valuationLock: expected(valuation) };
  }
  readChallengeIdentity(runId: string, candidateManifestRef?: ArtifactRef) { return this.identity(runId, undefined, undefined, undefined, candidateManifestRef); }
  async readPriorPassingIdentity(reportRef: ArtifactRef, ledgerRef: ArtifactRef) {
    const report = await this.readArtifact(reportRef.artifact_id, reportRef.version);
    const ledger = await this.readArtifact(ledgerRef.artifact_id, ledgerRef.version);
    if (!report || !ledger || !exact(ref(report), reportRef) || !exact(ref(ledger), ledgerRef)) fail("METHOD_V2_CHALLENGE_PRIOR_LINEAGE_MISMATCH");
    const { data, error } = await this.client.from("orotitan_method_v2_challenge_proofs").select("*")
      .eq("run_id", report.run_id).eq("challenge_report_id", report.artifact_id).eq("challenge_report_version", report.version).single();
    if (error || !data || data.question_ledger_id !== ledger.artifact_id || data.question_ledger_version !== ledger.version) fail("METHOD_V2_CHALLENGE_PRIOR_PROOF_ABSENT");
    return this.identity(report.run_id, ledger, report, data as ChallengeProof);
  }
  async readEvidenceExpectation(evidenceRef: ArtifactRef, current: ChallengeIdentity) {
    const report = await this.readArtifact(current.context.challenge_report_ref.artifact_id, current.context.challenge_report_ref.version);
    if (!report) fail("METHOD_V2_CHALLENGE_ABSENT");
    const parents = await this.parents(report);
    const ledger = parents.find(a => exact(ref(a), evidenceRef));
    if (!ledger || ledger.run_id !== current.context.run_id || ledger.artifact_type !== "EVIDENCE_LEDGER") fail("METHOD_V2_EVIDENCE_REGISTRY_LINEAGE_MISMATCH");
    if (current.historicalPassing) return this.expectation(ledger, true);
    if (current.candidateManifest && ledger.manifest_artifact_id === current.candidateManifest.ref.artifact_id
      && ledger.manifest_version === current.candidateManifest.ref.version) return { ...this.expectation(ledger),
      manifestRef: { artifact_id: current.candidateManifest.ref.artifact_id, version: current.candidateManifest.ref.version } };
    // A current authoritative Evidence Ledger must belong to its stage's current FINAL manifest.
    const { data: stage, error } = await this.client.from("orotitan_run_stages").select("active_manifest_artifact_id,active_manifest_version,active_manifest_kind")
      .eq("run_id", ledger.run_id).eq("stage_code", ledger.stage_code).single();
    if (error || !stage || stage.active_manifest_kind !== "FINAL" || ledger.manifest_artifact_id !== stage.active_manifest_artifact_id
      || ledger.manifest_version !== stage.active_manifest_version) fail("METHOD_V2_EVIDENCE_REGISTRY_LINEAGE_MISMATCH");
    return { ...this.expectation(ledger), manifestRef: { artifact_id: stage.active_manifest_artifact_id, version: stage.active_manifest_version } };
  }
}

export function verifyStoredMethodV2Challenge(runId: string, candidateManifestRef?: ArtifactRef) {
  return verifyPersistedMethodV2Challenge(new SupabaseMethodV2ChallengeRegistry(createServerSupabaseClient()), runId, candidateManifestRef);
}
