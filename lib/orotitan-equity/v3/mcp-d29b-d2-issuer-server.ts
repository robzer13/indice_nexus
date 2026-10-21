import "server-only";
/** D29b privileged issuer — no API route, no trigger and no DB writer.
 * This can only be invoked by an independently authenticated admin-side
 * process with an isolated HSM/KMS signing adapter and explicit run allowlist.
 * Never import this module from the public MCP route.
 */
import { createHash } from "node:crypto";
import { requireAdminSession } from "../../auth/admin-session";
import { createServerSupabaseClient } from "../../supabase/server";
import { createServerPilotageReadPort, assertProductionUrl } from "./pilotage-trusted-loader-server";
import { loadTrustedPilotageDecision } from "./pilotage-trusted-loader-v3-1";
import { observeSupabaseEnvironment } from "./pilotage-operator-cockpit-server";
import { asOperatorBlockers } from "./pilotage-operator-view-v3-1";
import pins from "../../../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";
import {
  HANDOFF_PURPOSE,HANDOFF_SIGNATURE_DOMAIN,
  type HandoffAttestationV1, type SignedHandoffEnvelope,
} from "./mcp-d29b-signed-handoff-gate";

const UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export type D29bSigningAdapter={
  keyId:string;
  /** This MUST be a remotely managed Ed25519 signing operation, not key bytes. */
  sign:(message:Uint8Array)=>Promise<Uint8Array>;
};
type IssuerInput={runId:string;subject:string;clientId:string;resource:string};
type StageCode=HandoffAttestationV1["stage"];

export class D29bIssuerError extends Error {
  constructor(readonly code:string){super(code);}
}
function deny(code:string):never {throw new D29bIssuerError(code);}

/** An issuance is not a Registry write, stage start, or publishing GO.
 * A separately authorized process must store the result with an auditable
 * revocation record; this method never exposes the private key or saves data.
 */
export async function issueIndependentlyVerifiedD29bReceipt(
  input:IssuerInput, signer:D29bSigningAdapter, env:NodeJS.ProcessEnv=process.env,
):Promise<{receipt:SignedHandoffEnvelope;runId:string;subject:string;
  clientId:string;resource:string;issuedAt:number;expiresAt:number}> {
  await requireAdminSession();
  assertProductionUrl();
  const permittedRuns=env.OROTITAN_D29B_ISSUER_ALLOWED_RUN_IDS?.split(",")??[];
  const permittedSubjects=env.OROTITAN_D29B_ISSUER_ALLOWED_SUBJECTS?.split(",")??[];
  const permittedClients=env.OROTITAN_D29B_ISSUER_ALLOWED_CLIENT_IDS?.split(",")??[];
  const approvedResource=env.OROTITAN_D29B_ISSUER_RESOURCE;
  if(!UUID.test(input.runId) || !UUID.test(input.subject) ||
    !/^[A-Za-z0-9_-]{6,160}$/.test(input.clientId) ||
    permittedRuns.length!==1 || permittedRuns[0]!==input.runId ||
    permittedSubjects.length!==1 || permittedSubjects[0]!==input.subject ||
    permittedClients.length!==1 || permittedClients[0]!==input.clientId ||
    approvedResource!=="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp" ||
    input.resource!==approvedResource ||
    !/^[A-Za-z0-9_-]{1,64}$/.test(signer.keyId) ||
    typeof signer.sign!=="function")deny("D29B_ISSUER_SCOPE_NOT_CONFIGURED");

  const port=createServerPilotageReadPort(createServerSupabaseClient());
  const run=await port.getRun(input.runId);
  if(!run || run.run_id!==input.runId || run.run_status!=="ACTIVE" ||
    run.process_version!=="3.1" || run.pilotage_contract_version!=="3.1" ||
    !run.current_stage) deny("D29B_ISSUER_RUN_NOT_ADMITTED");
  const stages=await port.getStages(input.runId);
  const current=stages.find(s=>s.stage_code===run.current_stage);
  if(!current || current.lifecycle_status==="BLOCKED" ||
    asOperatorBlockers(current.blocker_summary).length>0)
    deny("D29B_ISSUER_BLOCKERS_OR_STAGE_MISSING");

  // D2 independently validates immutable bytes, hashes, manifest, contract
  // pins, lineage and state; do not replace it with a Registry-only inference.
  const decision=await loadTrustedPilotageDecision({
    port, environment:await observeSupabaseEnvironment(),
    runId:input.runId,expectedRunStateVersion:run.state_version,
    expectedCurrentStageStateVersion:current.state_version,
  });
  if(!["CONTINUE","HANDOFF"].includes(decision.outcome) ||
    !decision.exactPrompt || !decision.phase ||
    !["RESEARCH","DEEP_DIVE","INTEGRATION"].includes(current.stage_code) ||
    !["IN_PROGRESS","PAUSED","COMPLETE"].includes(current.lifecycle_status) ||
    !["CHECKPOINT","FINAL"].includes(current.active_manifest_kind??"") ||
    !current.active_manifest_artifact_id || !current.active_manifest_version)
    deny("D29B_D2_DECISION_NOT_SIGNABLE");
  if(decision.outcome==="HANDOFF" &&
    (current.lifecycle_status!=="COMPLETE" ||
     current.handoff_gate_state!=="YES" ||
     current.active_manifest_kind!=="FINAL"))
    deny("D29B_HANDOFF_NOT_ADMITTED");
  if(decision.outcome==="CONTINUE" &&
    !["IN_PROGRESS","PAUSED"].includes(current.lifecycle_status))
    deny("D29B_CONTINUE_NOT_ADMITTED");
  const verifiedArtifacts=await port.getArtifacts(input.runId);
  const manifest=verifiedArtifacts.filter(a=>a.run_id===run.run_id &&
    a.stage_code===current.stage_code &&
    a.artifact_id===current.active_manifest_artifact_id &&
    a.version===current.active_manifest_version);
  if(manifest.length!==1 || !/^[0-9a-f]{64}$/.test(manifest[0].content_sha256))
    deny("D29B_VERIFIED_MANIFEST_MISSING");

  // Re-read the observed CAS and active manifest after D2. Any intervening
  // Registry state change prevents issuance. Read-only; not an atomic lock.
  const [updated,updatedStages]=await Promise.all([
    port.getRun(input.runId),port.getStages(input.runId),
  ]);
  const updateStage=updatedStages.find(s=>s.stage_code===current.stage_code);
  if(!updated || !updateStage || updated.state_version!==run.state_version ||
    updated.current_stage!==run.current_stage ||
    updated.run_status!==run.run_status ||
    updated.data_cutoff!==run.data_cutoff ||
    updateStage.state_version!==current.state_version ||
    updateStage.stage_revision!==current.stage_revision ||
    updateStage.lifecycle_status!==current.lifecycle_status ||
    updateStage.handoff_gate_state!==current.handoff_gate_state ||
    updateStage.active_manifest_kind!==current.active_manifest_kind ||
    updateStage.active_manifest_artifact_id!==current.active_manifest_artifact_id ||
    updateStage.active_manifest_version!==current.active_manifest_version ||
    asOperatorBlockers(updateStage.blocker_summary).length>0)
    deny("D29B_REGISTRY_CHANGED_BEFORE_SIGNING");

  const now=Math.floor(Date.now()/1000);
  const prompt=decision.exactPrompt;
  const payload:HandoffAttestationV1={
    purpose:HANDOFF_PURPOSE,verifiedBy:"D2_IMMUTABLE_BYTES_AND_CONTRACTS_V3_1",
    runId:run.run_id,subject:input.subject,clientId:input.clientId,
    resource:input.resource,runStatus:"ACTIVE",
    stage:current.stage_code as StageCode,
    stageLifecycle:current.lifecycle_status as HandoffAttestationV1["stageLifecycle"],
    handoffGateState:current.handoff_gate_state,
    activeManifestKind:current.active_manifest_kind!,
    processVersion:"3.1",pilotageVersion:"3.1",dataCutoff:run.data_cutoff,
    runStateVersion:run.state_version,stageStateVersion:current.state_version,
    outcome:decision.outcome as HandoffAttestationV1["outcome"],
    phase:decision.phase,contractSetSha256:pins.contract_set_sha256,
    manifestSha256:manifest[0].content_sha256,
    stateFingerprintSha256:decision.stateFingerprintSha256,
    verifiedArtifactsCount:verifiedArtifacts.filter(a=>a.stage_code===current.stage_code).length,
    prompt,promptSha256:createHash("sha256").update(prompt,"utf8").digest("hex"),
    issuedAt:now,expiresAt:now+120,
  };
  const payloadBase64Url=Buffer.from(JSON.stringify(payload)).toString("base64url");
  if(payloadBase64Url.length>24000)deny("D29B_SIGNED_PAYLOAD_TOO_LARGE");
  const signature=await signer.sign(Buffer.from(
    HANDOFF_SIGNATURE_DOMAIN+payloadBase64Url,"utf8"));
  if(!(signature instanceof Uint8Array) || signature.byteLength!==64)
    deny("D29B_SIGNING_ADAPTER_REJECTED");
  // Remote KMS signing can take time. Reject a changed Registry or switched
  // active manifest before handing a receipt to any future private writer.
  const [finalRun,finalStages,finalArtifacts]=await Promise.all([
    port.getRun(input.runId),port.getStages(input.runId),port.getArtifacts(input.runId),
  ]);
  const finalStage=finalStages.find(x=>x.stage_code===current.stage_code);
  const finalManifest=finalArtifacts.find(x=>
    x.artifact_id===manifest[0].artifact_id &&
    x.version===manifest[0].version && x.run_id===run.run_id);
  if(!finalRun || !finalStage || !finalManifest ||
    finalRun.state_version!==run.state_version ||
    finalRun.run_status!=="ACTIVE" ||
    finalRun.current_stage!==current.stage_code ||
    finalRun.data_cutoff!==run.data_cutoff ||
    finalStage.state_version!==current.state_version ||
    finalStage.stage_revision!==current.stage_revision ||
    finalStage.lifecycle_status!==current.lifecycle_status ||
    finalStage.handoff_gate_state!==current.handoff_gate_state ||
    finalStage.active_manifest_artifact_id!==current.active_manifest_artifact_id ||
    finalStage.active_manifest_version!==current.active_manifest_version ||
    finalStage.active_manifest_kind!==current.active_manifest_kind ||
    asOperatorBlockers(finalStage.blocker_summary).length>0 ||
    finalManifest.content_sha256!==manifest[0].content_sha256)
    deny("D29B_REGISTRY_CHANGED_DURING_SIGNING");
  return {
    receipt:{version:1,keyId:signer.keyId,payloadBase64Url,
      signatureBase64Url:Buffer.from(signature).toString("base64url")},
    runId:input.runId,subject:input.subject,clientId:input.clientId,
    resource:input.resource,issuedAt:now,expiresAt:now+120,
  };
}
