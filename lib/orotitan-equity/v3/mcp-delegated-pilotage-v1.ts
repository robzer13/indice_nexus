/** D29a: delegated OAuth Pilotage remains observational until an independently
 * verified, caller-scoped immutable-proof reader is separately approved.
 * Registry rows, CHECKPOINTs and human blocker summaries are NOT executable
 * prompts, stage admission or certification authority.
 */
import type { OperatorBlocker } from "./pilotage-operator-view-v3-1";

export type DelegatedPilotageObservation = {
  runId: string;
  issuerName: string | null;
  ticker: string | null;
  runStatus: string;
  registryStage: string | null;
  stageLifecycle: string;
  contractStatusCode: string | null;
  handoffGateName: string | null;
  handoffGateState: string;
  activeManifestKind: string | null;
  processVersion: string;
  pilotageVersion: string;
  runStateVersion: number;
  stageStateVersion: number | null;
  blockers: readonly OperatorBlocker[];
};

export function delegatedPilotageStatus(x: DelegatedPilotageObservation) {
  const versionSupported=x.processVersion==="3.1" && x.pilotageVersion==="3.1";
  const executableRun=x.runStatus==="ACTIVE" || x.runStatus==="READY_TO_PUBLISH";
  const stageKnown=x.registryStage==="RESEARCH" ||
    x.registryStage==="DEEP_DIVE" || x.registryStage==="INTEGRATION";
  const hasStage=x.stageStateVersion!==null && stageKnown &&
    x.stageLifecycle!=="UNKNOWN";
  const observedBlock=x.contractStatusCode?.startsWith("BLOCKED_")===true ||
    x.blockers.length>0 || x.stageLifecycle==="BLOCKED" ||
    x.handoffGateState==="NO";
  const status=!versionSupported ? "UNSUPPORTED_VERSION" :
    !executableRun ? "RUN_NOT_EXECUTABLE_UPSTREAM" :
    !hasStage ? "STAGE_OBSERVATION_MISSING" :
    observedBlock ? "BLOCKED_OBSERVED" : "UNVERIFIED_NO_PROMPT";
  const guidance=status==="BLOCKED_OBSERVED" ?
    "Treat blockers as unverified Registry observations; resolve them in the existing Research execution and re-run independent immutable-evidence verification. Do not advance stage." :
    status==="UNVERIFIED_NO_PROMPT" ?
    "Registry state alone cannot certify a handoff. Ask for independently verified immutable artifacts before preparing an execution prompt." :
    "No actionable handoff or verified execution prompt is available. Keep the run and its stage unchanged.";
  return {
    runId:x.runId,
    issuerName:x.issuerName,
    ticker:x.ticker,
    status,
    registryStage:x.registryStage,
    runStatus:x.runStatus,
    stageLifecycle:x.stageLifecycle,
    contractStatusCode:x.contractStatusCode,
    handoffGateName:x.handoffGateName,
    handoffGateState:x.handoffGateState,
    activeManifestKind:x.activeManifestKind,
    blockers:x.blockers,
    runStateVersion:x.runStateVersion,
    stageStateVersion:x.stageStateVersion,
    trust:"REGISTRY_OBSERVATION_NOT_ARTIFACT_VERIFICATION" as const,
    verificationPerformed:false as const,
    verifiedPrompt:null,
    prompt:null,
    stateFingerprintSha256:null,
    nextStepAuthorized:false as const,
    publicationAuthorized:false as const,
    mutationPerformed:false as const,
    guidance,
  };
}
