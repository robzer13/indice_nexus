/**
 * D9 operator cockpit projection; no actions, mutations or inferred I2/I3B authority.
 * The Registry listing is an observation, not cryptographic verification.
 */
import type { PilotageDecision } from "./pilotage-adapter-v3-1";

export type RunSummary = {
  run_id: string;
  run_status: string;
  current_stage: string | null;
  canonical_mode: string;
  data_cutoff: string;
  issuer_id: string | null;
  state_version: number;
  created_at: string;
};

export type OperatorRun = {
  runId: string;
  status: string;
  stage: string;
  mode: string;
  cutoff: string;
  stateVersion: number;
  createdAt: string;
};

export type OperatorDecision = {
  trust: "VERIFIED_FROM_IMMUTABLE_BYTES";
  outcome: PilotageDecision["outcome"];
  phase: string;
  reasonCode: string;
  nextAction: string;
  verifiedPrompt: string | null;
  runStateVersion: number;
  stageStateVersion: number;
  stateFingerprintSha256: string;
  publicationAuthorized: false;
  mutationPerformed: false;
};

export function asOperatorRun(row: RunSummary): OperatorRun {
  return {
    runId: row.run_id, status: row.run_status,
    stage: row.current_stage ?? "NOT_ASSIGNED", mode: row.canonical_mode,
    cutoff: row.data_cutoff, stateVersion: row.state_version,
    createdAt: row.created_at,
  };
}

const ACTION: Record<PilotageDecision["outcome"], string> = {
  CONTINUE: "Continuer l'analyse dans la phase indiquée après validation des preuves.",
  HANDOFF: "Handoff prêt pour revue humaine : aucune étape n'a été démarrée.",
  BLOCKED: "Corriger le blocage et relancer la vérification ; aucune admission.",
  DONE: "Run historiquement publié ; aucune nouvelle publication.",
  AWAIT_EXPLICIT_GO_PUBLISH: "Publication interdite : contrôles I2/I3-B, writer et autorisation séparée requis.",
};

export function asOperatorDecision(decision: PilotageDecision): OperatorDecision {
  // A planner's advisory output cannot override hard production publication gates.
  const outcome = decision.outcome === "AWAIT_EXPLICIT_GO_PUBLISH" ?
    "BLOCKED" : decision.outcome;
  return {
    trust: "VERIFIED_FROM_IMMUTABLE_BYTES",
    outcome,
    phase: decision.phase ?? "AUCUNE",
    reasonCode: decision.outcome === "AWAIT_EXPLICIT_GO_PUBLISH" ?
      "V31_PUBLICATION_WRITER_NOT_ADMITTED" : decision.reasonCode,
    nextAction: decision.outcome === "AWAIT_EXPLICIT_GO_PUBLISH" ?
      ACTION.AWAIT_EXPLICIT_GO_PUBLISH : ACTION[outcome],
    verifiedPrompt: outcome === "HANDOFF" || outcome === "CONTINUE" ?
      decision.exactPrompt : null,
    runStateVersion: decision.expectedRunStateVersion,
    stageStateVersion: decision.expectedStageStateVersion,
    stateFingerprintSha256: decision.stateFingerprintSha256,
    publicationAuthorized: false,
    mutationPerformed: false,
  };
}

export function safeVerifierReason(error: unknown): string {
  if (error && typeof error === "object" && "code" in error &&
    typeof (error as { code?: unknown }).code === "string") {
    const code = (error as { code: string }).code;
    if (/^[A-Z][A-Z0-9_]{1,120}$/.test(code)) return code;
  }
  return "VERIFICATION_UNAVAILABLE";
}
