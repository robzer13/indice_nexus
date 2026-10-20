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
  process_version: string;
  pilotage_contract_version: string;
};

export type OperatorStageObservation = {
  run_id: string;
  stage_code: string;
  lifecycle_status: string;
  contract_status_code: string | null;
  handoff_gate_state: string;
  active_manifest_kind: string | null;
  stage_revision: number;
  blocker_summary: unknown;
};

export type OperatorBlocker = {
  code: string | null;
  text: string;
  resolution: string | null;
};

export type OperatorRun = {
  runId: string;
  status: string;
  stage: string;
  mode: string;
  cutoff: string;
  stateVersion: number;
  createdAt: string;
  processVersion: string;
  pilotageVersion: string;
  stageLifecycle: string;
  contractStatusCode: string | null;
  handoffGateState: string;
  activeManifestKind: string | null;
  blockers: OperatorBlocker[];
  supportsV31Verifier: boolean;
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

/**
 * Interpret heterogeneous Registry blocker JSON for human triage. Values remain
 * observations from unverified rows and are not prompts or certification.
 */
export function asOperatorBlockers(raw: unknown): OperatorBlocker[] {
  if (!Array.isArray(raw)) return [];
  return raw.slice(0, 8).map((entry): OperatorBlocker | null => {
    if (typeof entry === "string") {
      const text = entry.trim().slice(0, 420);
      return text ? {code: null, text, resolution: null} : null;
    }
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) return null;
    const x = entry as Record<string, unknown>;
    const message = [x.description, x.detail, x.summary, x.diagnostic]
      .find(v => typeof v === "string" && v.trim().length > 0);
    const code = [x.code, x.blocker_code, x.blocker_id]
      .find(v => typeof v === "string" && v.trim().length > 0);
    const resolution = [x.required_resolution, x.resolution_required]
      .find(v => typeof v === "string" && v.trim().length > 0);
    return {
      code: typeof code === "string" ? code.slice(0, 100) : null,
      text: typeof message === "string" ? message.trim().slice(0, 420) : "Blocage signalé sans description lisible.",
      resolution: typeof resolution === "string" ? resolution.trim().slice(0, 420) : null,
    };
  }).filter((value): value is OperatorBlocker => value !== null);
}

export function supportsPilotageV31(processVersion: string, pilotageVersion: string): boolean {
  return processVersion === "3.1" && pilotageVersion === "3.1";
}

export function asOperatorRun(
  row: RunSummary,
  stage: OperatorStageObservation | null = null,
): OperatorRun {
  const relevant = stage?.stage_code === row.current_stage && stage.run_id === row.run_id ? stage : null;
  return {
    runId: row.run_id, status: row.run_status,
    stage: row.current_stage ?? "NOT_ASSIGNED", mode: row.canonical_mode,
    cutoff: row.data_cutoff, stateVersion: row.state_version,
    createdAt: row.created_at,
    processVersion: row.process_version, pilotageVersion: row.pilotage_contract_version,
    stageLifecycle: relevant?.lifecycle_status ?? "UNKNOWN",
    contractStatusCode: relevant?.contract_status_code ?? null,
    handoffGateState: relevant?.handoff_gate_state ?? "UNKNOWN",
    activeManifestKind: relevant?.active_manifest_kind ?? null,
    blockers: asOperatorBlockers(relevant?.blocker_summary),
    supportsV31Verifier: supportsPilotageV31(row.process_version, row.pilotage_contract_version),
  };
}

export type OperatorFilter = "v31" | "blocked" | "all";

/** Registry observations, not a verified gate or permission to advance. */
export function hasObservedBlockers(x: OperatorRun): boolean {
  return x.status === "BLOCKED" || x.stageLifecycle === "BLOCKED" ||
    x.contractStatusCode?.startsWith("BLOCKED_") === true || x.blockers.length > 0;
}

export function filterOperatorRuns(rows: readonly OperatorRun[], filter: OperatorFilter): OperatorRun[] {
  if (filter === "blocked") return rows.filter(hasObservedBlockers);
  if (filter === "v31") return rows.filter(x => x.supportsV31Verifier && x.status !== "PUBLISHED");
  return [...rows];
}

export function operatorTriageTotals(rows: readonly OperatorRun[]) {
  return {
    observed: rows.length,
    active: rows.filter(x => x.status === "ACTIVE").length,
    blocked: rows.filter(hasObservedBlockers).length,
    v31: rows.filter(x => x.supportsV31Verifier).length,
    v31Active: rows.filter(x => x.supportsV31Verifier && x.status === "ACTIVE").length,
    v31Blocked: rows.filter(x => x.supportsV31Verifier && hasObservedBlockers(x)).length,
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
