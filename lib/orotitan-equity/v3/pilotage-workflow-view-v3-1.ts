/**
 * Presentation-only workflow positioning. The Registry lists observed state;
 * only a successful D2/D1 decision supplies a verified phase/handoff.
 * Step position is NOT percent-complete or analytical certification.
 */
import type { OperatorDecision, OperatorRun } from "./pilotage-operator-view-v3-1";

export const WORKFLOW_STEPS = [
  { phase: "RESEARCH", label: "Research", description: "Sources et suffisance" },
  { phase: "FUNDAMENTALS", label: "Fundamentals", description: "Qualité fondamentale" },
  { phase: "VALUATION", label: "Valorisation", description: "Calculs et valeur" },
  { phase: "CERTIFICATION_RECONCILIATION", label: "Certification", description: "Contrôles et scores" },
  { phase: "INTEGRATION", label: "Intégration", description: "Snapshot et cohérence" },
] as const;

export type WorkflowStepState = "previous" | "current" | "upcoming" | "unknown";
export type WorkflowPosition = {
  index: number; // 0 if no defensible position can be inferred
  source: "VERIFIED_HANDOFF" | "REGISTRY_OBSERVATION" | "UNKNOWN";
  title: string;
  detail: string;
  actionLabel: string;
  isBlocked: boolean;
  steps: { label: string; description: string; state: WorkflowStepState }[];
};

function observedPosition(stage: string): number {
  switch (stage) {
    case "RESEARCH": return 1;
    case "DEEP_DIVE": return 2; // Only the first possible Deep Dive phase is inferable
    case "INTEGRATION": return 5;
    default: return 0;
  }
}

function verifiedPosition(phase: string): number {
  const index = WORKFLOW_STEPS.findIndex(s => s.phase === phase);
  return index < 0 ? 0 : index + 1;
}

const ACTIONS: Record<(typeof WORKFLOW_STEPS)[number]["phase"], string> = {
  RESEARCH: "Poursuivre la collecte et la vérification des preuves",
  FUNDAMENTALS: "Préparer l’analyse fondamentale",
  VALUATION: "Préparer la valorisation",
  CERTIFICATION_RECONCILIATION: "Préparer la certification et la réconciliation",
  INTEGRATION: "Préparer l’intégration des résultats",
};

export function workflowPosition(input: {
  run: OperatorRun;
  preview: OperatorDecision | null;
  previewState: "NOT_SELECTED" | "VERIFIED" | "BLOCKED";
}): WorkflowPosition {
  const { run, preview, previewState } = input;
  // A BLOCKED verification never makes a phase authoritative.
  const verified = previewState === "VERIFIED" && preview !== null &&
    (preview.outcome === "HANDOFF" || preview.outcome === "CONTINUE") &&
    verifiedPosition(preview.phase) > 0;
  const index = verified ? verifiedPosition(preview.phase) : observedPosition(run.stage);
  const blocked = run.status === "BLOCKED" || run.stageLifecycle === "BLOCKED" ||
    previewState === "BLOCKED" || preview?.outcome === "BLOCKED" ||
    run.contractStatusCode?.startsWith("BLOCKED_") === true ||
    run.contractStatusCode === "BLOCKED_INSUFFICIENT_INPUT";
  const phase = verified ? WORKFLOW_STEPS[index - 1].phase : null;
  const current = index > 0 ? WORKFLOW_STEPS[index - 1].label : null;

  let title: string;
  let detail: string;
  let actionLabel: string;
  if (blocked) {
    title = "Points à résoudre avant de poursuivre";
    detail = "Le parcours reste bloqué ou non admissible. La position affichée ne valide aucune étape.";
    actionLabel = run.blockers.length > 0 ?
      `Examiner les ${run.blockers.length} point(s) bloquant(s)` :
      "Examiner le diagnostic et les preuves";
  } else if (verified && phase) {
    const handoff = preview.outcome === "HANDOFF";
    title = handoff ? "Prochaine phase admissible" : "Phase à poursuivre";
    detail = handoff ?
      "Prompt proposé après vérification des preuves ; copier le texte ne démarre pas la phase." :
      "Le prompt permet de continuer le travail manuel dans ChatGPT.";
    actionLabel = ACTIONS[phase];
  } else if (run.status === "PUBLISHED") {
    title = "Run historiquement publié";
    detail = "Aucune nouvelle publication ni nouvelle étape autorisée depuis ce cockpit.";
    actionLabel = "Consulter l’historique";
  } else {
    title = "Vérifier le dossier pour connaître la prochaine action";
    detail = current ?
      "Position minimale observée dans le Registry. La phase exacte et les gates ne sont pas encore vérifiés." :
      "Aucune étape identifiable sans observation fiable du Registry.";
    actionLabel = "Sélectionner le run et vérifier les preuves";
  }

  return {
    index,
    source: verified ? "VERIFIED_HANDOFF" : index > 0 ? "REGISTRY_OBSERVATION" : "UNKNOWN",
    title, detail, actionLabel, isBlocked: blocked,
    steps: WORKFLOW_STEPS.map((s, i) => ({
      label: s.label, description: s.description,
      state: index === 0 ? "unknown" : i + 1 === index ? "current" :
        i + 1 < index ? "previous" : "upcoming",
    })),
  };
}
