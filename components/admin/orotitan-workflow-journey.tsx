import { OroTitanPromptCopy } from "./orotitan-prompt-copy";
import type {
  OperatorDecision, OperatorRun,
} from "@/lib/orotitan-equity/v3/pilotage-operator-view-v3-1";
import {
  workflowPosition,
} from "@/lib/orotitan-equity/v3/pilotage-workflow-view-v3-1";

type Props = {
  run: OperatorRun;
  preview: OperatorDecision | null;
  previewState: "NOT_SELECTED" | "VERIFIED" | "BLOCKED";
};

export function OroTitanWorkflowJourney({ run, preview, previewState }: Props) {
  const journey = workflowPosition({ run, preview, previewState });
  const trusted = journey.source === "VERIFIED_HANDOFF";
  const hasPrompt = trusted && preview?.verifiedPrompt !== null &&
    typeof preview?.verifiedPrompt === "string";
  const phaseTitle = journey.index > 0 ? journey.steps[journey.index - 1].label : "Non déterminée";

  return <section aria-label="Parcours de l’analyse OroTitan"
    className="relative isolate overflow-hidden rounded-2xl border border-slate-700/80 bg-slate-900/90 shadow-[0_24px_80px_-40px_rgba(6,182,212,0.22)]">
    <div aria-hidden="true"
      className="pointer-events-none absolute -right-32 -top-40 -z-10 h-80 w-80 rounded-full bg-cyan-500/10 blur-3xl" />
    <div className="space-y-6 p-5 sm:p-7">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase tracking-[.2em] text-cyan-300">
            Tableau de bord · Parcours analytique
          </p>
          <h2 className="text-2xl font-semibold tracking-tight text-white">Votre prochaine action</h2>
          <p className="max-w-2xl text-sm leading-6 text-slate-400">
            Vue de pilotage sur ordinateur. Aucune IA n&apos;est lancée et aucune étape
            n&apos;est validée par la copie du prompt.
          </p>
        </div>
        <span className={"rounded-full border px-3 py-1.5 text-xs font-medium " +
          (trusted ? "border-cyan-800 bg-cyan-950/60 text-cyan-200" :
            "border-slate-700 bg-slate-800/70 text-slate-300")}>
          {trusted ? "Proposition vérifiée" : "Observation du Registry"}
        </span>
      </header>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
        <div className={"rounded-xl border p-5 " +
          (journey.isBlocked ? "border-amber-800/70 bg-amber-950/20" :
            trusted ? "border-cyan-800/70 bg-cyan-950/20" : "border-slate-700 bg-slate-950/60")}>
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            {journey.title}
          </p>
          <h3 className="mt-2 text-xl font-semibold leading-snug tracking-tight text-white sm:text-2xl">
            {journey.actionLabel}
          </h3>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">{journey.detail}</p>
          <p className="mt-4 text-xs text-slate-400">
            {run.blockers.length > 0 ?
              `${run.blockers.length} point(s) signalé(s) par le Registry` :
              "Aucun blocage détaillé dans la ligne observée ; cela ne prouve pas la suffisance."}
          </p>
          {journey.isBlocked && run.blockers.length > 0 ? <a
            href="#orotitan-blockers"
            className="mt-3 inline-flex rounded-lg border border-amber-700/70 px-3 py-2 text-sm font-semibold text-amber-200 transition hover:bg-amber-900/30 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-300">
            Consulter les points à résoudre →
          </a> : null}
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-5">
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Run sélectionné</p>
          <p className="mt-2 break-all font-mono text-xs text-slate-100">{run.runId}</p>
          <dl className="mt-4 grid grid-cols-2 gap-3 text-sm xl:grid-cols-1">
            <div><dt className="text-xs text-slate-500">Étape Registry</dt>
              <dd className="mt-1 font-medium text-slate-200">{run.stage}</dd></div>
            <div><dt className="text-xs text-slate-500">Data cutoff</dt>
              <dd className="mt-1 font-medium tabular-nums text-slate-200">{run.cutoff}</dd></div>
            <div><dt className="text-xs text-slate-500">Manifeste actif</dt>
              <dd className="mt-1 font-medium text-slate-200">{run.activeManifestKind ?? "Non indiqué"}</dd></div>
          </dl>
          <p className="mt-3 text-xs text-slate-500">Métadonnées observées · preuves vérifiées séparément</p>
        </div>
      </div>

      <div className="space-y-4" aria-label="Progression des phases">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="text-sm font-semibold text-white">Position dans les 5 phases</h3>
          <p className="text-xs tabular-nums text-slate-400">
            {journey.index > 0 ? `${journey.index} / 5 · ${phaseTitle}` : "Phase non déterminée"}
          </p>
        </div>
        <div role="progressbar" aria-label="Position dans le parcours, pas un pourcentage de travaux terminés"
          aria-valuemin={0} aria-valuemax={5} aria-valuenow={journey.index}
          aria-valuetext={journey.index > 0 ?
            `Phase ${journey.index} sur 5 : ${phaseTitle}. Position, non taux d’achèvement.` :
            "Position indéterminée"}>
          <div className="grid grid-cols-5 gap-1.5" aria-hidden="true">
            {journey.steps.map((step, index) =>
              <div key={step.label} className={"h-1.5 rounded-full " +
                (index + 1 > journey.index ? "bg-slate-700/80" :
                  journey.isBlocked ? "bg-amber-500/60" :
                    trusted ? "bg-cyan-400" : "bg-slate-400/70")} />)}
          </div>
        </div>
        <ol className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5" aria-label="Cinq phases du processus">
          {journey.steps.map((step, index) => {
            const here = step.state === "current";
            const prior = step.state === "previous";
            const phaseState = here ? journey.isBlocked ? "À résoudre" :
              trusted ? preview?.outcome === "HANDOFF" ? "Proposée" : "À poursuivre" : "Observée" :
              prior ? "Étape antérieure" : "À venir";
            return <li key={step.label} aria-current={here ? "step" : undefined}
              className={"min-w-0 rounded-xl border p-3.5 " +
                (here ? journey.isBlocked ?
                  "border-amber-700/80 bg-amber-950/20" :
                  "border-cyan-700/70 bg-cyan-950/20" :
                  "border-slate-800 bg-slate-950/40")}>
              <div className="flex items-center justify-between gap-2">
                <span className={"font-mono text-xs tabular-nums " +
                  (here ? journey.isBlocked ? "text-amber-300" : "text-cyan-300" :
                    "text-slate-500")}>
                  {String(index + 1).padStart(2, "0")}
                </span>
                {here ? <span aria-hidden="true"
                  className={"h-2 w-2 shrink-0 rounded-full " +
                    (journey.isBlocked ? "bg-amber-400" : "bg-cyan-400")} /> : null}
              </div>
              <p className={"mt-3 text-sm font-semibold " +
                (here ? "text-white" : "text-slate-300")}>{step.label}</p>
              <p className="mt-1 text-xs leading-5 text-slate-500">{step.description}</p>
              <p className={"mt-3 text-[11px] font-medium " +
                (here ? journey.isBlocked ? "text-amber-200" : "text-cyan-200" :
                  "text-slate-500")}>{phaseState}</p>
            </li>;
          })}
        </ol>
        <p className="text-xs leading-5 text-slate-500">
          Barre de position uniquement : elle ne mesure ni le temps passé,
          ni un pourcentage de recherche réalisée. Les étapes antérieures
          ne constituent pas une certification de leurs résultats.
        </p>
      </div>

      {hasPrompt && preview ? <div className="rounded-xl border border-slate-700 bg-slate-950/60 p-4 sm:p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="text-base font-semibold text-white">Prompt de travail prêt à copier</h3>
          <span className="text-xs text-slate-400">Exécution manuelle dans ChatGPT</span>
        </div>
        <OroTitanPromptCopy prompt={preview.verifiedPrompt!} phase={preview.phase} />
      </div> : <p className="rounded-lg border border-slate-800 bg-slate-950/50 px-4 py-3 text-sm leading-6 text-slate-300">
        Aucun prompt de phase suivante admis. Consulter le diagnostic et les éventuels
        blocages, puis relancer la vérification après leur résolution.
      </p>}
    </div>
  </section>;
}
