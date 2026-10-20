import Link from "next/link";
import { redirect } from "next/navigation";
import type { Metadata } from "next";

import { AdminNav } from "@/components/admin/admin-nav";
import { OroTitanWorkflowJourney } from "@/components/admin/orotitan-workflow-journey";
import { isAdminAuthenticated } from "@/lib/auth/admin-session";
import { loadAdminPilotageCockpit } from "@/lib/orotitan-equity/v3/pilotage-operator-cockpit-server";
import { filterOperatorRuns, hasObservedBlockers, operatorTriageTotals, type OperatorFilter } from "@/lib/orotitan-equity/v3/pilotage-operator-view-v3-1";

export const metadata: Metadata = { title: "OroTitan — Parcours des analyses" };
export const dynamic = "force-dynamic";

const explanations: Record<string, string> = {
  ENVIRONMENT_OBSERVATION_TOKEN_MISSING: "Vérification approfondie indisponible : configurer le jeton de lecture Management API Supabase côté serveur.",
  ENVIRONMENT_OBSERVATION_FETCH_FAILED: "Impossible de confirmer l’environnement de production ; aucune preuve n’est admise.",
  ENVIRONMENT_OBSERVATION_HTTP_REJECTED: "Le jeton Management API ne permet pas de confirmer ce projet.",
  ENVIRONMENT_OBSERVATION_NOT_ADMITTED: "L’état, le nom ou la région du projet ne correspondent pas aux autorités épinglées.",
  RUN_NOT_FOUND: "Run absent du Registry.",
  LEGACY_RUN_USE_VERSIONED_PILOTAGE: "Ce run utilise une méthodologie historique. Consultez les données enregistrées ; le vérificateur V3.1 ne s’applique pas à sa version.",
  RUN_NOT_EXECUTABLE_UPSTREAM: "Le Registry signale un run bloqué, en pause ou annulé. Corrigez d’abord ses blocages enregistrés avant toute reprise.",
  REGISTRY_STAGES_QUERY_FAILED: "Impossible de consulter les étapes du Registry. Ne pas interpréter l’absence de blocages comme un feu vert.",
  CURRENT_STAGE_MISSING: "L’étape courante déclarée par le run est absente du Registry.",
  D5_FULL_PRECISION_INPUT_NOT_VERIFIED: "Artefact full-precision requis non vérifié.",
  D6_PROPOSED_SCHEMA_INVALID: "Format de provenance D6 invalide ou non conforme.",
  V31_PUBLICATION_WRITER_NOT_ADMITTED: "Le writer V3.1 n’est pas homologué ; publication interdite.",
  REGISTRY_LIST_QUERY_FAILED: "Impossible de charger le Registry ; vérifier la connexion Supabase côté serveur.",
};

const STAGE_NAMES: Record<string, string> = {
  RESEARCH: "Research", DEEP_DIVE: "Deep Dive", INTEGRATION: "Intégration",
};
const RUN_STATUSES: Record<string, string> = {
  ACTIVE: "Actif", BLOCKED: "Bloqué", PAUSED: "En pause",
  PUBLISHED: "Publié", READY_TO_PUBLISH: "Publication en attente",
};
function displayStage(code: string): string {
  return STAGE_NAMES[code] ?? code;
}
function displayStatus(code: string): string {
  return RUN_STATUSES[code] ?? code;
}

function getMessage(code: string | null): string {
  if (!code) return "";
  return explanations[code] ??
    "Admission refusée à cette étape. Examiner le code de blocage et les pièces du Registry avant toute correction.";
}

export default async function OroTitanPilotagePage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  if (!(await isAdminAuthenticated())) redirect("/admin");
  const query = await searchParams;
  const selected = typeof query.run === "string" ? query.run : null;
  const view: OperatorFilter = query.view === "all" || query.view === "blocked" ? query.view : "v31";
  let data: Awaited<ReturnType<typeof loadAdminPilotageCockpit>> | null = null;
  try { data = await loadAdminPilotageCockpit(selected); }
  catch { /* No internal exception details or secrets in the administrator UI. */ }
  const totals = data ? operatorTriageTotals(data.runs) : null;
  const displayed = data ? filterOperatorRuns(data.runs, view) : [];
  const currentObservation = selected ? data?.runs.find(row => row.runId === selected) : null;

  return <main className="mx-auto max-w-7xl space-y-6 pb-12">
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl space-y-2">
        <p className="text-xs font-semibold uppercase tracking-[.2em] text-cyan-400">
          OroTitan · Espace de travail
        </p>
        <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
          Pilotage des analyses
        </h1>
        <p className="text-sm leading-6 text-slate-400">
          Suivez les phases, identifiez la prochaine action et copiez le prompt de travail.
          L&apos;analyse reste manuelle dans ChatGPT ; aucune IA ni transition automatique.
        </p>
      </div>
      <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-300">
        Mode manuel · V3.1
      </span>
    </header>
    <AdminNav />
    <aside className="flex flex-wrap items-start gap-3 rounded-xl border border-amber-900/60 bg-amber-950/15 px-4 py-3 text-sm leading-6 text-amber-100"
      aria-label="Limite de publication">
      <span aria-hidden="true" className="font-bold text-amber-400">!</span>
      <p className="flex-1"><strong>Publication V3.1 non autorisée.</strong> Le cockpit permet
        la lecture et la préparation des prompts, jamais la publication ni la modification
        du Registry. Les contrôles I2/I3-B et le writer restent distincts.</p>
    </aside>

    <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 sm:p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-lg font-semibold text-white">Ouvrir un dossier</h2>
        <p className="text-xs text-slate-400">Sélectionnez un run existant · accès lecture seule</p>
      </div>
      <form method="GET" className="mt-4 flex flex-col gap-3 sm:flex-row" action="/admin/orotitan">
        <input type="text" name="run" required defaultValue={selected ?? ""}
          placeholder="Identifiant du run (UUID)"
          aria-label="Identifiant UUID du run"
          className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 font-mono text-sm text-white outline-none transition focus:border-cyan-500 focus:ring-2 focus:ring-cyan-700/40"
          maxLength={36} />
        <button className="rounded-lg bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300">
          Ouvrir et vérifier
        </button>
      </form>
      <p className="mt-2 text-xs text-slate-500">La liste est une observation du Registry, non une validation des artefacts.
        La vérification complète nécessite aussi une observation indépendante de l’état Supabase.</p>
    </section>

    {currentObservation && data ? <OroTitanWorkflowJourney
      run={currentObservation}
      preview={data.preview}
      previewState={data.previewState}
    /> : null}

    {!data ? <section role="status" className="rounded-xl border border-rose-900 bg-rose-950/20 p-4 text-sm text-rose-200">
      Le diagnostic n’est pas disponible. Vérifier la configuration serveur et la connexion Registry.
    </section> : <>
      {totals && <section aria-label="État de la file observée" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["Runs observés", totals.observed],
          ["En cours", totals.active],
          ["Avec points à traiter", totals.blocked],
          ["V3.1 à traiter", filterOperatorRuns(data.runs, "v31").length],
        ].map(([label, value]) => <div key={label} className="rounded-lg border border-slate-800 bg-slate-900/65 p-4">
          <p className="text-xs text-slate-400">{label}</p>
          <p className="mt-1 text-2xl font-semibold text-white">{value}</p>
        </div>)}
        <p className="col-span-full text-xs text-slate-500">
          {totals.v31} run(s) V3.1, dont {totals.v31Active} actif(s) et {totals.v31Blocked} avec points à traiter.
          Ces catégories peuvent se recouper. Observations non vérifiées cryptographiquement, parmi les 100 derniers runs.
        </p>
      </section>}

      {currentObservation && <section id="orotitan-blockers" className="space-y-3 rounded-2xl border border-slate-700 bg-slate-900/70 p-5 scroll-mt-6 sm:p-6">
        <div><p className="text-xs uppercase tracking-wider text-slate-400">État enregistré — non vérifié par D2–D8</p>
          <h2 className="mt-1 text-lg font-semibold text-white">
            Run {currentObservation.processVersion} · {currentObservation.stage} · {currentObservation.status}
          </h2>
        </div>
        <p className="text-xs text-slate-400">
          Étape : {currentObservation.stageLifecycle} · Handoff : {currentObservation.handoffGateState}
          · Manifeste : {currentObservation.activeManifestKind ?? "aucun"}
        </p>
        {currentObservation.contractStatusCode && <p className="break-all font-mono text-xs text-amber-200">
          Code Registry : {currentObservation.contractStatusCode}
        </p>}
        {currentObservation.blockers.length > 0 ? <div>
          <h3 className="font-semibold text-amber-100">Blocages et résolutions enregistrés</h3>
          <ul className="mt-3 space-y-3">
            {currentObservation.blockers.map((item, i) => <li key={i} className="rounded-lg border border-amber-900/50 bg-amber-950/10 p-3 text-sm text-slate-200">
              {item.code && <p className="mb-1 font-mono text-xs text-amber-300">{item.code}</p>}
              <p className="break-words">{item.text}</p>
              {item.resolution && <p className="mt-2 break-words text-cyan-200">
                Résolution demandée : {item.resolution}
              </p>}
            </li>)}
          </ul>
        </div> : <p className="text-sm text-slate-400">
          Aucun blocage détaillé dans ce champ du Registry. Cela ne prouve pas l’admissibilité du run.
        </p>}
      </section>}

      {data.previewState === "BLOCKED" ? <section role="status" className="rounded-xl border border-amber-800 bg-amber-950/20 p-5">
        <h2 className="font-semibold text-amber-100">Vérification bloquée — aucune admission</h2>
        <p className="mt-2 font-mono text-sm text-amber-200">{data.blockerCode}</p>
        <p className="mt-2 text-sm leading-6 text-amber-100/90">{getMessage(data.blockerCode)}</p>
      </section> : null}

      {data.preview && <section className="space-y-4 rounded-2xl border border-cyan-900/70 bg-slate-900/70 p-5 sm:p-6">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-cyan-400">Contrôle cryptographique D2–D8</p>
          <h2 className="mt-2 text-lg font-semibold text-white">Diagnostic vérifié : {data.preview.outcome}</h2>
          <p className="mt-1 text-sm text-slate-400">
            Phase : {data.preview.phase} · Code : <code>{data.preview.reasonCode}</code>
          </p>
        </div>
        <p className="text-sm text-slate-200">{data.preview.nextAction}</p>
        <dl className="grid gap-3 text-xs text-slate-400 sm:grid-cols-2">
          <div><dt>Version run</dt><dd className="font-mono text-white">{data.preview.runStateVersion}</dd></div>
          <div><dt>Version étape</dt><dd className="font-mono text-white">{data.preview.stageStateVersion}</dd></div>
          <div className="sm:col-span-2"><dt>Empreinte d’état observé</dt>
            <dd className="break-all font-mono text-white">{data.preview.stateFingerprintSha256}</dd></div>
        </dl>
      </section>}

      <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 sm:p-6">
        <h2 className="text-lg font-semibold text-white">Mes dossiers et historique</h2>
        <p className="mt-1 text-xs text-slate-500">
          Observation uniquement · les versions historiques ne sont pas évaluées par le vérificateur V3.1.
        </p>
        <nav aria-label="Filtrer les runs" className="mt-4 flex flex-wrap gap-2">
          {[
            ["v31", "V3.1 à traiter"],
            ["blocked", "Points à traiter"],
            ["all", "Toutes les versions"],
          ].map(([filter,label]) => <Link key={filter} href={"/admin/orotitan?view=" + filter}
            className={"rounded-lg border px-3 py-2 text-sm " +
              (view === filter ? "border-cyan-700 bg-slate-800 text-white" :
              "border-slate-700 text-slate-300 hover:border-cyan-800")}>{label}</Link>)}
        </nav>
        {displayed.length === 0 ? <p className="mt-4 text-sm text-slate-400">
          Aucun run dans ce filtre. Vous pouvez consulter les autres versions.
        </p> :
          <div className="mt-4 space-y-3">
            {displayed.map(run => <Link key={run.runId}
              href={"/admin/orotitan?run=" + encodeURIComponent(run.runId) + "&view=" + view}
              className={"block rounded-xl border p-4 text-sm transition hover:border-cyan-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300 " +
                (run.runId === data.selectedRunId ? "border-cyan-800 bg-slate-800" : "border-slate-800 bg-slate-950/40")}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="break-all font-mono text-xs text-cyan-200">{run.runId}</span>
                <span className={"rounded-full border px-2.5 py-1 text-xs font-medium " +
                  (hasObservedBlockers(run) ? "border-amber-800 bg-amber-950/30 text-amber-200" :
                    "border-slate-700 bg-slate-900 text-slate-300")}>
                  V{run.processVersion} · {hasObservedBlockers(run) ? "À examiner" : displayStatus(run.status)}
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-400">
                {displayStage(run.stage)} · {run.stageLifecycle} · {run.mode} · cutoff {run.cutoff} · état #{run.stateVersion}
              </p>
              {run.contractStatusCode && <p className="mt-2 break-all font-mono text-xs text-amber-200">
                {run.contractStatusCode}
              </p>}
              {run.blockers.length > 0 && <p className="mt-2 line-clamp-2 text-xs text-amber-100">
                {run.blockers.length} point(s) affiché(s) · {run.blockers[0].text}
              </p>}
            </Link>)}
          </div>}
      </section>
    </>}
  </main>;
}
