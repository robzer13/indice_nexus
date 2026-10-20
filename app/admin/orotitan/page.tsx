import Link from "next/link";
import { redirect } from "next/navigation";
import type { Metadata } from "next";

import { AdminNav } from "@/components/admin/admin-nav";
import { startFundamentalsFromResearch } from "./actions";
import { isAdminAuthenticated } from "@/lib/auth/admin-session";
import { loadAdminPilotageCockpit } from "@/lib/orotitan-equity/v3/pilotage-operator-cockpit-server";
import { filterOperatorRuns, operatorTriageTotals, type OperatorFilter } from "@/lib/orotitan-equity/v3/pilotage-operator-view-v3-1";

export const metadata: Metadata = { title: "Pilotage OroTitan V3.1 — diagnostic" };
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
  RESEARCH_FINAL_NOT_VERIFIED: "Research FINAL non admis : vérifier la suffisance et les sept artefacts avant Fundamentals.",
  DOWNSTREAM_STAGE_ALREADY_REGISTERED: "Deep Dive existe déjà : utiliser la reprise plutôt que créer une seconde étape.",
  STALE_RUN_CAS: "Le run a évolué. Actualiser le diagnostic avant de relancer la transition.",
  STALE_STAGE_CAS: "L’étape Research a évolué. Actualiser avant de relancer.",
  INVALID_EXPECTED_VERSIONS: "Versions du Registry manquantes ou invalides.",
  REGISTRY_STAGE_START_REJECTED: "Le Registry a refusé la transition ; aucune admission forcée.",
  REGISTRY_STAGE_START_NOT_CONFIRMED: "Réponse Registry inattendue. Vérifier l’état courant avant toute répétition.",
  RESEARCH_STAGE_NOT_EXECUTABLE: "Research ne satisfait pas les préconditions d’exécution.",
};

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
  const transition = typeof query.transition === "string" ? query.transition : null;
  const view: OperatorFilter = query.view === "all" || query.view === "blocked" ? query.view : "v31";
  let data: Awaited<ReturnType<typeof loadAdminPilotageCockpit>> | null = null;
  try { data = await loadAdminPilotageCockpit(selected); }
  catch { /* No internal exception details or secrets in the administrator UI. */ }
  const totals = data ? operatorTriageTotals(data.runs) : null;
  const displayed = data ? filterOperatorRuns(data.runs, view) : [];
  const currentObservation = selected ? data?.runs.find(row => row.runId === selected) : null;

  return <main className="space-y-6">
    <header className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-[.2em] text-cyan-400">OroTitan · Pilotage V3.1</p>
      <h1 className="text-3xl font-semibold text-white">Diagnostic des runs</h1>
      <p className="max-w-3xl text-sm leading-6 text-slate-400">
        État réel du Registry et vérification des preuves immuables. Le démarrage de Fundamentals
        est une action administrateur explicite, limitée aux Research FINAL admissibles.
        Aucune création de run, validation I3-B ou publication ici.
      </p>
    </header>
    <AdminNav />
    <section className="rounded-xl border border-amber-800/70 bg-amber-950/20 p-4 text-sm text-amber-100" aria-label="Limite de publication">
      <strong>Publication V3.1 interdite.</strong> Le gel D8 est un gel technique de revue,
      pas une homologation du writer. Les contrôles I2/I3-B, l’attestation de publication
      et le CAS transactionnel demeurent des prérequis distincts.
    </section>

    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <h2 className="font-semibold text-white">Rechercher un run</h2>
      <form method="GET" className="mt-3 flex flex-col gap-3 sm:flex-row" action="/admin/orotitan">
        <input type="text" name="run" required defaultValue={selected ?? ""}
          placeholder="UUID du run OroTitan"
          aria-label="Identifiant UUID du run"
          className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 font-mono text-sm text-white"
          maxLength={36} />
        <button className="rounded-lg bg-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-300">
          Vérifier en lecture seule
        </button>
      </form>
      <p className="mt-2 text-xs text-slate-500">La liste est une observation du Registry, non une validation des artefacts.
        La vérification complète nécessite aussi une observation indépendante de l’état Supabase.</p>
    </section>

    {transition && <section role="status"
      className={"rounded-xl border p-4 text-sm " +
        (transition === "started" && currentObservation?.stage === "DEEP_DIVE" &&
        currentObservation.stageLifecycle === "IN_PROGRESS" ?
          "border-emerald-800 bg-emerald-950/20 text-emerald-100" :
          "border-amber-800 bg-amber-950/20 text-amber-100")}>
      {transition === "started" ?
        (currentObservation?.stage === "DEEP_DIVE" && currentObservation.stageLifecycle === "IN_PROGRESS" ?
          "Deep Dive démarré dans le Registry. Fundamentals peut maintenant être exécuté." :
          "Transition demandée. Vérifier le run courant dans le Registry avant de poursuivre.") :
        <><strong>Transition non admise : </strong><code>{transition}</code>. {getMessage(transition)}</>}
    </section>}

    {!data ? <section role="status" className="rounded-xl border border-rose-900 bg-rose-950/20 p-4 text-sm text-rose-200">
      Le diagnostic n’est pas disponible. Vérifier la configuration serveur et la connexion Registry.
    </section> : <>
      {totals && <section aria-label="État de la file observée" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["Runs observés", totals.observed],
          ["En cours", totals.active],
          ["Bloqués", totals.blocked],
          ["V3.1 à traiter", totals.v31Active + totals.v31Blocked],
        ].map(([label, value]) => <div key={label} className="rounded-lg border border-slate-800 bg-slate-900/65 p-4">
          <p className="text-xs text-slate-400">{label}</p>
          <p className="mt-1 text-2xl font-semibold text-white">{value}</p>
        </div>)}
        <p className="col-span-full text-xs text-slate-500">
          {totals.v31} run(s) V3.1, dont {totals.v31Active} actif(s) et {totals.v31Blocked} bloqué(s).
          Statuts observés, sans preuve cryptographique. Fenêtre limitée aux 100 derniers runs.
        </p>
      </section>}

      {currentObservation && <section className="space-y-3 rounded-xl border border-slate-700 bg-slate-900/70 p-5">
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

      {data.preview && <section className="space-y-4 rounded-xl border border-cyan-900/70 bg-slate-900/70 p-5">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-cyan-400">Contrôle cryptographique D2–D8</p>
          <h2 className="mt-2 text-lg font-semibold text-white">Diagnostic vérifié : {data.preview.outcome}</h2>
          <p className="mt-1 text-sm text-slate-400">
            Phase : {data.preview.phase} · Code : <code>{data.preview.reasonCode}</code>
          </p>
        </div>
        <p className="text-sm text-slate-200">{data.preview.nextAction}</p>
        {data.preview.outcome === "HANDOFF" &&
        data.preview.phase === "FUNDAMENTALS" &&
        data.preview.reasonCode === "RESEARCH_FINAL_ADMITTED_START_REQUIRED" &&
        selected ? <form action={startFundamentalsFromResearch}
          className="space-y-3 rounded-lg border border-cyan-800/70 bg-slate-950/70 p-4">
          <p className="text-sm text-slate-200">
            Research FINAL vérifié : les sept sorties exigées sont admises. Le Registry
            peut créer la première étape Deep Dive et ouvrir Fundamentals.
          </p>
          <input type="hidden" name="runId" value={selected} />
          <input type="hidden" name="expectedRunStateVersion" value={data.preview.runStateVersion} />
          <input type="hidden" name="expectedResearchStageStateVersion" value={data.preview.stageStateVersion} />
          <button type="submit"
            className="rounded-lg bg-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-300">
            Démarrer Fundamentals
          </button>
          <p className="text-xs text-slate-400">
            Nouvelle vérification des preuves avant l’action et CAS transactionnel.
            Aucun contournement des blocages Research ni publication.
          </p>
        </form> : null}
        <dl className="grid gap-3 text-xs text-slate-400 sm:grid-cols-2">
          <div><dt>Version run</dt><dd className="font-mono text-white">{data.preview.runStateVersion}</dd></div>
          <div><dt>Version étape</dt><dd className="font-mono text-white">{data.preview.stageStateVersion}</dd></div>
          <div className="sm:col-span-2"><dt>Empreinte d’état observé</dt>
            <dd className="break-all font-mono text-white">{data.preview.stateFingerprintSha256}</dd></div>
        </dl>
        {data.preview.verifiedPrompt ? <details className="border-t border-slate-800 pt-4">
          <summary className="cursor-pointer text-sm font-medium text-cyan-300">
            Voir le handoff proposé (non exécuté)
          </summary>
          <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-slate-950 p-4 text-xs text-slate-300">
            {data.preview.verifiedPrompt}
          </pre>
        </details> : null}
      </section>}

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <h2 className="font-semibold text-white">File de traitement et historique du Registry</h2>
        <p className="mt-1 text-xs text-slate-500">
          Observation uniquement · les versions historiques ne sont pas évaluées par le vérificateur V3.1.
        </p>
        <nav aria-label="Filtrer les runs" className="mt-4 flex flex-wrap gap-2">
          {[
            ["v31", "V3.1 à traiter"],
            ["blocked", "Tous les blocages"],
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
              className={"block rounded-lg border p-4 text-sm transition hover:border-cyan-700 " +
                (run.runId === data.selectedRunId ? "border-cyan-800 bg-slate-800" : "border-slate-800 bg-slate-950/40")}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="break-all font-mono text-xs text-cyan-200">{run.runId}</span>
                <span className="font-medium text-slate-200">V{run.processVersion} · {run.status}</span>
              </div>
              <p className="mt-2 text-xs text-slate-400">
                {run.stage} / {run.stageLifecycle} · {run.mode} · cutoff {run.cutoff} · état #{run.stateVersion}
              </p>
              {run.contractStatusCode && <p className="mt-2 break-all font-mono text-xs text-amber-200">
                {run.contractStatusCode}
              </p>}
              {run.blockers.length > 0 && <p className="mt-2 line-clamp-2 text-xs text-amber-100">
                {run.blockers.length} blocage(s) documenté(s) · {run.blockers[0].text}
              </p>}
            </Link>)}
          </div>}
      </section>
    </>}
  </main>;
}
