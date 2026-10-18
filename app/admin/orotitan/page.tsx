import Link from "next/link";
import { redirect } from "next/navigation";
import type { Metadata } from "next";

import { AdminNav } from "@/components/admin/admin-nav";
import { isAdminAuthenticated } from "@/lib/auth/admin-session";
import { loadAdminPilotageCockpit } from "@/lib/orotitan-equity/v3/pilotage-operator-cockpit-server";

export const metadata: Metadata = { title: "Pilotage OroTitan V3.1 — diagnostic" };
export const dynamic = "force-dynamic";

const explanations: Record<string, string> = {
  ENVIRONMENT_OBSERVATION_TOKEN_MISSING: "Vérification approfondie indisponible : configurer le jeton de lecture Management API Supabase côté serveur.",
  ENVIRONMENT_OBSERVATION_FETCH_FAILED: "Impossible de confirmer l’environnement de production ; aucune preuve n’est admise.",
  ENVIRONMENT_OBSERVATION_HTTP_REJECTED: "Le jeton Management API ne permet pas de confirmer ce projet.",
  ENVIRONMENT_OBSERVATION_NOT_ADMITTED: "L’état, le nom ou la région du projet ne correspondent pas aux autorités épinglées.",
  RUN_NOT_FOUND: "Run absent du Registry.",
  CURRENT_STAGE_MISSING: "L’étape courante déclarée par le run est absente du Registry.",
  D5_FULL_PRECISION_INPUT_NOT_VERIFIED: "Artefact full-precision requis non vérifié.",
  D6_PROPOSED_SCHEMA_INVALID: "Format de provenance D6 invalide ou non conforme.",
  V31_PUBLICATION_WRITER_NOT_ADMITTED: "Le writer V3.1 n’est pas homologué ; publication interdite.",
  REGISTRY_LIST_QUERY_FAILED: "Impossible de charger le Registry ; vérifier la connexion Supabase côté serveur.",
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
  let data: Awaited<ReturnType<typeof loadAdminPilotageCockpit>> | null = null;
  try { data = await loadAdminPilotageCockpit(selected); }
  catch { /* No internal exception details or secrets in the administrator UI. */ }

  return <main className="space-y-6">
    <header className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-[.2em] text-cyan-400">OroTitan · Pilotage V3.1</p>
      <h1 className="text-3xl font-semibold text-white">Diagnostic des runs</h1>
      <p className="max-w-3xl text-sm leading-6 text-slate-400">
        État réel du Registry et vérification des preuves immuables. Lecture seule :
        aucune création de run, exécution de stage, validation I3-B ou publication.
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

    {!data ? <section role="status" className="rounded-xl border border-rose-900 bg-rose-950/20 p-4 text-sm text-rose-200">
      Le diagnostic n’est pas disponible. Vérifier la configuration serveur et la connexion Registry.
    </section> : <>
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
        <h2 className="font-semibold text-white">30 derniers runs du Registry</h2>
        <p className="mt-1 text-xs text-slate-500">Observation uniquement · sélectionner un run pour vérifier les preuves.</p>
        {data.runs.length === 0 ? <p className="mt-4 text-sm text-slate-400">Aucun run trouvé dans la fenêtre consultée.</p> :
          <div className="mt-4 space-y-2">
            {data.runs.map(run => <Link key={run.runId}
              href={"/admin/orotitan?run=" + encodeURIComponent(run.runId)}
              className={"block rounded-lg border p-3 text-sm transition hover:border-cyan-700 " +
                (run.runId === data.selectedRunId ? "border-cyan-800 bg-slate-800" : "border-slate-800 bg-slate-950/40")}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="break-all font-mono text-xs text-cyan-200">{run.runId}</span>
                <span className="font-medium text-slate-200">{run.status}</span>
              </div>
              <p className="mt-1 text-xs text-slate-400">
                {run.stage} · {run.mode} · cutoff {run.cutoff} · état #{run.stateVersion}
              </p>
            </Link>)}
          </div>}
      </section>
    </>}
  </main>;
}
