"use client";

import { useEffect, useState } from "react";

type UploadEntry = {
  filename: string;
  sha256: string;
  sizeBytes: number | null;
  stage: string;
  status: "RECEIVED_NOT_VERIFIED";
};
type Payload = {
  ok?: boolean;
  enabled?: boolean;
  files?: UploadEntry[];
  code?: string;
  status?: string;
  duplicate?: boolean;
};

const ERROR: Record<string, string> = {
  UNAUTHORIZED: "Votre session a expiré. Reconnectez-vous à l’administration.",
  INTAKE_DISABLED: "Le dépôt de fichiers n’est pas encore activé.",
  INTAKE_ORIGIN_REJECTED: "Origine de la demande refusée.",
  INTAKE_RUN_NOT_EDITABLE: "Ce run est publié, annulé ou fermé aux nouveaux dépôts.",
  INTAKE_IDENTITY_INVALID: "L’identifiant du run est invalide.",
  INTAKE_RUN_NOT_FOUND: "Run introuvable dans le Registry.",
  INTAKE_FILE_SIZE_INVALID: "Fichier vide ou dépassant 1 Mo.",
  INTAKE_FILETYPE_NOT_ALLOWED: "Formats admis : .md, .txt, .json ou .csv.",
  INTAKE_FILENAME_INVALID: "Nom de fichier invalide.",
  INTAKE_JSON_INVALID: "Le fichier JSON doit contenir un objet JSON valide.",
  INTAKE_NOT_UTF8_TEXT: "Le fichier doit être du texte UTF-8.",
  INTAKE_LIMIT_REACHED: "Le maximum de fichiers reçus pour cette étape est atteint.",
  INTAKE_STORAGE_UPLOAD_FAILED: "Le stockage privé n’a pas confirmé la réception.",
  INTAKE_LIST_UNAVAILABLE: "Impossible de consulter les fichiers reçus.",
  INTAKE_REQUEST_TOO_LARGE: "Le fichier est trop volumineux.",
  INTAKE_UNAVAILABLE: "Service de dépôt momentanément indisponible.",
};

export function OroTitanManualIntake({ runId }: { runId: string }) {
  const [enabled, setEnabled] = useState(false);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [entries, setEntries] = useState<UploadEntry[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      setLoading(true);
      setEntries([]);
      setEnabled(false);
      setError("");
      try {
        const response = await fetch("/api/orotitan/manual-intake?run=" + encodeURIComponent(runId), {
          credentials: "same-origin", cache: "no-store", signal: controller.signal,
        });
        const result = (await response.json()) as Payload;
        if (!response.ok || !result.ok) throw new Error(result.code ?? "INTAKE_UNAVAILABLE");
        if (!controller.signal.aborted) {
          setEnabled(result.enabled === true);
          setEntries(result.files ?? []);
        }
      } catch (err) {
        if (!controller.signal.aborted) setError(ERROR[(err as Error).message] ?? ERROR.INTAKE_UNAVAILABLE);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => controller.abort();
  }, [runId]);

  async function submit() {
    if (!file || !enabled || submitting || file.size === 0 || file.size > 1_000_000) {
      setError("Sélectionnez un fichier texte de 1 Mo maximum.");
      return;
    }
    setSubmitting(true);
    setError("");
    setNotice("");
    try {
      const body = new FormData();
      body.set("run", runId);
      body.set("file", file);
      const response = await fetch("/api/orotitan/manual-intake", {
        method: "POST", credentials: "same-origin", cache: "no-store", body,
      });
      const result = (await response.json()) as Payload;
      if (!response.ok || !result.ok) throw new Error(result.code ?? "INTAKE_UNAVAILABLE");
      setNotice(result.duplicate === true ?
        "Ce fichier était déjà reçu. Il reste non vérifié." :
        "Fichier reçu dans l’espace privé de dépôt. Vérification et admission encore nécessaires.");
      setFile(null);
      const input = document.getElementById("orotitan-intake-file") as HTMLInputElement | null;
      if (input) input.value = "";
      const listing = await fetch("/api/orotitan/manual-intake?run=" + encodeURIComponent(runId), {
        cache: "no-store", credentials: "same-origin",
      });
      const listed = (await listing.json()) as Payload;
      if (!listing.ok || !listed.ok) throw new Error(listed.code ?? "INTAKE_LIST_UNAVAILABLE");
      setEntries(listed.files ?? []);
    } catch (err) {
      setError(ERROR[(err as Error).message] ?? ERROR.INTAKE_UNAVAILABLE);
    } finally { setSubmitting(false); }
  }

  return <section id="orotitan-intake" aria-label="Retour des livrables ChatGPT"
    className="space-y-5 rounded-2xl border border-slate-700/80 bg-slate-900/75 p-5 sm:p-7">
    <header className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[.2em] text-cyan-300">
          Retour d&apos;analyse · Étape manuelle
        </p>
        <h2 className="mt-2 text-xl font-semibold tracking-tight text-white sm:text-2xl">
          Déposer les livrables ChatGPT
        </h2>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
          Déposez les fichiers produits pour ce run. Ils restent séparés des artefacts
          officiels tant que les contrôles et le circuit d&apos;enregistrement ne sont pas terminés.
        </p>
      </div>
      <span className={"rounded-full border px-3 py-1.5 text-xs font-medium " +
        (enabled ? "border-cyan-800 bg-cyan-950/50 text-cyan-200" :
          "border-slate-700 bg-slate-800/70 text-slate-300")}>
        {loading ? "Consultation…" : enabled ? "Dépôt disponible" : "Dépôt non activé"}
      </span>
    </header>

    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="space-y-3 rounded-xl border border-slate-800 bg-slate-950/50 p-4">
        <h3 className="text-sm font-semibold text-white">Ajouter un document</h3>
        <label htmlFor="orotitan-intake-file" className="block text-xs leading-5 text-slate-400">
          Un fichier à la fois · JSON, Markdown, texte ou CSV · 1 Mo maximum
        </label>
        <input id="orotitan-intake-file" type="file" accept=".json,.md,.txt,.csv"
          disabled={!enabled || submitting || loading}
          onChange={event => { setFile(event.target.files?.[0] ?? null); setError(""); }}
          className="block w-full min-w-0 rounded-xl border border-dashed border-slate-700 bg-slate-900 p-3 text-sm text-slate-200 file:mr-3 file:rounded-lg file:border-0 file:bg-slate-700 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 disabled:opacity-60" />
        <button type="button" onClick={() => void submit()}
          disabled={!enabled || !file || submitting || loading}
          className="min-h-11 w-full rounded-xl bg-cyan-400 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300 disabled:cursor-not-allowed disabled:opacity-40">
          {submitting ? "Réception en cours…" : "Déposer sans valider l’étape"}
        </button>
        {!enabled && !loading ? <p className="text-xs leading-5 text-amber-200">
          La réception en stockage privé nécessite une activation serveur distincte,
          après validation de la mise en production. Aucun dépôt automatique.
        </p> : null}
        {notice ? <p role="status" className="rounded-lg border border-cyan-900 bg-cyan-950/30 p-3 text-xs leading-5 text-cyan-100">
          {notice}
        </p> : null}
        {error ? <p role="alert" className="rounded-lg border border-rose-900 bg-rose-950/30 p-3 text-xs leading-5 text-rose-200">
          {error}
        </p> : null}
      </div>

      <div className="space-y-3 rounded-xl border border-slate-800 bg-slate-950/50 p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="text-sm font-semibold text-white">Documents reçus</h3>
          <span className="text-xs tabular-nums text-slate-400">
            {loading ? "Lecture…" : `${entries.length} fichier(s)`}
          </span>
        </div>
        {entries.length === 0 ? <p className="text-sm leading-6 text-slate-400">
          Aucun fichier en attente de contrôle pour l&apos;étape courante.
        </p> : <ul className="max-h-72 space-y-2 overflow-y-auto">
          {entries.map(entry => <li key={entry.sha256 + entry.filename}
            className="rounded-lg border border-slate-800 bg-slate-900/75 p-3">
            <p className="break-all text-sm font-medium text-slate-100">{entry.filename}</p>
            <p className="mt-1 text-xs text-amber-200">Reçu · non vérifié</p>
            <p className="mt-1 font-mono text-[11px] text-slate-500">
              SHA-256 : {entry.sha256.slice(0, 16)}… · {entry.stage}
              {entry.sizeBytes !== null ? ` · ${entry.sizeBytes} octets` : ""}
            </p>
          </li>)}
        </ul>}
        <p className="border-t border-slate-800 pt-3 text-xs leading-5 text-slate-400">
          Étape suivante : vérification des preuves, des contrats et des manifestes,
          puis enregistrement par le processus Registry autorisé. La réception
          seule ne change ni l&apos;étape ni la barre de progression.
        </p>
      </div>
    </div>
  </section>;
}
