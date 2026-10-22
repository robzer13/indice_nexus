"use client";

import { useState } from "react";

const PHASE_LABELS: Record<string, string> = {
  RESEARCH: "Research",
  FUNDAMENTALS: "Fundamentals",
  VALUATION: "Valorisation",
  CERTIFICATION_RECONCILIATION: "Certification",
  INTEGRATION: "Intégration",
};

export function OroTitanPromptCopy({ prompt, phase }: { prompt: string; phase: string }) {
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "failed">("idle");
  const label = PHASE_LABELS[phase] ?? phase;

  async function copyPrompt() {
    setCopyStatus("idle");
    try {
      await navigator.clipboard.writeText(prompt);
      setCopyStatus("copied");
    } catch {
      setCopyStatus("failed");
    }
  }

  return <div className="mt-4 space-y-4">
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <div className="space-y-1">
        <p className="text-sm font-medium text-slate-100">Étape proposée : {label}</p>
        <p className="text-xs leading-5 text-slate-400">
          Copier, ouvrir ChatGPT, coller puis travailler manuellement.
        </p>
      </div>
      <button type="button" onClick={copyPrompt}
        className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-cyan-950/30 transition hover:bg-cyan-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300">
        <svg aria-hidden="true" width="17" height="17" viewBox="0 0 24 24"
          fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <rect x="8" y="8" width="12" height="12" rx="2" />
          <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
        </svg>
        {copyStatus === "copied" ? "Prompt copié" : `Copier le prompt ${label}`}
      </button>
    </div>
    <div role="status" aria-live="polite" className="min-h-5 text-xs text-slate-300">
      {copyStatus === "copied" ? "Copié. Vous pouvez maintenant le coller dans ChatGPT." :
        copyStatus === "failed" ?
          "Copie indisponible. Affichez le prompt ci-dessous et sélectionnez le texte." : ""}
    </div>
    <details className="group rounded-lg border border-slate-800 bg-slate-950/70">
      <summary className="cursor-pointer select-none px-4 py-3 text-sm font-medium text-cyan-300 transition hover:text-cyan-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300">
        Afficher le prompt complet et ses références
      </summary>
      <pre className="mx-3 mb-3 max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-slate-900/70 p-4 font-mono text-xs leading-6 text-slate-300">
        {prompt}
      </pre>
    </details>
    <p className="text-xs leading-5 text-slate-400">
      La copie ne lance aucune IA et ne modifie aucun statut. Les résultats devront
      être vérifiés et enregistrés par le circuit prévu avant le prochain handoff.
    </p>
  </div>;
}
