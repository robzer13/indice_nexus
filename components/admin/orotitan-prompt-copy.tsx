"use client";

import { useState } from "react";

export function OroTitanPromptCopy({ prompt, phase }: { prompt: string; phase: string }) {
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "failed">("idle");

  async function copyPrompt() {
    setCopyStatus("idle");
    try {
      await navigator.clipboard.writeText(prompt);
      setCopyStatus("copied");
    } catch {
      setCopyStatus("failed");
    }
  }

  return <div className="space-y-3 border-t border-slate-800 pt-4">
    <p className="text-sm text-slate-300">
      Phase proposée : <strong className="text-white">{phase}</strong>.
      Ce prompt est destiné à être copié et exécuté manuellement dans ChatGPT.
      Ce bouton ne lance aucune IA et ne modifie aucune étape du Registry.
    </p>
    <button type="button" onClick={copyPrompt}
      className="rounded-lg bg-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-300">
      Copier le prompt {phase}
    </button>
    <span role="status" aria-live="polite" className="ml-3 text-sm text-slate-300">
      {copyStatus === "copied" ? "Prompt copié dans le presse-papiers." :
        copyStatus === "failed" ? "Copie indisponible. Ouvrir le prompt ci-dessous et le sélectionner manuellement." : ""}
    </span>
    <details>
      <summary className="cursor-pointer text-sm font-medium text-cyan-300">
        Afficher le prompt complet
      </summary>
      <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-slate-950 p-4 text-xs text-slate-300">
        {prompt}
      </pre>
    </details>
    <p className="text-xs text-slate-400">
      Après l&apos;analyse, les résultats doivent être vérifiés et enregistrés en artefacts
      avant qu'un nouveau handoff puisse être admis. Aucun changement d'état automatique.
    </p>
  </div>;
}
