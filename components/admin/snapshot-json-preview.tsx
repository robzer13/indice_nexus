'use client';

import { useMemo, useState } from 'react';

export function SnapshotJsonPreview() {
  const [raw, setRaw] = useState('');
  const preview = useMemo(() => {
    if (!raw.trim()) return { ok: false, message: 'Collez un objet JSON OroTitan.' };
    try {
      const value: unknown = JSON.parse(raw);
      if (!value || typeof value !== 'object' || Array.isArray(value)) return { ok: false, message: 'Le JSON doit être un objet.' };
      const object = value as Record<string, unknown>;
      const required = ['analysis_date', 'model_version', 'status', 'source_title'];
      const missing = required.filter((key) => !object[key]);
      if (missing.length) return { ok: false, message: `Champs requis manquants : ${missing.join(', ')}` };
      return {
        ok: true,
        message: `Prêt à insérer · ${Object.keys(object).length} champs · modèle ${String(object.model_version)}`,
      };
    } catch {
      return { ok: false, message: 'JSON invalide.' };
    }
  }, [raw]);

  return <div className="space-y-3">
    <textarea name="snapshot_json" value={raw} onChange={(event) => setRaw(event.target.value)} rows={16} placeholder='{"analysis_date":"2026-09-05","model_version":"OROTITAN-DEEP-2026-09","status":"FINALIST","source_title":"...","orotitan_score":88,"price_o90":100,"score_components":{}}' className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 font-mono text-xs text-slate-200 outline-none focus:border-cyan-500"/>
    <div className={`rounded-lg border px-3 py-2 text-sm ${preview.ok ? 'border-emerald-900 bg-emerald-950/25 text-emerald-200' : 'border-slate-800 bg-slate-950/60 text-slate-500'}`}>{preview.message}</div>
  </div>;
}
