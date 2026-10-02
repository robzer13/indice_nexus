'use client';

import { useMemo, useState } from 'react';
import { stageLabel } from '@/lib/orotitan-ui/presentation';
import type { ArtifactMeta, ArtifactRef } from '@/lib/orotitan-ui/types';

export function ArtifactRegistry({
  refs,
  catalog,
  searchable = false,
}: {
  refs: ArtifactRef[];
  catalog: Record<string, ArtifactMeta>;
  searchable?: boolean;
}) {
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState<ArtifactRef | null>(null);

  const visibleRefs = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return refs;
    return refs.filter((ref) => {
      const metadata = catalog[ref.artifact_id];
      return [
        metadata?.logicalName,
        metadata?.artifactType,
        ref.artifact_id,
      ].some((value) => value?.toLowerCase().includes(normalized));
    });
  }, [catalog, query, refs]);

  return (
    <>
      {searchable ? (
        <div className="mb-3 flex justify-end">
          <label className="flex w-full max-w-sm items-center gap-2 rounded-lg border border-[rgba(123,173,214,.16)] bg-[#07111d]/70 px-3 py-2">
            <span aria-hidden="true" className="text-cyan-300">⌕</span>
            <span className="sr-only">Rechercher un document</span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Rechercher un document…"
              className="min-w-0 flex-1 bg-transparent text-xs text-slate-300 outline-none placeholder:text-slate-700"
            />
          </label>
        </div>
      ) : null}

      <div className="overflow-hidden rounded-[10px] border border-[rgba(123,173,214,.16)] bg-[rgba(8,22,36,.58)]">
        <div className="hidden grid-cols-[minmax(0,1fr)_120px_120px_70px] gap-4 border-b border-[rgba(123,173,214,.14)] bg-[#0a1624]/55 px-4 py-2.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600 sm:grid">
          <div>Document</div>
          <div>Étape</div>
          <div>Autorité</div>
          <div>Version</div>
        </div>

        {visibleRefs.length === 0 ? (
          <div className="px-4 py-8 text-center text-sm text-slate-600">Aucun document correspondant.</div>
        ) : (
          visibleRefs.map((ref) => {
            const metadata = catalog[ref.artifact_id];
            const authority = metadata?.authorityState === 'AUTHORITATIVE' ? 'Autoritatif' : 'Checkpoint';
            return (
              <button
                key={ref.artifact_id + ':' + ref.version}
                type="button"
                onClick={() => setSelected(ref)}
                className="grid w-full gap-2 border-b border-[rgba(123,173,214,.10)] px-4 py-3 text-left text-sm transition last:border-b-0 hover:bg-white/[.025] sm:grid-cols-[minmax(0,1fr)_120px_120px_70px] sm:items-center sm:gap-4"
              >
                <div className="min-w-0">
                  <div className="truncate text-slate-200">{metadata?.logicalName ?? ref.artifact_id}</div>
                  <div className="mt-0.5 truncate font-mono text-[10px] text-slate-700">{metadata?.artifactType ?? 'ARTIFACT'}</div>
                </div>
                <div className="text-xs text-slate-500">{metadata ? stageLabel(metadata.stageCode) : '—'}</div>
                <div className={'text-xs ' + (metadata?.authorityState === 'AUTHORITATIVE' ? 'text-emerald-400' : 'text-amber-300')}>
                  {authority}
                </div>
                <div className="font-mono text-xs text-slate-600">v{ref.version}</div>
              </button>
            );
          })
        )}
      </div>

      {selected ? (
        <ArtifactDrawer refValue={selected} metadata={catalog[selected.artifact_id]} onClose={() => setSelected(null)} />
      ) : null}
    </>
  );
}

function ArtifactDrawer({
  refValue,
  metadata,
  onClose,
}: {
  refValue: ArtifactRef;
  metadata: ArtifactMeta | undefined;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-[70]">
      <button type="button" aria-label="Fermer le détail du document" onClick={onClose} className="absolute inset-0 bg-black/60 backdrop-blur-[2px]" />
      <aside className="absolute bottom-0 right-0 top-0 w-full max-w-[470px] overflow-y-auto border-l border-[rgba(123,190,235,.24)] bg-[#07111d] p-6 shadow-[-18px_0_55px_rgba(0,0,0,.34)]">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-400">Document</div>
            <h2 className="mt-2 text-xl font-semibold text-white">{metadata?.logicalName ?? refValue.artifact_id}</h2>
            <div className="mt-2 font-mono text-[10px] text-slate-600">{metadata?.artifactType ?? 'ARTIFACT'}</div>
          </div>
          <button type="button" onClick={onClose} className="rounded-md border border-slate-800 px-2.5 py-1.5 text-sm text-slate-500 hover:text-white">
            ×
          </button>
        </div>

        <dl className="mt-7 grid grid-cols-2 gap-4 border-y border-[rgba(123,173,214,.14)] py-5 text-xs">
          <Data label="Étape" value={metadata ? stageLabel(metadata.stageCode) : '—'} />
          <Data label="Autorité" value={metadata?.authorityState === 'AUTHORITATIVE' ? 'Autoritatif' : 'Checkpoint'} />
          <Data label="Version" value={'v' + refValue.version} />
          <Data label="État" value={metadata?.availabilityState ?? '—'} />
        </dl>

        <div className="mt-7">
          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600">Identité technique</div>
          <dl className="mt-4 space-y-5 text-xs">
            <Data label="Artifact ID" value={refValue.artifact_id} mono />
            <Data label="SHA-256" value={refValue.content_sha256 ?? '—'} mono />
            <Data label="Authority class" value={refValue.required_authority_class ?? '—'} mono />
            <Data label="Stockage" value={metadata?.storageBackend ?? '—'} mono />
          </dl>
        </div>
      </aside>
    </div>
  );
}

function Data({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt className="text-slate-600">{label}</dt>
      <dd className={'mt-1.5 break-all text-slate-300 ' + (mono ? 'font-mono text-[11px]' : '')}>{value}</dd>
    </div>
  );
}
