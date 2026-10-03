'use client';

import { useEffect, useMemo, useState } from 'react';
import { stageLabel } from '@/lib/orotitan-ui/presentation';
import type {
  ArtifactMeta,
  ArtifactRef,
  VerifiedArtifactContent,
} from '@/lib/orotitan-ui/types';

type ContentState =
  | { status: 'loading' }
  | { status: 'ready'; content: VerifiedArtifactContent }
  | { status: 'error'; code: string; message: string };

export function ArtifactRegistry({
  refs,
  catalog,
  issuerSlug,
  runId,
  searchable = false,
}: {
  refs: ArtifactRef[];
  catalog: Record<string, ArtifactMeta>;
  issuerSlug: string;
  runId: string;
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
              className="min-w-0 flex-1 bg-transparent text-xs text-slate-300 outline-none placeholder:text-slate-600"
            />
          </label>
        </div>
      ) : null}

      <div className="overflow-hidden rounded-[10px] border border-[rgba(123,173,214,.16)] bg-[rgba(8,22,36,.58)]">
        <div className="hidden grid-cols-[minmax(0,1fr)_120px_120px_70px] gap-4 border-b border-[rgba(123,173,214,.18)] bg-[#0a1624]/75 px-4 py-2.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500 sm:grid">
          <div>Document</div>
          <div>Étape</div>
          <div>Autorité</div>
          <div>Version</div>
        </div>

        {visibleRefs.length === 0 ? (
          <div className="px-4 py-8 text-center text-sm text-slate-600">
            Aucun document correspondant.
          </div>
        ) : (
          visibleRefs.map((ref) => {
            const metadata = catalog[ref.artifact_id];
            const authority =
              metadata?.authorityState === 'AUTHORITATIVE'
                ? 'Autoritatif'
                : 'Checkpoint';
            return (
              <button
                key={ref.artifact_id + ':' + ref.version}
                type="button"
                onClick={() => setSelected(ref)}
                className="grid w-full gap-2 border-b border-[rgba(123,173,214,.10)] px-4 py-3 text-left text-sm transition last:border-b-0 hover:bg-cyan-400/[.035] sm:grid-cols-[minmax(0,1fr)_120px_120px_70px] sm:items-center sm:gap-4"
              >
                <div className="min-w-0">
                  <div className="truncate text-slate-200">
                    {metadata?.logicalName ?? ref.artifact_id}
                  </div>
                  <div className="mt-0.5 truncate font-mono text-[10px] text-slate-600">
                    {metadata?.artifactType ?? 'ARTIFACT'}
                  </div>
                </div>
                <div className="text-xs text-slate-400">
                  {metadata ? stageLabel(metadata.stageCode) : '—'}
                </div>
                <div
                  className={
                    'text-xs ' +
                    (metadata?.authorityState === 'AUTHORITATIVE'
                      ? 'text-emerald-400'
                      : 'text-amber-300')
                  }
                >
                  {authority}
                </div>
                <div className="font-mono text-xs text-slate-500">
                  v{ref.version}
                </div>
              </button>
            );
          })
        )}
      </div>

      {selected ? (
        <ArtifactDrawer
          key={selected.artifact_id + ':' + selected.version}
          refValue={selected}
          metadata={catalog[selected.artifact_id]}
          issuerSlug={issuerSlug}
          runId={runId}
          onClose={() => setSelected(null)}
        />
      ) : null}
    </>
  );
}

function ArtifactDrawer({
  refValue,
  metadata,
  issuerSlug,
  runId,
  onClose,
}: {
  refValue: ArtifactRef;
  metadata: ArtifactMeta | undefined;
  issuerSlug: string;
  runId: string;
  onClose: () => void;
}) {
  const [contentState, setContentState] = useState<ContentState>({
    status: 'loading',
  });

  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams({
      issuer: issuerSlug,
      run: runId,
      version: String(refValue.version),
    });

    fetch(
      '/api/orotitan/artifacts/' +
        encodeURIComponent(refValue.artifact_id) +
        '?' +
        params.toString(),
      {
        method: 'GET',
        cache: 'no-store',
        signal: controller.signal,
      },
    )
      .then(async (response) => {
        const body = (await response.json()) as
          | VerifiedArtifactContent
          | { error?: string; message?: string };
        if (!response.ok) {
          const errorBody = body as { error?: string; message?: string };
          throw {
            code: errorBody.error ?? 'ARTIFACT_RESOLUTION_FAILED',
            message:
              errorBody.message ?? 'Impossible de vérifier le contenu.',
          };
        }
        setContentState({
          status: 'ready',
          content: body as VerifiedArtifactContent,
        });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const value =
          typeof error === 'object' && error !== null
            ? (error as { code?: string; message?: string })
            : {};
        setContentState({
          status: 'error',
          code: value.code ?? 'ARTIFACT_RESOLUTION_FAILED',
          message: value.message ?? 'Impossible de vérifier le contenu.',
        });
      });

    return () => controller.abort();
  }, [issuerSlug, refValue.artifact_id, refValue.version, runId]);

  return (
    <div className="fixed inset-0 z-[70]">
      <button
        type="button"
        aria-label="Fermer le détail du document"
        onClick={onClose}
        className="absolute inset-0 bg-black/60 backdrop-blur-[2px]"
      />
      <aside className="absolute bottom-0 right-0 top-0 w-full max-w-[760px] overflow-y-auto border-l border-[rgba(123,190,235,.24)] bg-[#07111d] p-6 shadow-[-18px_0_55px_rgba(0,0,0,.34)]">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-400">
              Document
            </div>
            <h2 className="mt-2 text-xl font-semibold text-white">
              {metadata?.logicalName ?? refValue.artifact_id}
            </h2>
            <div className="mt-2 font-mono text-[10px] text-slate-600">
              {metadata?.artifactType ?? 'ARTIFACT'}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-slate-800 px-2.5 py-1.5 text-sm text-slate-500 hover:text-white"
          >
            ×
          </button>
        </div>

        <dl className="mt-7 grid grid-cols-2 gap-4 border-y border-[rgba(123,173,214,.14)] py-5 text-xs">
          <Data
            label="Étape"
            value={metadata ? stageLabel(metadata.stageCode) : '—'}
          />
          <Data
            label="Autorité"
            value={
              metadata?.authorityState === 'AUTHORITATIVE'
                ? 'Autoritatif'
                : 'Checkpoint'
            }
          />
          <Data label="Version" value={'v' + refValue.version} />
          <Data label="État" value={metadata?.availabilityState ?? '—'} />
        </dl>

        <div className="mt-7">
          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600">
            Identité technique
          </div>
          <dl className="mt-4 space-y-5 text-xs">
            <Data label="Artifact ID" value={refValue.artifact_id} mono />
            <Data
              label="SHA-256"
              value={refValue.content_sha256 ?? '—'}
              mono
            />
            <Data
              label="Authority class"
              value={refValue.required_authority_class ?? '—'}
              mono
            />
            <Data
              label="Stockage"
              value={metadata?.storageBackend ?? '—'}
              mono
            />
          </dl>
        </div>

        <div className="mt-8 border-t border-[rgba(123,173,214,.14)] pt-6">
          <div className="flex items-center justify-between gap-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600">
              Contenu privé vérifié
            </div>
            {contentState.status === 'ready' ? (
              <span className="rounded-full border border-emerald-400/25 bg-emerald-400/[.06] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-emerald-300">
                Intégrité vérifiée
              </span>
            ) : null}
          </div>

          {contentState.status === 'loading' ? (
            <div className="mt-4 rounded-lg border border-slate-800 bg-[#050c14] px-4 py-5 text-sm text-slate-500">
              Résolution exacte et vérification des bytes…
            </div>
          ) : null}

          {contentState.status === 'error' ? (
            <ArtifactReadError
              code={contentState.code}
              message={contentState.message}
            />
          ) : null}

          {contentState.status === 'ready' ? (
            <VerifiedContent content={contentState.content} />
          ) : null}
        </div>
      </aside>
    </div>
  );
}

function ArtifactReadError({
  code,
  message,
}: {
  code: string;
  message: string;
}) {
  const unauthorized = code === 'UNAUTHORIZED';
  return (
    <div className="mt-4 rounded-lg border border-amber-400/20 bg-amber-400/[.04] px-4 py-4">
      <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-amber-300">
        {code}
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-400">{message}</p>
      {unauthorized ? (
        <a
          href="/admin"
          className="mt-3 inline-flex rounded-md border border-cyan-400/20 px-3 py-2 text-xs font-medium text-cyan-300 hover:bg-cyan-400/[.04]"
        >
          Ouvrir la connexion admin
        </a>
      ) : null}
    </div>
  );
}

function VerifiedContent({
  content,
}: {
  content: VerifiedArtifactContent;
}) {
  return (
    <div className="mt-4">
      <dl className="grid gap-3 rounded-lg border border-emerald-400/15 bg-emerald-400/[.025] p-4 text-xs sm:grid-cols-2">
        <Data label="Media type" value={content.mediaType} mono />
        <Data
          label="Taille vérifiée"
          value={content.sizeBytes.toLocaleString('fr-FR') + ' octets'}
        />
        <Data
          label="Manifest actif"
          value={
            content.verification.manifestMembershipVerified ? 'Oui' : 'Non'
          }
        />
        <Data
          label="SHA-256 vérifié"
          value={content.verification.sha256Verified ? 'Oui' : 'Non'}
        />
        <Data
          label="Git blob vérifié"
          value={
            content.verification.gitBlobVerified === null
              ? 'N/A'
              : content.verification.gitBlobVerified
                ? 'Oui'
                : 'Non'
          }
        />
      </dl>

      {content.previewKind === 'TEXT' && content.previewText !== null ? (
        <pre className="mt-4 max-h-[52vh] overflow-auto whitespace-pre-wrap break-words rounded-lg border border-[rgba(123,173,214,.14)] bg-[#040a11] p-4 font-mono text-[11px] leading-5 text-slate-300">
          {content.previewText}
        </pre>
      ) : (
        <div className="mt-4 rounded-lg border border-slate-800 bg-[#050c14] px-4 py-5 text-sm text-slate-500">
          {content.previewReason ?? 'Aucun aperçu inline disponible.'}
        </div>
      )}
    </div>
  );
}

function Data({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <dt className="text-slate-500">{label}</dt>
      <dd
        className={
          'mt-1.5 break-all text-slate-300 ' +
          (mono ? 'font-mono text-[11px]' : '')
        }
      >
        {value}
      </dd>
    </div>
  );
}
