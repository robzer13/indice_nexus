'use client';

import Link from 'next/link';
import { usePathname, useSearchParams } from 'next/navigation';
import { buildRunHref, runStatusLabel, stageLabel } from '@/lib/orotitan-ui/presentation';
import type { RunSummary } from '@/lib/orotitan-ui/types';

export function IssuerNav({
  issuerSlug,
  runs,
}: {
  issuerSlug: string;
  runs: RunSummary[];
}) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const run = searchParams.get('run');
  const selectedRun = run ? runs.find((candidate) => candidate.runId === run) ?? null : null;
  const base = '/orotitan/' + issuerSlug;

  const links = [
    { href: buildRunHref(base, run), label: "Vue d'ensemble", path: base },
    { href: buildRunHref(base + '/documents', run), label: 'Documents', path: base + '/documents' },
    { href: buildRunHref(base + '/context', run), label: 'Contexte', path: base + '/context' },
  ];

  return (
    <>
      <nav className="mt-6 space-y-1 text-sm">
        {links.map((link) => {
          const active = pathname === link.path;
          return (
            <Link
              key={link.path}
              href={link.href}
              className={
                'block border-l-2 px-3 py-2 transition ' +
                (active
                  ? 'border-cyan-400 text-slate-100'
                  : 'border-transparent text-slate-500 hover:border-slate-700 hover:text-slate-300')
              }
            >
              {link.label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-6 border-t border-slate-800 pt-4">
        <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-700">Analyse</div>
        {selectedRun ? (
          <>
            <div className="mt-2 text-xs text-slate-400">
              {stageLabel(selectedRun.currentStage)} ·{' '}
              <span className={selectedRun.runStatus === 'BLOCKED' ? 'text-rose-300' : selectedRun.runStatus === 'ACTIVE' ? 'text-amber-300' : 'text-slate-400'}>
                {runStatusLabel(selectedRun.runStatus)}
              </span>
            </div>
            <div className="mt-2 text-[11px] text-slate-700">Lecture seule</div>
          </>
        ) : (
          <div className="mt-2 text-xs text-slate-700">Aucune analyse sélectionnée</div>
        )}
      </div>
    </>
  );
}
