'use client';

import Link from 'next/link';
import { usePathname, useSearchParams } from 'next/navigation';
import { buildRunHref, deriveStageStates, lifecycleLabel, runStatusLabel, stageLabel } from '@/lib/orotitan-ui/presentation';
import type { RunSummary, StageLifecycle } from '@/lib/orotitan-ui/types';

const dotClasses: Record<StageLifecycle, string> = {
  NOT_STARTED: 'border-slate-600 bg-[#07111d]',
  IN_PROGRESS: 'border-cyan-300 bg-cyan-400',
  PAUSED: 'border-amber-400 bg-amber-400',
  BLOCKED: 'border-rose-400 bg-rose-400',
  COMPLETE: 'border-emerald-300 bg-emerald-400',
};

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

  const stageStates = selectedRun
    ? deriveStageStates(selectedRun.currentStage, selectedRun.runStatus === 'BLOCKED' ? 'BLOCKED' : 'IN_PROGRESS')
    : null;

  return (
    <>
      <nav className="mt-7 grid grid-cols-3 gap-2 text-sm lg:block lg:space-y-1">
        {links.map((link) => {
          const active = pathname === link.path;
          return (
            <Link
              key={link.path}
              href={link.href}
              className={
                'relative block rounded-lg border px-3 py-2.5 transition lg:border-transparent ' +
                (active
                  ? 'border-cyan-400/25 bg-cyan-400/10 text-cyan-100 lg:border-l-cyan-400 lg:bg-[rgba(43,200,255,.07)]'
                  : 'border-[rgba(123,173,214,.12)] text-slate-500 hover:bg-slate-900/60 hover:text-slate-300 lg:border-l-transparent')
              }
            >
              {link.label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-7 border-t border-[rgba(123,173,214,.14)] pt-5">
        <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-700">Analyse</div>
        {selectedRun && stageStates ? (
          <div className="mt-4 grid grid-cols-3 gap-3 lg:block lg:space-y-0">
            {stageStates.map(({ stage, lifecycle }, index) => (
              <div key={stage} className="relative flex gap-3 pb-4 lg:pb-5">
                <div className="relative flex w-4 shrink-0 justify-center">
                  <span className={'mt-1.5 h-2.5 w-2.5 rounded-full border ' + dotClasses[lifecycle]} />
                  {index < stageStates.length - 1 ? <span className="absolute bottom-0 top-4 hidden w-px bg-slate-800 lg:block" /> : null}
                </div>
                <div>
                  <div className="text-xs font-medium text-slate-300">{stageLabel(stage)}</div>
                  <div className={'mt-1 text-[11px] ' + (lifecycle === 'BLOCKED' ? 'text-rose-300' : lifecycle === 'COMPLETE' ? 'text-emerald-400' : 'text-slate-600')}>
                    {lifecycleLabel(lifecycle)}
                  </div>
                </div>
              </div>
            ))}
            <div className="hidden text-[10px] text-slate-700 lg:block">
              {runStatusLabel(selectedRun.runStatus)} · lecture seule
            </div>
          </div>
        ) : (
          <div className="mt-3 text-xs text-slate-700">Aucune analyse sélectionnée</div>
        )}
      </div>
    </>
  );
}
