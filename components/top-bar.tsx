'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export function TopBar({ onMenuClick }: { onMenuClick: () => void }) {
  const pathname = usePathname();
  const currentSection = pathname.startsWith('/admin') ? 'Admin' : pathname.startsWith('/screener') ? 'Screener' : pathname.startsWith('/company/') ? 'Société' : 'Dashboard';

  return (
    <header className="sticky top-0 z-30 border-b border-slate-700/60 bg-cockpit-bg/90 backdrop-blur-xl">
      <div className="flex min-h-16 items-center gap-3 px-4 sm:px-6 xl:px-8">
        <button type="button" onClick={onMenuClick} className="rounded-control border border-slate-700/60 p-2 text-ink-secondary hover:border-state-accent/40 hover:bg-cockpit-hover hover:text-ink-primary lg:hidden" aria-label="Ouvrir la navigation">
          <MenuIcon className="h-5 w-5" />
        </button>
        <div className="hidden min-w-0 items-center gap-2 text-sm text-ink-muted sm:flex">
          <span className="text-ink-secondary">OroTitan</span>
          <span aria-hidden="true" className="text-slate-600">/</span>
          <span className="truncate text-ink-primary">{currentSection}</span>
        </div>
        <Link href="/screener" className="group flex min-w-0 flex-1 items-center gap-3 rounded-control border border-slate-700/70 bg-cockpit-panel/70 px-3 py-2 text-sm text-ink-muted transition-colors hover:border-state-accent/50 hover:bg-cockpit-hover hover:text-ink-secondary sm:mx-auto sm:max-w-xl" aria-label="Rechercher une société dans le screener">
          <SearchIcon className="h-4 w-4 shrink-0 text-ink-muted group-hover:text-state-accent" />
          <span className="truncate">Rechercher une société, un ticker ou un secteur…</span>
          <kbd className="ml-auto hidden rounded border border-slate-700 bg-slate-950/70 px-1.5 py-0.5 font-mono text-[10px] text-ink-muted md:inline">⌘ K</kbd>
        </Link>
        <div className="hidden items-center gap-3 text-right sm:flex">
          <div><div className="text-[10px] uppercase tracking-[0.18em] text-ink-muted">Version</div><div className="font-mono text-xs text-state-accent">V1.2</div></div>
          <Link href="/admin" className="rounded-control border border-slate-700/60 px-3 py-2 text-xs font-medium text-ink-secondary hover:border-state-accent/40 hover:bg-cockpit-hover hover:text-ink-primary">Administration</Link>
        </div>
      </div>
    </header>
  );
}

function Icon({ className, children }: { className: string; children: React.ReactNode }) {
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className={className}>{children}</svg>;
}

function SearchIcon({ className }: { className: string }) { return <Icon className={className}><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4.5 4.5" /></Icon>; }
function MenuIcon({ className }: { className: string }) { return <Icon className={className}><path d="M4 7h16" /><path d="M4 12h16" /><path d="M4 17h16" /></Icon>; }
