'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

type NavItem = { href: string; label: string; icon: (className: string) => React.ReactNode };

const navItems: NavItem[] = [
  { href: '/', label: 'Dashboard', icon: (className) => <HomeIcon className={className} /> },
  { href: '/screener', label: 'Screener', icon: (className) => <SearchIcon className={className} /> },
  { href: '/admin', label: 'Admin', icon: (className) => <SettingsIcon className={className} /> },
];

export function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const pathname = usePathname();

  return (
    <>
      <div aria-hidden="true" onClick={onClose} className={`fixed inset-0 z-40 bg-slate-950/70 backdrop-blur-sm transition-opacity lg:hidden ${open ? 'opacity-100' : 'pointer-events-none opacity-0'}`} />
      <aside aria-label="Navigation principale" className={`fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-slate-700/60 bg-cockpit-sidebar px-4 py-5 shadow-panel-soft transition-transform duration-200 lg:translate-x-0 ${open ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex items-start justify-between px-2">
          <Link href="/" onClick={onClose} className="group rounded-control focus-visible:outline-offset-4" aria-label="OroTitan, accueil">
            <span className="block text-base font-semibold uppercase tracking-[0.28em] text-ink-primary transition-colors group-hover:text-state-accent">OroTitan</span>
            <span className="mt-1 block text-[10px] uppercase tracking-[0.22em] text-ink-muted">Discipline builds value</span>
          </Link>
          <button type="button" onClick={onClose} className="rounded-control p-2 text-ink-muted hover:bg-cockpit-hover hover:text-ink-primary lg:hidden" aria-label="Fermer la navigation">
            <CloseIcon className="h-5 w-5" />
          </button>
        </div>

        <nav className="mt-10 space-y-1" aria-label="Sections de l’application">
          {navItems.map((item) => {
            const active = item.href === '/' ? pathname === '/' : pathname.startsWith(item.href);
            return (
              <Link key={item.href} href={item.href} onClick={onClose} aria-current={active ? 'page' : undefined} className={`group flex items-center gap-3 rounded-control border px-3 py-2.5 text-sm font-medium transition-colors duration-150 focus-visible:outline-offset-2 ${active ? 'border-state-accent/40 bg-cockpit-active text-ink-primary shadow-[inset_3px_0_0_#39d4df]' : 'border-transparent text-ink-secondary hover:border-slate-700/70 hover:bg-cockpit-hover hover:text-ink-primary'}`}>
                {item.icon(`h-5 w-5 shrink-0 ${active ? 'text-state-accent' : 'text-ink-muted transition-colors group-hover:text-ink-secondary'}`)}
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="mt-auto border-t border-slate-700/50 px-2 pt-5">
          <p className="font-serif text-sm italic leading-6 text-ink-secondary">« Les meilleures entreprises pour un monde plus riche. »</p>
          <div className="mt-5 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-ink-muted">
            <span>OroTitan</span>
            <span>V1.2</span>
          </div>
        </div>
      </aside>
    </>
  );
}

function Icon({ className, children }: { className: string; children: React.ReactNode }) {
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className={className}>{children}</svg>;
}

function HomeIcon({ className }: { className: string }) { return <Icon className={className}><path d="m3 10 9-7 9 7" /><path d="M5 9v11h14V9" /><path d="M9 20v-6h6v6" /></Icon>; }
function SearchIcon({ className }: { className: string }) { return <Icon className={className}><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4.5 4.5" /></Icon>; }
function SettingsIcon({ className }: { className: string }) { return <Icon className={className}><path d="M12 3v2" /><path d="M12 19v2" /><path d="m4.2 6.2 1.4 1.4" /><path d="m18.4 16.4 1.4 1.4" /><path d="M3 12h2" /><path d="M19 12h2" /><path d="m4.2 17.8 1.4-1.4" /><path d="m18.4 7.6 1.4-1.4" /><circle cx="12" cy="12" r="4" /></Icon>; }
function CloseIcon({ className }: { className: string }) { return <Icon className={className}><path d="m6 6 12 12" /><path d="M18 6 6 18" /></Icon>; }
