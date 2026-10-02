import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';

export const metadata: Metadata = {
  title: { default: 'OroTitan', template: '%s · OroTitan' },
  description: "OroTitan · recherche actions, analyses versionnées et screener.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="fr">
      <body>
        <div className="min-h-screen">
          <header className="sticky top-0 z-40 border-b border-slate-800/90 bg-slate-950/90 backdrop-blur">
            <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
              <Link href="/" className="text-sm font-bold uppercase tracking-[0.22em] text-cyan-300">
                OroTitan
              </Link>
              <nav className="flex items-center gap-1 text-sm">
                <Link className="rounded-md px-3 py-2 text-slate-300 hover:bg-slate-900 hover:text-white" href="/orotitan">
                  Equity Research
                </Link>
                <Link className="rounded-md px-3 py-2 text-slate-400 hover:bg-slate-900 hover:text-slate-100" href="/screener">
                  Screener
                </Link>
                <Link className="rounded-md px-3 py-2 text-slate-600 hover:bg-slate-900 hover:text-slate-300" href="/admin">
                  Admin
                </Link>
              </nav>
            </div>
          </header>
          <main className="mx-auto max-w-7xl px-4 py-7 sm:px-6 lg:px-8">{children}</main>
          <footer className="mx-auto max-w-7xl px-4 pb-8 text-xs text-slate-700 sm:px-6 lg:px-8">
            OroTitan · recherche versionnée et traçable.
          </footer>
        </div>
      </body>
    </html>
  );
}
