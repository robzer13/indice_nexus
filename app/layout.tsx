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
          <header className="sticky top-0 z-40 border-b border-[rgba(123,173,214,.14)] bg-[#030a12]/95 backdrop-blur-xl">
            <div className="mx-auto flex h-[58px] max-w-[1600px] items-center gap-4 px-4 sm:px-6 lg:px-8">
              <Link href="/" className="flex shrink-0 items-center gap-3">
                <span className="relative h-7 w-7 rounded-full border border-cyan-400/50 bg-cyan-400/10">
                  <span className="absolute inset-[5px] rounded-full bg-gradient-to-br from-cyan-300 to-blue-600" />
                </span>
                <span className="text-sm font-semibold uppercase tracking-[0.24em] text-slate-100">OroTitan</span>
              </Link>

              <div className="hidden h-5 w-px bg-slate-800 sm:block" />
              <span className="hidden shrink-0 text-sm text-slate-400 sm:block">Recherche actions</span>

              <Link
                href="/orotitan"
                className="mx-auto hidden h-9 w-full max-w-[580px] items-center gap-3 rounded-lg border border-[rgba(123,190,235,.22)] bg-[#07111d]/85 px-3 text-sm text-slate-500 transition hover:border-cyan-400/35 hover:text-slate-300 md:flex"
              >
                <span aria-hidden="true" className="text-base text-cyan-300">⌕</span>
                <span>Rechercher une société, un ticker…</span>
                <span className="ml-auto rounded border border-slate-700 px-1.5 py-0.5 text-[10px] text-slate-600">⌘ K</span>
              </Link>

              <nav className="ml-auto flex items-center gap-1 text-sm">
                <Link className="rounded-md px-3 py-2 text-slate-400 hover:bg-slate-900 hover:text-cyan-200" href="/screener">
                  Screener
                </Link>
                <Link className="hidden rounded-md px-3 py-2 text-slate-600 hover:bg-slate-900 hover:text-slate-300 sm:block" href="/admin">
                  Admin
                </Link>
              </nav>
            </div>
          </header>

          <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">{children}</main>

          <footer className="mx-auto max-w-[1600px] px-4 pb-8 text-[11px] text-slate-700 sm:px-6 lg:px-8">
            OroTitan · recherche versionnée et traçable.
          </footer>
        </div>
      </body>
    </html>
  );
}
