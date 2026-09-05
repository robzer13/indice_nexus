import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';

export const metadata: Metadata = {
  title: { default: 'OroTitan Screener', template: '%s · OroTitan' },
  description: "Cockpit d'une base fermée de sociétés déjà analysées.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="fr"><body><div className="min-h-screen"><header className="sticky top-0 z-40 border-b border-slate-800/90 bg-slate-950/90 backdrop-blur"><div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8"><Link href="/" className="flex items-baseline gap-2"><span className="text-sm font-bold uppercase tracking-[0.22em] text-cyan-300">OroTitan</span><span className="text-xs text-slate-500">Screener V1</span></Link><nav className="flex items-center gap-1 text-sm text-slate-400"><Link className="rounded-lg px-3 py-2 hover:bg-slate-900 hover:text-slate-100" href="/screener">Screener</Link><Link className="rounded-lg px-3 py-2 hover:bg-slate-900 hover:text-slate-100" href="/admin">Admin</Link></nav></div></header><main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">{children}</main><footer className="mx-auto max-w-7xl px-4 pb-8 text-xs text-slate-600 sm:px-6 lg:px-8">OroTitan Screener · analyses versionnées, prix séparés, aucun silent overwrite.</footer></div></body></html>;
}
