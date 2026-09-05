import Link from 'next/link';

const links = [
  ['/admin', 'Vue d’ensemble'],
  ['/admin/companies', 'Sociétés'],
  ['/admin/snapshots/new', 'Nouveau snapshot'],
  ['/admin/prices', 'Cours'],
  ['/admin/data-health', 'Data Health'],
] as const;

export function AdminNav() {
  return <nav className="flex flex-wrap gap-2 rounded-xl border border-slate-800 bg-slate-900/55 p-2">
    {links.map(([href, label]) => <Link key={href} href={href} className="rounded-lg px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 hover:text-white">{label}</Link>)}
  </nav>;
}
