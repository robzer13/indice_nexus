'use client';

import Link from 'next/link';
import { usePathname, useSearchParams } from 'next/navigation';
import { buildRunHref } from '@/lib/orotitan-ui/presentation';

export function IssuerNav({ issuerSlug }: { issuerSlug: string }) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const run = searchParams.get('run');
  const base = '/orotitan/' + issuerSlug;

  const links = [
    { href: buildRunHref(base, run), label: "Vue d'ensemble", path: base },
    { href: buildRunHref(base + '/documents', run), label: 'Documents', path: base + '/documents' },
    { href: buildRunHref(base + '/context', run), label: 'Contexte', path: base + '/context' },
  ];

  return (
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
  );
}
