'use client';

import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { buildRunHref } from '@/lib/orotitan-ui/presentation';

export function IssuerNav({ issuerSlug }: { issuerSlug: string }) {
  const searchParams = useSearchParams();
  const run = searchParams.get('run');
  const base = '/orotitan/' + issuerSlug;

  const links = [
    { href: buildRunHref(base, run), label: "Vue d'ensemble" },
    { href: buildRunHref(base + '/documents', run), label: 'Documents' },
    { href: buildRunHref(base + '/context', run), label: 'Contexte' },
  ];

  return (
    <nav className="mt-5 space-y-1 text-sm">
      {links.map((link, index) => (
        <Link
          key={link.href}
          href={link.href}
          className={
            'block rounded-lg px-3 py-2 hover:bg-slate-800 ' +
            (index === 0 ? 'text-slate-300' : 'text-slate-400 hover:text-slate-200')
          }
        >
          {link.label}
        </Link>
      ))}
    </nav>
  );
}
