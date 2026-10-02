'use client';

import { FormEvent, useState } from 'react';
import { useRouter } from 'next/navigation';

const acceptedQueries = new Set(['veolia', 'vie', 'vie.pa', 'veolia environnement s.a.', 'veolia environnement']);

export function CompanySearch() {
  const router = useRouter();
  const [query, setQuery] = useState('Veolia');
  const [error, setError] = useState<string | null>(null);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      setError('Saisissez un nom, un ticker ou un symbole de marché.');
      return;
    }
    if (!acceptedQueries.has(normalized)) {
      setError('Aucun dossier disponible pour cette recherche dans cette version.');
      return;
    }
    setError(null);
    router.push('/orotitan/veolia');
  }

  return (
    <form onSubmit={submit}>
      <label htmlFor="orotitan-company-search" className="sr-only">Rechercher une société</label>
      <div className="flex items-center border-b border-slate-700 bg-slate-950/40 transition focus-within:border-cyan-500">
        <span aria-hidden="true" className="pl-1 text-lg text-slate-600">⌕</span>
        <input
          id="orotitan-company-search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Veolia, VIE, VIE.PA…"
          className="min-w-0 flex-1 bg-transparent px-3 py-4 text-lg text-white outline-none placeholder:text-slate-700"
        />
        <button type="submit" className="ml-3 shrink-0 rounded-md border border-slate-700 px-3 py-2 text-sm font-medium text-slate-300 transition hover:border-slate-600 hover:bg-slate-900 hover:text-white">
          Ouvrir
        </button>
      </div>
      {error ? <p className="mt-3 text-sm text-amber-300">{error}</p> : null}
    </form>
  );
}
