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
      setError('Le mock V1 contient uniquement le dossier Veolia.');
      return;
    }
    setError(null);
    router.push('/orotitan/veolia');
  }

  return (
    <form onSubmit={submit} className="mx-auto max-w-2xl">
      <label htmlFor="orotitan-company-search" className="sr-only">Rechercher une société</label>
      <div className="flex flex-col gap-3 sm:flex-row">
        <input
          id="orotitan-company-search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Nom, ticker ou symbole marché"
          className="min-w-0 flex-1 rounded-xl border border-slate-700 bg-slate-950/80 px-4 py-3 text-base text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-500"
        />
        <button type="submit" className="rounded-xl border border-cyan-700 bg-cyan-950/50 px-5 py-3 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-900/50">
          Ouvrir le dossier
        </button>
      </div>
      {error ? <p className="mt-3 text-sm text-amber-300">{error}</p> : null}
      <p className="mt-3 text-xs text-slate-600">Mock V1 disponible : Veolia · VIE · VIE.PA</p>
    </form>
  );
}
