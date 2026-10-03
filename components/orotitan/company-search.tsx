'use client';

import { FormEvent, useState } from 'react';
import { useRouter } from 'next/navigation';

export function CompanySearch() {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [error, setError] = useState<string | null>(null);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = query.trim();
    if (!value) {
      setError('Saisissez un nom, un ticker ou un symbole de marché.');
      return;
    }

    setError(null);
    router.push('/orotitan/' + encodeURIComponent(value));
  }

  return (
    <form onSubmit={submit}>
      <label htmlFor="orotitan-company-search" className="sr-only">Rechercher une société</label>
      <div className="flex items-center rounded-xl border border-[rgba(123,190,235,.24)] bg-[rgba(7,17,29,.84)] p-1.5 shadow-[0_16px_50px_rgba(0,0,0,.20)] transition focus-within:border-cyan-400/60">
        <span aria-hidden="true" className="pl-3 text-xl text-cyan-300">⌕</span>
        <input
          id="orotitan-company-search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Nom exact, ticker ou symbole…"
          className="min-w-0 flex-1 bg-transparent px-3 py-3 text-base text-white outline-none placeholder:text-slate-700"
        />
        <button
          type="submit"
          className="shrink-0 rounded-lg border border-cyan-400/25 bg-cyan-400/10 px-4 py-2.5 text-sm font-medium text-cyan-100 transition hover:border-cyan-300/40 hover:bg-cyan-400/15"
        >
          Ouvrir
        </button>
      </div>
      {error ? <p className="mt-3 text-sm text-amber-300">{error}</p> : null}
    </form>
  );
}
