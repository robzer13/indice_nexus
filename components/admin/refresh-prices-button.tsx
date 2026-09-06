'use client';

import { useFormStatus } from 'react-dom';

export function RefreshPricesButton() {
  const { pending } = useFormStatus();

  return <button
    type="submit"
    disabled={pending}
    className="rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
  >
    {pending ? 'Synchronisation…' : 'Actualiser maintenant'}
  </button>;
}
