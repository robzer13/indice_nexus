import type {Metadata} from 'next';
import {loginAction} from '@/app/admin/actions';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedC8ReadOnlyEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {C8OwnerOtpPanel} from './owner-otp-panel';

export const metadata:Metadata={title:'OroTitan · J0 C8 · Attestation propriétaire SHADOW'};
export const dynamic='force-dynamic';

export default async function C8OwnerAttestationPage(){
  if(!isHostedC8ReadOnlyEnabled(process.env)){
    return <main className="p-6 text-rose-300">SHADOW_ONLY</main>;
  }
  const adminAuthenticated=await isAdminAuthenticated();
  return <main className="mx-auto max-w-xl space-y-5 p-6 text-slate-100">
    <h1 className="text-2xl font-semibold">J0 · Attestation SOLO_OWNER</h1>
    <p className="text-sm text-slate-300">Projet SHADOW seulement. Cette page vérifie le compte propriétaire ; elle ne prépare aucune revue, n’accepte pas J0 et ne démarre pas J1.</p>
    {!adminAuthenticated?
      <form action={loginAction} className="space-y-3 rounded-lg border border-slate-700 p-4">
        <p>Connexion administrateur requise.</p>
        <label className="block text-sm">Mot de passe administrateur
          <input name="password" type="password" required autoComplete="current-password"
            className="mt-1 w-full rounded border border-slate-600 bg-slate-900 p-2"/>
        </label>
        <button className="rounded bg-cyan-300 px-3 py-2 font-semibold text-slate-950">Se connecter</button>
      </form>:
      <C8OwnerOtpPanel/>}
    <p className="text-xs text-slate-400">Lecture seule C8 maintenue. Aucune décision de jalon n'est autorisée depuis cette page.</p>
  </main>;
}
