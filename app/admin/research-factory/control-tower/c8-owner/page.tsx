import type {Metadata} from 'next';
import {loginAction} from '@/app/admin/actions';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedC8ReadOnlyEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {C8OwnerOtpPanel} from './owner-otp-panel';
import {J0C8GatePanel} from '../j0-c8-gate-panel';
import {getC8GateReviewClient} from '@/lib/orotitan-equity/product/c8-gate-read-server';

export const metadata:Metadata={title:'OroTitan · J0 C8 · Attestation propriétaire SHADOW'};
export const dynamic='force-dynamic';

export default async function C8OwnerAttestationPage(){
  if(!isHostedC8ReadOnlyEnabled(process.env)){
    return <main className="p-6 text-rose-300">SHADOW_ONLY</main>;
  }
  const adminAuthenticated=await isAdminAuthenticated();
  let proofs: {eventId:string;title:string;criterionIndex:number;verified:boolean}[]=[];
  let evidenceReady=false;
  if(adminAuthenticated){
    try{
      const result=await getC8GateReviewClient()
        .from('orotitan_j0_events_v1')
        .select('event_id,event_seq,milestone_code,event_type,title,verification_state,operator_details')
        .eq('milestone_code','J0').order('event_seq',{ascending:false}).limit(100);
      if(!result.error&&Array.isArray(result.data)&&result.data.length<100){
        proofs=result.data.filter(event=>
          event.event_type==='EVIDENCE_RECORDED' &&
          typeof event.operator_details?.criterionIndex==='number' &&
          event.operator_details.criterionIndex>=1 &&
          event.operator_details.criterionIndex<=7
        ).map(event=>({
          eventId:event.event_id as string,
          title:event.title as string,
          criterionIndex:event.operator_details.criterionIndex as number,
          verified:event.verification_state==='VERIFIED_DIRECT_REGISTRY_READBACK',
        }));
        evidenceReady=[1,2,3,4,5,6,7].every(index=>
          proofs.some(p=>p.criterionIndex===index&&
            (![1,6,7].includes(index)||p.verified)));
      }
    }catch{/* Fail closed: no evidence form when Registry readback fails. */}
  }
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
      <div className="space-y-6">
        <C8OwnerOtpPanel/>
        {evidenceReady?
          <J0C8GatePanel proofs={proofs} soloOnly/>:
          <p className="rounded border border-rose-800 p-3 text-rose-300">
            Preuves J0 indisponibles ou incomplètes : revue bloquée.
          </p>}
      </div>}
    <p className="text-xs text-slate-400">Lecture seule C8 maintenue. Aucune décision de jalon n&apos;est autorisée depuis cette page.</p>
  </main>;
}
