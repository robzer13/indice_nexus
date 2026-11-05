import type {Metadata} from 'next';
import {loginAction} from '@/app/admin/actions';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedC8ReadOnlyEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {C8OwnerOtpPanel} from './owner-otp-panel';
import {J0C8GatePanel} from '../j0-c8-gate-panel';
import {getC8GateReviewClient} from '@/lib/orotitan-equity/product/c8-gate-read-server';
import {getSoloOwnerConfig} from '@/lib/auth/c8-solo-owner-core';

export const metadata:Metadata={title:'OroTitan · J0 C8 · Attestation propriétaire SHADOW'};
export const dynamic='force-dynamic';

export default async function C8OwnerAttestationPage(){
  if(!isHostedC8ReadOnlyEnabled(process.env)){
    return <main className="p-6 text-rose-300">SHADOW_ONLY</main>;
  }
  const adminAuthenticated=await isAdminAuthenticated();
  let proofs: {eventId:string;title:string;criterionIndex:number;verified:boolean}[]=[];
  let evidenceReady=false;
  let existingReview:{
    ok:true;status:string;reviewId:string;reviewSha256:string;
    persistedReadbackVerified:true;accepted:false;governanceMode:'SOLO_OWNER'
  }|null=null;
  if(adminAuthenticated){
    try{
      const client=getC8GateReviewClient();
      const result=await client
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
        if(evidenceReady){
          const owner=getSoloOwnerConfig(process.env);
          const reviews=await client.from('orotitan_j0_c8_reviews_v1')
            .select('review_id,review_sha256,milestone_code,governance_mode,auth_provider,owner_user_id,amendment_ref,amendment_sha256,evidence_event_ids,journal_watermark')
            .eq('milestone_code','J0').eq('governance_mode','SOLO_OWNER')
            .eq('owner_user_id',owner.ownerUserId).limit(2);
          if(!reviews.error&&reviews.data?.length===1){
            const review=reviews.data[0];
            const ids=review.evidence_event_ids;
            const lastEvent=result.data[0]?.event_seq;
            const validIds=Array.isArray(ids)&&ids.length===7&&
              ids.every((eventId:unknown,i:number)=>typeof eventId==='string'&&
                proofs.some(p=>p.eventId===eventId&&p.criterionIndex===i+1&&
                  (![1,6,7].includes(i+1)||p.verified)));
            if(validIds&&
               review.auth_provider==='SUPABASE_AUTH'&&
               review.owner_user_id===owner.ownerUserId&&
               review.amendment_ref===owner.amendmentRef&&
               review.amendment_sha256===owner.amendmentSha256&&
               review.journal_watermark===lastEvent&&
               typeof review.review_id==='string'&&
               typeof review.review_sha256==='string'&&
               /^[a-f0-9]{64}$/.test(review.review_sha256)){
              existingReview={
                ok:true,status:'PREPARED',
                reviewId:review.review_id,
                reviewSha256:review.review_sha256,
                persistedReadbackVerified:true,accepted:false,
                governanceMode:'SOLO_OWNER',
              };
            }
          }
        }
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
          <J0C8GatePanel proofs={proofs} soloOnly existingReview={existingReview}/>:
          <p className="rounded border border-rose-800 p-3 text-rose-300">
            Preuves J0 indisponibles ou incomplètes : revue bloquée.
          </p>}
      </div>}
    <p className="text-xs text-slate-400">Lecture seule C8 maintenue. Aucune décision de jalon n&apos;est autorisée depuis cette page.</p>
  </main>;
}
