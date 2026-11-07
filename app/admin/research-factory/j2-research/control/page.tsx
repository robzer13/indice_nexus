import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedJ2ResearchPreviewEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {J2ResearchStartClient} from './j2-research-start-client';

export const dynamic='force-dynamic';
export const metadata:Metadata={title:'OroTitan J2 · Research SHADOW'};
const UUID=/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i;
export default async function J2ResearchControl({
 searchParams,
}:{searchParams:Promise<{runId?:string}>}){
 if(!await isAdminAuthenticated())redirect('/admin');
 if(!isHostedJ2ResearchPreviewEnabled(process.env))
  return <main className="p-6">J2_SHADOW_ONLY</main>;
 const params=await searchParams;
 const runId=params.runId;
 if(!runId||!UUID.test(runId))
  return <main className="p-6 text-amber-100">
   RUN_ID canonical requis pour le préflight J2.</main>;
 return <main className="mx-auto max-w-3xl space-y-5 p-6">
  <h1 className="text-2xl font-semibold">OroTitan — J2 Research</h1>
  <J2ResearchStartClient runId={runId}/>
 </main>;
}
