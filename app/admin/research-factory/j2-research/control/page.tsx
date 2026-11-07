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
  return <main className="mx-auto max-w-3xl space-y-4 p-6">
   <h1 className="text-2xl font-semibold">OroTitan — J2 Research</h1>
   <p className="text-amber-100">Sélectionne un RUN_ID canonique déjà vérifié en J1.</p>
   <form method="GET" className="flex flex-wrap gap-3">
    <input name="runId" aria-label="RUN_ID" required pattern="[A-Fa-f0-9-]{36}"
     placeholder="RUN_ID" className="min-w-80 rounded bg-slate-900 p-3"/>
    <button type="submit" className="rounded bg-cyan-400 px-4 py-3 text-slate-950">
     Contrôler le Run
    </button>
   </form>
  </main>;
 return <main className="mx-auto max-w-3xl space-y-5 p-6">
  <h1 className="text-2xl font-semibold">OroTitan — J2 Research</h1>
  <J2ResearchStartClient runId={runId}/>
 </main>;
}
