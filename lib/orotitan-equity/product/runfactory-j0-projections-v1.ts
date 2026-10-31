/**
 * Fail-closed J0 projections. These are operational observations,
 * never a substitute for independent milestone-gate acceptance.
 */
export type J0IssueEvent=Readonly<{
 event_id:string;event_seq:number;milestone_code:string;event_type:string;
 evidence_ref:string|null;created_at:string;
}>;
export function unresolvedJ0Issues<T extends J0IssueEvent>(
 events:readonly T[],milestone:string):T[]{
 const issues=new Map<string,T>();
 // A source of truth may return rows in any order: sort explicitly and
 // use deterministic event IDs to break equal-timestamp ties.
 const ordered=[...events].filter(e=>
  e.milestone_code===milestone &&
  ['BLOCKER_OPENED','BLOCKER_RESOLVED'].includes(e.event_type)&&
  !!e.evidence_ref).sort((a,b)=>
  a.event_seq-b.event_seq);
 for(const e of ordered){
  if(!e.evidence_ref)continue;
  if(e.event_type==='BLOCKER_OPENED')issues.set(e.evidence_ref,e);
  else issues.delete(e.evidence_ref);
 }
 return [...issues.values()].sort((a,b)=>a.event_seq-b.event_seq);
}
export function j0DisplayedStatus(input:{
 initialStatus:string;unresolvedBlockerCount:number;
}):string{
 if(!Number.isSafeInteger(input.unresolvedBlockerCount)||
    input.unresolvedBlockerCount<0)throw Error('J0_BLOCKER_COUNT_INVALID');
 if(!['NON_DEMARRE','EN_COURS','BLOQUE','TESTE','ACCEPTE']
  .includes(input.initialStatus))
  throw Error('J0_UNKNOWN_MILESTONE_STATE');
 // A documented blocker must never be masked by the initial plan status.
 return input.unresolvedBlockerCount>0?'BLOQUE':input.initialStatus;
}
