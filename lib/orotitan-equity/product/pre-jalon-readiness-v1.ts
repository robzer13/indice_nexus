/**
 * PRE-JALON DESIGN GATE — engineering milestones J1..J8 only.
 * Pure, fail-closed policy projection; it writes nothing and starts nothing.
 * Caller MUST derive acceptance from persisted C8 decisions, never from
 * milestone.initial_status, and MUST verify the plan/review/GO server-side.
 */
export type PreJalonMilestone =
 'J1'|'J2'|'J3'|'J4'|'J5'|'J6'|'J7'|'J8';
export type VerifiedDesignPlan = Readonly<{
 immutableRef: string;
 sha256: string;
 sourceHashVerified: boolean;
 exactEightCriteriaVerified: boolean;
 functionalOutcomeDefined: boolean;
 existingReuseConsidered: boolean;
 alternativesCompared: boolean;
 deliverySlicesAndProofsDefined: boolean;
 negativeTestsAndStopConditionsDefined: boolean;
 authorPrincipal: string;
 reviewerPrincipal: string;
 reviewerIdentityVerified: boolean;
 independentReviewVerdict: 'APPROVED'|'REVISE'|'BLOCKED';
 reviewedSha256: string;
}>;
export type VerifiedBuildAuthorization = Readonly<{
 milestoneCode: string;
 authorityRef: string;
 independentlyVerified: boolean;
}>;
export type PreJalonInput = Readonly<{
 milestoneCode: string;
 criteria: readonly string[];
 predecessorC8Accepted: boolean;
 predecessorOpenBlockers: number;
 plan: VerifiedDesignPlan|null;
 buildGo: VerifiedBuildAuthorization|null;
}>;
export type PreJalonResult = Readonly<{
 milestoneCode: string;
 predecessor: string;
 planningDraftAllowed: boolean;
 planApproved: boolean;
 buildAuthorized: boolean;
 status: 'BLOCKED'|'DRAFT_ONLY'|'REVISE_PLAN'|'PLAN_APPROVED'|'BUILD_AUTHORIZED';
 blockers: readonly string[];
}>;
const sha256=/^[a-f0-9]{64}$/;
const goPattern=/^GO_BUILD_J[1-8]_SHADOW_[A-Z0-9_]{8,120}$/;
const immutableOfficialRef=/^https:\/\/github\.com\/robzer13\/indice_nexus\/blob\/[a-f0-9]{40}\/docs\/[^?#]+\.md$/;
export function evaluatePreJalonReadinessV1(input:PreJalonInput):PreJalonResult {
 const match=/^J([1-8])$/.exec(input.milestoneCode);
 if(!match)throw Error('PJDG_NOT_AN_ENGINEERING_MILESTONE_J1_J8');
 const predecessor='J'+(Number(match[1])-1);
 const blockers:string[]=[];
 if(!Array.isArray(input.criteria)||input.criteria.length!==8||
  input.criteria.some(x=>typeof x!=='string'||x.trim().length<3))
  blockers.push('EXACT_EIGHT_CRITERIA_MISSING');
 if(input.predecessorC8Accepted!==true)
  blockers.push('PREDECESSOR_C8_NOT_ACCEPTED');
 if(!Number.isSafeInteger(input.predecessorOpenBlockers)||
  input.predecessorOpenBlockers<0)
  blockers.push('PREDECESSOR_INCIDENT_STATE_UNVERIFIED');
 else if(input.predecessorOpenBlockers>0)
  blockers.push('PREDECESSOR_OPEN_BLOCKER');
 const plan=input.plan;
 if(!plan)blockers.push('IMMUTABLE_DESIGN_PLAN_MISSING');
 else {
  if(!immutableOfficialRef.test(plan.immutableRef)||!sha256.test(plan.sha256)||
     plan.sourceHashVerified!==true)
   blockers.push('PLAN_IMMUTABILITY_NOT_VERIFIED');
  if(!plan.exactEightCriteriaVerified||!plan.functionalOutcomeDefined||
     !plan.existingReuseConsidered||!plan.alternativesCompared||
     !plan.deliverySlicesAndProofsDefined||
     !plan.negativeTestsAndStopConditionsDefined)
   blockers.push('DESIGN_PLAN_INCOMPLETE');
  if(!plan.reviewerIdentityVerified||
     !plan.authorPrincipal.trim()||!plan.reviewerPrincipal.trim()||
     plan.authorPrincipal.trim()===plan.reviewerPrincipal.trim()||
     plan.reviewedSha256!==plan.sha256)
   blockers.push('INDEPENDENT_REVIEW_NOT_VERIFIED');
  if(plan.independentReviewVerdict!=='APPROVED')
   blockers.push(plan.independentReviewVerdict==='REVISE'?
    'PLAN_REQUIRES_REVISION':'PLAN_REVIEW_NOT_APPROVED');
 }
 const planApproved=blockers.length===0;
 if(!input.buildGo)blockers.push('DISTINCT_GO_BUILD_MISSING');
 else if(input.buildGo.milestoneCode!==input.milestoneCode||
     !goPattern.test(input.buildGo.authorityRef)||
     !input.buildGo.authorityRef.startsWith('GO_BUILD_'+input.milestoneCode+'_SHADOW_')||
     input.buildGo.independentlyVerified!==true)
  blockers.push('GO_BUILD_NOT_VERIFIED');
 const buildAuthorized=planApproved&&blockers.length===0;
 const status:PreJalonResult['status']=buildAuthorized?'BUILD_AUTHORIZED':
  !input.predecessorC8Accepted||input.predecessorOpenBlockers!==0?
   'DRAFT_ONLY':plan?.independentReviewVerdict==='REVISE'?
    'REVISE_PLAN':planApproved?'PLAN_APPROVED':'BLOCKED';
 return {milestoneCode:input.milestoneCode,predecessor,
  planningDraftAllowed:true,planApproved,buildAuthorized,status,blockers};
}
