/**
 * J0/C5: verify that denial responses originate from the OroTitan
 * application, NOT Vercel's separate Authentication/Deployment Protection.
 * Pure and side-effect free, suitable for isolated regression tests.
 */
export type J0C5HttpProbeName =
  'CROSS_ORIGIN'|'UNSUPPORTED_CONTENT_TYPE'|'UNEXPECTED_FIELDS'|
  'FORGED_ACCEPTANCE';
export type J0C5HttpObservation=Readonly<{
  name:J0C5HttpProbeName;
  status:number;
  error:string;
  accepted:boolean;
  stageStarted:boolean;
  cacheControl:string;
}>;
const EXPECTED:Readonly<Record<J0C5HttpProbeName,Readonly<{
 status:number;error:string;
}>>>={
 CROSS_ORIGIN:{status:403,error:'ORIGIN_MISMATCH'},
 UNSUPPORTED_CONTENT_TYPE:{status:400,error:'REQUEST_INVALID'},
 UNEXPECTED_FIELDS:{status:400,error:'J0_UNEXPECTED_FIELDS'},
 FORGED_ACCEPTANCE:{status:400,error:'J0_INPUT_INVALID'},
};
export function verifyHostedJ0C5HttpMatrix(
 items:readonly J0C5HttpObservation[],
):Readonly<{passed:true;cases:4;results:readonly J0C5HttpObservation[]}>{
 if(items.length!==4)throw Error('J0_C5_HTTP_CASE_COUNT');
 const found=new Set<J0C5HttpProbeName>();
 for(const row of items){
  if(found.has(row.name)||!Object.prototype.hasOwnProperty.call(EXPECTED,row.name))
   throw Error('J0_C5_HTTP_DUPLICATE_OR_UNKNOWN');
  found.add(row.name);
  const expected=EXPECTED[row.name];
  if(row.status!==expected.status||row.error!==expected.error||
    row.accepted!==false||row.stageStarted!==false||
    !row.cacheControl.toLowerCase().includes('no-store'))
   throw Error('J0_C5_HTTP_APPLICATION_DENIAL_MISSING:'+row.name);
 }
 return {passed:true,cases:4,results:[...items]};
}
