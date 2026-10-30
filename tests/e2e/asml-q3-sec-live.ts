/** CI-only: read official SEC Q3 HTML bytes, no credentials, storage or Registry. */
import {createHash} from 'node:crypto';
import {retrieveAsmlQ3SecExhibit,ASML_Q3_SOURCE_IDS,ASML_Q3_SEC_SOURCES}
 from '../../lib/orotitan-equity/product/asml-q3-official-sources-v1';
if(process.env.CI!=='true'||process.env.NODE_ENV==='production')
 throw Error('ASML_Q3_SEC_LIVE_CI_ONLY');
async function main(){
 let verifiedCount=0,blockedCount=0;
 for(const id of ASML_Q3_SOURCE_IDS){
  const url=ASML_Q3_SEC_SOURCES[id].url;
  let observedStatus:number|null=null;
  const traced=(async(input:string|URL|Request,options?:RequestInit)=>{
    if(String(input)!==url||options?.redirect!=='error')throw Error('UNEXPECTED_SOURCE');
    const response=await fetch(input,options);
    observedStatus=response.status;
    console.log(JSON.stringify({id,status:response.status,mime:response.headers.get('content-type'),redirected:response.redirected}));
    return response;
  }) as typeof fetch;
  let retrieved:Awaited<ReturnType<typeof retrieveAsmlQ3SecExhibit>>;
  try{retrieved=await retrieveAsmlQ3SecExhibit(id,traced);}
  catch(error){
    // A provider's explicit 403 is an observed BLOCKER, not evidence
    // retrieval success. Fail the contract on all other unexpected errors.
    if(error instanceof Error && error.message==='ASML_Q3_ORIGIN_OR_STATUS_UNVERIFIED' &&
       observedStatus===403){
      blockedCount++;
      console.log(JSON.stringify({result:'BLOCKED_BY_PROVIDER_403',id,
        verified:false,archived:false,admitted:false}));
      continue;
    }
    throw error;
  }
  const {candidate,bytes}=retrieved;
  if(createHash('sha256').update(bytes).digest('hex')!==candidate.sha256)
    throw Error('SEC_LIVE_HASH_MISMATCH');
  verifiedCount++;
  console.log(JSON.stringify({result:'SEC_Q3_LIVE_VERIFIED',id,size:candidate.byteSize,
    hash:candidate.sha256,published:candidate.publicationDate,admitted:false,archived:false}));
 }
 console.log(JSON.stringify({result:'SEC_Q3_DIAGNOSTIC_ONLY',verifiedCount,blockedCount,
   candidateArchiveObjectsCreated:0,registryAdmitted:false}));
}
main().catch(e=>{console.error('ASML_Q3_SEC_PROBE_FAILED',e);process.exitCode=1;});
