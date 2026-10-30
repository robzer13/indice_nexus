/** CI-only: read official SEC Q3 HTML bytes, no credentials, storage or Registry. */
import {createHash} from 'node:crypto';
import {retrieveAsmlQ3SecExhibit,ASML_Q3_SOURCE_IDS,ASML_Q3_SEC_SOURCES}
 from '../../lib/orotitan-equity/product/asml-q3-official-sources-v1';
if(process.env.CI!=='true'||process.env.NODE_ENV==='production')
 throw Error('ASML_Q3_SEC_LIVE_CI_ONLY');
async function main(){
 for(const id of ASML_Q3_SOURCE_IDS){
  const url=ASML_Q3_SEC_SOURCES[id].url;
  const traced=(async(input:string|URL|Request,options?:RequestInit)=>{
    if(String(input)!==url||options?.redirect!=='error')throw Error('UNEXPECTED_SOURCE');
    const response=await fetch(input,options);
    console.log(JSON.stringify({id,status:response.status,mime:response.headers.get('content-type'),redirected:response.redirected}));
    return response;
  }) as typeof fetch;
  const {candidate,bytes}=await retrieveAsmlQ3SecExhibit(id,traced);
  if(createHash('sha256').update(bytes).digest('hex')!==candidate.sha256)
    throw Error('SEC_LIVE_HASH_MISMATCH');
  console.log(JSON.stringify({result:'SEC_Q3_LIVE_VERIFIED',id,size:candidate.byteSize,
    hash:candidate.sha256,published:candidate.publicationDate,admitted:false,archived:false}));
 }
}
main().catch(e=>{console.error('ASML_Q3_SEC_PROBE_FAILED',e);process.exitCode=1;});
