import 'server-only';
/**
 * OroTitan P0-E: full original-byte gate for the explicit, immutable pack.
 * No live Registry write is permitted until ALL pin blobs pass.
 * Uses pre-existing V1 resolver (Git blob SHA-1 + canonical SHA-256).
 */
import {
 getContractProfileSummary,verifyPinnedContractBytes,type ContractProfileId,
} from './runfactory-contract-profile-v1';
import type {GithubBlobRequest} from '../v1/contract-pin-resolver';

const MAX_BYTES=25_000_000;
const repo='robzer13/indice_nexus';
export async function fetchRunFactoryPinnedGithubBlob(
 input:GithubBlobRequest,
):Promise<Uint8Array>{
 if(input.repository!==repo||!/^[0-9a-f]{40}$/.test(input.commitSha)||
    !/^(?!\/)(?!.*\.\.)(?!.*\/\/)[a-zA-Z0-9_./-]+$/.test(input.path))
  throw Error('P0E_PIN_LOCATOR_INVALID');
 const path=input.path.split('/').map(encodeURIComponent).join('/');
 const url='https://raw.githubusercontent.com/'+repo+'/'+input.commitSha+'/'+path;
 let response:Response;
 try{
  response=await fetch(url,{
   cache:'no-store',headers:{Accept:'application/octet-stream'},
   signal:AbortSignal.timeout(20000),
  });
 }catch{throw Error('P0E_PIN_FETCH_FAILED');}
 if(!response.ok||!response.body)throw Error('P0E_PIN_FETCH_FAILED');
 const known=Number(response.headers.get('content-length')??'NaN');
 if(Number.isFinite(known)&&known>MAX_BYTES)
  throw Error('P0E_PIN_TOO_LARGE');
 const chunks:Uint8Array[]=[];
 let size=0;
 const reader=response.body.getReader();
 try{
  for(;;){
   const r=await reader.read();
   if(r.done)break;
   size+=r.value.byteLength;
   if(size>MAX_BYTES)throw Error('P0E_PIN_TOO_LARGE');
   chunks.push(r.value);
  }
 }finally{await reader.cancel().catch(()=>{});}
 return Buffer.concat(chunks.map(c=>Buffer.from(c)));
}
export type P0EFullByteReport=Readonly<{
 profileId:ContractProfileId;
 contractSetSha256:string;
 pinCount:number;passedPinCount:number;
 canonicalBytes:number;
 checks:readonly Readonly<{
  logicalName:string;sha256:string;byteLength:number;
  status:'ORIGINAL_CONTRACT_BYTES_VERIFIED';
 }>[];
 allOriginalBytesVerified:true;registryMutated:false;
}>;
export async function verifyAllRunFactoryPinnedBytes(
 profileId:ContractProfileId,
):Promise<P0EFullByteReport>{
 const summary=getContractProfileSummary(profileId);
 const checks:P0EFullByteReport['checks'][number][]=[];
 // Bounded concurrency avoids a request burst and limits memory.
 const pins=[...summary.pins];
 for(let offset=0;offset<pins.length;offset+=3){
  const batch=await Promise.all(
   pins.slice(offset,offset+3).map(async pin=>{
    const found=await verifyPinnedContractBytes(
     profileId,pin.logicalName,fetchRunFactoryPinnedGithubBlob);
    if(found.expectedSha256!==pin.contentSha256||
       found.resolvedSha256!==pin.contentSha256||
       found.status!=='ORIGINAL_CONTRACT_BYTES_VERIFIED')
      throw Error('P0E_PIN_INTEGRITY_FAILURE');
    return {logicalName:pin.logicalName,sha256:found.resolvedSha256,
     byteLength:found.resolvedBytes,
     status:found.status};
   }));
  checks.push(...batch);
 }
 if(checks.length!==summary.pinCount||summary.pinCount!==15)
  throw Error('P0E_PIN_COUNT_MISMATCH');
 return {profileId,contractSetSha256:summary.contractSetSha256,
  pinCount:summary.pinCount,passedPinCount:checks.length,
  canonicalBytes:checks.reduce((sum,r)=>sum+r.byteLength,0),
  checks,allOriginalBytesVerified:true,registryMutated:false};
}
