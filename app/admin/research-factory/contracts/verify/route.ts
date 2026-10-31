import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getContractProfileSummary,requireExplicitProfile,
 verifyPinnedContractBytes} from
 '@/lib/orotitan-equity/product/runfactory-contract-profile-v1';
import type {GithubBlobRequest} from
 '@/lib/orotitan-equity/v1/contract-pin-resolver';

export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=60;

const MAX_BLOB_BYTES=25_000_000;
const MAX_BODY_BYTES=512;
const headers={'Cache-Control':'private, no-store'};
const safeErrorCodes=new Set([
 'EXPLICIT_CONTRACT_PROFILE_REQUIRED','CONTRACT_PROFILE_INVALID',
 'CONTRACT_PIN_NOT_IN_SELECTED_PROFILE','CONTRACT_ORIGINAL_BYTES_UNAVAILABLE',
 'CONTRACT_ORIGINAL_BYTES_TOO_LARGE','CONTRACT_PIN_BYTE_INTEGRITY_FAILURE',
]);
async function fetchPinnedGithubBlob(input:GithubBlobRequest):Promise<Uint8Array>{
 // The V1 resolver independently enforces the same repository and path/hash
 // invariants before calling this function. Re-check at HTTP boundary.
 if(input.repository!=='robzer13/indice_nexus'||
    !/^[a-f0-9]{40}$/.test(input.commitSha)||
    !/^(?!\/)(?!.*\.\.)(?!.*\/\/)[a-zA-Z0-9_./-]+$/.test(input.path))
  throw Error('CONTRACT_ORIGINAL_BYTES_UNAVAILABLE');
 const pieces=input.path.split('/').map(encodeURIComponent).join('/');
 const url='https://raw.githubusercontent.com/robzer13/indice_nexus/'+
  input.commitSha+'/'+pieces;
 let res:Response;
 try{
  res=await fetch(url,{cache:'no-store',
   headers:{Accept:'application/octet-stream'},
   signal:AbortSignal.timeout(20000)});
 }catch{throw Error('CONTRACT_ORIGINAL_BYTES_UNAVAILABLE');}
 if(!res.ok||!res.body)throw Error('CONTRACT_ORIGINAL_BYTES_UNAVAILABLE');
 const declared=Number(res.headers.get('content-length')??'NaN');
 if(Number.isFinite(declared)&&declared>MAX_BLOB_BYTES)
  throw Error('CONTRACT_ORIGINAL_BYTES_TOO_LARGE');
 const reader=res.body.getReader();
 const chunks:Uint8Array[]=[];
 let total=0;
 try{
  while(true){
   const next=await reader.read();
   if(next.done)break;
   total+=next.value.byteLength;
   if(total>MAX_BLOB_BYTES)
    throw Error('CONTRACT_ORIGINAL_BYTES_TOO_LARGE');
   chunks.push(next.value);
  }
 }catch(e){
  if(e instanceof Error&&e.message==='CONTRACT_ORIGINAL_BYTES_TOO_LARGE')throw e;
  throw Error('CONTRACT_ORIGINAL_BYTES_UNAVAILABLE');
 }finally{await reader.cancel().catch(()=>{});}
 return Buffer.concat(chunks.map(x=>Buffer.from(x)));
}
export async function POST(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))
  return NextResponse.json({ok:false,error:'SHADOW_ONLY'},{status:404,headers});
 try{await requireAdminSession();}
 catch{return NextResponse.json({ok:false,error:'UNAUTHORIZED'},{status:401,headers});}
 if(req.headers.get('origin')!==new URL(req.url).origin)
  return NextResponse.json({ok:false,error:'ORIGIN_MISMATCH'},{status:403,headers});
 const len=Number(req.headers.get('content-length')??'NaN');
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json'||
    !Number.isSafeInteger(len)||len<5||len>MAX_BODY_BYTES)
  return NextResponse.json({ok:false,error:'REQUEST_INVALID'},{status:400,headers});
 try{
  const body:unknown=await req.json();
  if(!body||typeof body!=='object'||Array.isArray(body))
   return NextResponse.json({ok:false,error:'REQUEST_INVALID'},{status:400,headers});
  const parsed=body as Record<string,unknown>;
  const profileId=requireExplicitProfile(parsed.profileId);
  if(typeof parsed.logicalName!=='string')
   return NextResponse.json({ok:false,error:'REQUEST_INVALID'},{status:400,headers});
  const summary=getContractProfileSummary(profileId);
  const record=await verifyPinnedContractBytes(
   profileId,parsed.logicalName,fetchPinnedGithubBlob,
  );
  return NextResponse.json({ok:true,contractSetSha256:summary.contractSetSha256,
   ...record,issuerResearchAuthorized:false,registryAdmitted:false,
   evidenceAdmitted:false,checkedAt:new Date().toISOString()},
   {headers});
 }catch(e){
  const message=e instanceof Error?e.message:'';
  const error=safeErrorCodes.has(message)?message:
   message.includes('mismatch')||message.includes('hash')||
   message.includes('decompression')||message.includes('archive')?
    'CONTRACT_PIN_BYTE_INTEGRITY_FAILURE':'CONTRACT_ORIGINAL_BYTES_UNAVAILABLE';
  return NextResponse.json({ok:false,error,registryAdmitted:false},
   {status:422,headers});
 }
}
