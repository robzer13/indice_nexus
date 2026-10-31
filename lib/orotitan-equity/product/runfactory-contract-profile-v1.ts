/**
 * OroTitan Run Factory P0-A — read-only contract-profile catalog.
 *
 * The user/approved policy chooses a profile EXPLICITLY. Never infer "latest".
 * The authoritative data are existing immutable pin packs, never copied pins.
 * Structure/hash verification is distinct from ORIGINAL BYTE verification.
 * No Registry, Source Vault, stage or production writes.
 */
import {computeContractSetSha256} from '../v1/stage-manifest';
import {resolveContractPin,sha256Hex,type ContractPin,
 type GithubBlobFetcher} from '../v1/contract-pin-resolver';
import packV1 from '../../../contracts/orotitan-equity/v1/contract-pin-pack-v1/OROTITAN_CONTRACT_PIN_PACK_V1.json';
import packV2 from '../../../contracts/orotitan-equity/v2/contract-pin-pack-v2/OROTITAN_CONTRACT_PIN_PACK_V2.json';
import packV3 from '../../../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json';

export type ContractProfileId='OROTITAN_V1'|'OROTITAN_V2'|'OROTITAN_V3_1';
export type ProfilePinDescriptor=Readonly<{
 logicalName:string;name:string;version:string;
 contentSha256:string;locatorFormat:string;
 sourceCommitSha:string;
}>;
export type ContractProfileSummary=Readonly<{
 profileId:ContractProfileId;label:string;pinPackVersion:string;
 sourceCommitSha:string;contractSetSha256:string;
 pinCount:number;pins:readonly ProfilePinDescriptor[];
 structuralStatus:'STRUCTURE_AND_SET_HASH_VERIFIED';
 originalBytesStatus:'NOT_VERIFIED';
 issuerResearchAuthorized:false;registryAdmitted:false;
}>;
export type PinByteVerification=Readonly<{
 profileId:ContractProfileId;logicalName:string;
 expectedSha256:string;resolvedSha256:string;resolvedBytes:number;
 status:'ORIGINAL_CONTRACT_BYTES_VERIFIED';registryAdmitted:false;
}>;

const REQUIRED_13=[
 'analysis_standard','deep_dive_stage','execution_patch','i2','i3b',
 'integration_spec','integration_stage','investment_policy','master_prompt',
 'pilotage','process','research_stage','screener_schema',
] as const;
const V3_EXTRA=['dcf_timing','valuation_date_alignment'] as const;
const HEX40=/^[0-9a-f]{40}$/;
const HEX64=/^[0-9a-f]{64}$/;
const safePath=/^(?!\/)(?!.*\.\.)(?!.*\/\/)[a-zA-Z0-9_./-]+$/;
const isRecord=(v:unknown):v is Record<string,unknown>=>
 typeof v==='object'&&v!==null&&!Array.isArray(v);
const invalid=():never=>{throw Error('CONTRACT_PROFILE_INVALID');};

const CATALOG={
 OROTITAN_V1:{
  label:'OroTitan V1 FREEZE',pack:packV1,
  format:'OROTITAN_CONTRACT_PIN_PACK_V1',version:'1.0',
  setHash:'34b009f05715bab482dbc00b02194b3714e9b2f8151872144677bb6db19f3c63',
  required:REQUIRED_13,
 },
 OROTITAN_V2:{
  label:'OroTitan V2 FREEZE',pack:packV2,
  format:'OROTITAN_CONTRACT_PIN_PACK_V2',version:'2.0',
  setHash:'1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e',
  required:REQUIRED_13,
 },
 OROTITAN_V3_1:{
  label:'OroTitan V3.1 FREEZE (pack 3.0.2)',pack:packV3,
  format:'OROTITAN_CONTRACT_PIN_PACK_V3',version:'3.0.2',
  setHash:'3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2',
  required:[...REQUIRED_13,...V3_EXTRA],
 },
} as const;

export function requireExplicitProfile(value:unknown):ContractProfileId{
 if(value==='OROTITAN_V1'||value==='OROTITAN_V2'||value==='OROTITAN_V3_1')
  return value;
 throw Error('EXPLICIT_CONTRACT_PROFILE_REQUIRED');
}

function validatedPack(id:ContractProfileId):
 {pins:Record<string,ContractPin>;summary:ContractProfileSummary}{
 const config=CATALOG[id];
 const raw:unknown=config.pack;
 if(!isRecord(raw)||raw.format!==config.format||
    raw.version!==config.version||
    raw.repository!=='robzer13/indice_nexus'||
    !HEX40.test(String(raw.source_commit_sha))||
    raw.contract_set_sha256!==config.setHash||
    !isRecord(raw.contract_pins))throw Error('CONTRACT_PROFILE_INVALID');
 const rawPins=raw.contract_pins;
 const actual=Object.keys(rawPins).sort();
 const expected=[...config.required].sort();
 if(actual.length!==expected.length||
    expected.some((key,i)=>actual[i]!==key))invalid();
 if(computeContractSetSha256(rawPins)!==config.setHash)invalid();
 const pins:Record<string,ContractPin>={};
 const descriptions:ProfilePinDescriptor[]=[];
 for(const key of actual){
  const candidate=rawPins[key];
  if(!isRecord(candidate)||typeof candidate.name!=='string'||
     candidate.name.trim()===''||typeof candidate.version!=='string'||
     candidate.version.trim()===''||
     typeof candidate.content_sha256!=='string'||
     !HEX64.test(candidate.content_sha256)||
     !isRecord(candidate.locator))throw Error('CONTRACT_PROFILE_INVALID');
  const locator=candidate.locator;
  if(locator.backend!=='GITHUB_IMMUTABLE'||
     locator.repository!=='robzer13/indice_nexus'||
     typeof locator.path!=='string'||!safePath.test(locator.path)||
     typeof locator.commit_sha!=='string'||
     !HEX40.test(locator.commit_sha)||
     typeof locator.blob_sha!=='string'||!HEX40.test(locator.blob_sha)||
     ![undefined,'RAW_CANONICAL_V1','OROTITAN_GZIP_V1',
       'OROTITAN_MULTIPART_GZIP_V1'].includes(
        locator.locator_format as string|undefined))throw Error('CONTRACT_PROFILE_INVALID');
  const pin=candidate as unknown as ContractPin;
  pins[key]=pin;
  descriptions.push({
   logicalName:key,name:pin.name,version:pin.version,
   contentSha256:pin.content_sha256,
   locatorFormat:pin.locator.locator_format??'RAW_CANONICAL_V1',
   sourceCommitSha:pin.locator.commit_sha,
  });
 }
 return {
  pins,summary:{
   profileId:id,label:config.label,
   pinPackVersion:config.version,
   sourceCommitSha:String(raw.source_commit_sha),
   contractSetSha256:config.setHash,
   pinCount:descriptions.length,pins:descriptions,
   structuralStatus:'STRUCTURE_AND_SET_HASH_VERIFIED',
   originalBytesStatus:'NOT_VERIFIED',
   issuerResearchAuthorized:false,registryAdmitted:false,
  },
 };
}

export function getContractProfileSummary(id:ContractProfileId):ContractProfileSummary{
 return validatedPack(requireExplicitProfile(id)).summary;
}
export function listContractProfileSummaries():readonly ContractProfileSummary[]{
 return (['OROTITAN_V1','OROTITAN_V2','OROTITAN_V3_1'] as const)
  .map(id=>getContractProfileSummary(id));
}
export async function verifyPinnedContractBytes(
 id:ContractProfileId,logicalName:string,fetchBlob:GithubBlobFetcher,
):Promise<PinByteVerification>{
 const chosen=requireExplicitProfile(id);
 const {pins}=validatedPack(chosen);
 if(typeof logicalName!=='string'||!Object.prototype.hasOwnProperty.call(pins,logicalName))
  throw Error('CONTRACT_PIN_NOT_IN_SELECTED_PROFILE');
 const pin=pins[logicalName];
 // Existing V1 resolver verifies both Git blob SHA-1 AND canonical SHA-256.
 const bytes=await resolveContractPin(pin,fetchBlob);
 const digest=sha256Hex(bytes);
 if(digest!==pin.content_sha256)invalid();
 return {profileId:chosen,logicalName,
  expectedSha256:pin.content_sha256,resolvedSha256:digest,
  resolvedBytes:bytes.byteLength,
  status:'ORIGINAL_CONTRACT_BYTES_VERIFIED',registryAdmitted:false,
 };
}
