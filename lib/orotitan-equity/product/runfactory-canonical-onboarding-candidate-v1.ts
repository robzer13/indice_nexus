/**
 * OroTitan Run Factory P0-C — governed canonical onboarding candidate (READ-ONLY).
 * NEVER create identities from the legacy Screener without independent
 * issuer/listing evidence. This is a planning/review artifact, NOT admission.
 */
import {createHash} from 'node:crypto';
import {assertIsin} from './runfactory-identity-preflight-v1';

export type P0CLegacyCompany=Readonly<{
 id:string;name:string;ticker:string;exchange:string;currency:string;
 active:boolean;country:string|null;
}>;
export type P0CIssuer=Readonly<{issuerId:string;displayName:string;legalName:string|null}>;
export type P0CSecurity=Readonly<{
 securityId:string;issuerId:string;isin:string;ticker:string;exchange:string;
 tradingCurrency:string;primaryListing:boolean|null;
}>;
export type P0CDossier=Readonly<{dossierId:string;issuerId:string;active:boolean}>;
export type P0CLegacyMap=Readonly<{
 legacyCompanyId:string;issuerId:string;securityId:string;dossierId:string;
}>;
export type P0CInventory=Readonly<{
 companies:readonly P0CLegacyCompany[];
 issuers:readonly P0CIssuer[];
 securities:readonly P0CSecurity[];
 dossiers:readonly P0CDossier[];
 maps:readonly P0CLegacyMap[];
}>;
export type P0CRequest=Readonly<{
 isin:string;ticker:string;exchange:string;
 issuerDisplayName?:string|null;
 evidenceUrl?:string|null;
}>;
export type P0CStatus=
 'BLOCKED_SECURITY_COLLISION'|'BLOCKED_NAME_AMBIGUOUS'|
 'BLOCKED_LEGACY_AMBIGUOUS'|'BLOCKED_LEGACY_MAPPING_CONFLICT'|
 'BLOCKED_CANONICAL_INCONSISTENT'|'EXISTING_SECURITY_REUSE_CANDIDATE'|
 'EXISTING_ISIN_OTHER_LISTING_REVIEW'|
 'EXISTING_ISSUER_NAME_REVIEW'|
 'SOURCE_AND_METADATA_REVIEW_REQUIRED';
export type P0CDecision=Readonly<{
 format:'OROTITAN_CANONICAL_IDENTITY_ONBOARDING_CANDIDATE_V1';
 status:P0CStatus;isin:string;ticker:string;exchange:string;
 proposedDisplayName:string|null;tradingCurrencyCandidate:string|null;
 existingIssuerId:string|null;existingSecurityId:string|null;
 existingDossierId:string|null;legacyCompanyId:string|null;
 issuerNameMatchIds:readonly string[];
 evidenceUrl:string|null;
 evidenceStatus:'USER_SUPPLIED_POINTER_NOT_FETCHED'|'NO_SOURCE_REFERENCE';
 metadataSource:'LEGACY_SCREENER_UNVERIFIED'|'OPERATOR_UNVERIFIED'|'ABSENT';
 needsIndependentLegalAndListingVerification:true;
 needsExchangeCurrencyQuoteUnitVerification:true;
 needsGovernedAtomicIdentityRpc:true;
 constraints:['NO_GLOBAL_ISIN_UNIQUE_INDEX','NO_CANONICAL_IDENTITY_ONBOARDING_RPC'];
 proposedWrites:readonly string[];
 blockerReasons:readonly string[];
 candidateFingerprintSha256:string;
 operationAuthorized:false;
 issuerIdentityAdmitted:false;securityIdentityAdmitted:false;
 researchDossierAdmitted:false;runCreated:false;
 rpcExecuted:false;registryMutated:false;productionTouched:false;
}>;
const invalid=(code:string):never=>{throw Error(code);};
const norm=(v:string):string=>v.trim().toUpperCase();
const normalizedName=(v:string):string=>v.trim().toLocaleLowerCase('en-US')
 .normalize('NFKC').replace(/\s+/g,' ');
function safeEvidenceUrl(raw:string|null|undefined):string|null{
 if(raw===undefined||raw===null||raw.trim()==='')return null;
 if(raw.length>2048)invalid('INVALID_EVIDENCE_URL');
 let url:URL;
 try{url=new URL(raw);}catch{return invalid('INVALID_EVIDENCE_URL');}
 if(url.protocol!=='https:'||url.username||url.password||
    url.port||url.hostname.length>253||
    !/^[a-z0-9-]+(?:\.[a-z0-9-]+)+$/i.test(url.hostname)||
    url.hostname.endsWith('.local')||
    /^\d+\.\d+\.\d+\.\d+$/.test(url.hostname))
  invalid('INVALID_EVIDENCE_URL');
 // Storing the reference is not downloading or verifying the content.
 return url.toString();
}
export function planCanonicalIdentityOnboarding(
 request:P0CRequest,observed:P0CInventory,
):P0CDecision{
 if(!request||typeof request.isin!=='string'||
    typeof request.ticker!=='string'||typeof request.exchange!=='string')
  invalid('IDENTITY_INPUT_INVALID');
 if(!observed||!Array.isArray(observed.companies)||
    !Array.isArray(observed.issuers)||!Array.isArray(observed.securities)||
    !Array.isArray(observed.dossiers)||!Array.isArray(observed.maps))
  invalid('IDENTITY_OBSERVATION_INVALID');
 const isin=assertIsin(request.isin);
 const ticker=norm(request.ticker),exchange=norm(request.exchange);
 if(!/^[A-Z0-9._-]{1,30}$/.test(ticker)||
    !/^[A-Z0-9._-]{2,30}$/.test(exchange))
  invalid('IDENTITY_LISTING_INPUT_INVALID');
 if(request.issuerDisplayName!==null&&request.issuerDisplayName!==undefined&&
    (typeof request.issuerDisplayName!=='string'||
    request.issuerDisplayName.length>180))
  invalid('IDENTITY_NAME_INPUT_INVALID');
 const evidenceUrl=safeEvidenceUrl(request.evidenceUrl);
 const suppliedName=request.issuerDisplayName?.trim()||null;
 const exactSecurity=observed.securities.filter(s=>
  norm(s.isin)===isin&&norm(s.ticker)===ticker&&norm(s.exchange)===exchange);
 const otherIsinSameListing=observed.securities.filter(s=>
  norm(s.ticker)===ticker&&norm(s.exchange)===exchange&&norm(s.isin)!==isin);
 const sameIsinOtherListing=observed.securities.filter(s=>
  norm(s.isin)===isin&&(norm(s.ticker)!==ticker||norm(s.exchange)!==exchange));
 const matchedLegacy=observed.companies.filter(c=>
  norm(c.ticker)===ticker&&norm(c.exchange)===exchange);
 const legacy=matchedLegacy.length===1?matchedLegacy[0]:null;
 const legacyMaps=legacy?observed.maps.filter(m=>m.legacyCompanyId===legacy.id):[];
 const effectiveName=suppliedName??legacy?.name.trim()??null;
 const nameMatches=effectiveName?observed.issuers.filter(i=>
  normalizedName(i.displayName)===normalizedName(effectiveName)||
  (i.legalName&&normalizedName(i.legalName)===normalizedName(effectiveName))):[];
 const exact=exactSecurity.length===1?exactSecurity[0]:null;
 const issuer=exact?observed.issuers.filter(i=>i.issuerId===exact.issuerId):[];
 const activeDossiers=exact?observed.dossiers.filter(d=>d.issuerId===exact.issuerId&&d.active):[];
 const verifiedExisting=exact&&issuer.length===1&&activeDossiers.length===1;
 const mappingConflicts=legacyMaps.length>1||Boolean(!exact&&legacyMaps.length>0)||Boolean(exact&&legacyMaps.length===1&&
  (legacyMaps[0].securityId!==exact.securityId||
   legacyMaps[0].issuerId!==exact.issuerId||
   (activeDossiers.length===1&&legacyMaps[0].dossierId!==activeDossiers[0].dossierId)));
 let status:P0CStatus='SOURCE_AND_METADATA_REVIEW_REQUIRED';
 const blockerReasons:string[]=[];
 if(otherIsinSameListing.length>0||exactSecurity.length>1){
  status='BLOCKED_SECURITY_COLLISION';
  blockerReasons.push('Exact exchange/ticker has a different ISIN or duplicate records.');
 }else if(matchedLegacy.length>1){
  status='BLOCKED_LEGACY_AMBIGUOUS';
  blockerReasons.push('Legacy Screener contains multiple identical exchange/ticker rows.');
 }else if(mappingConflicts){
  status='BLOCKED_LEGACY_MAPPING_CONFLICT';
  blockerReasons.push('Legacy crosswalk disagrees with canonical issuer/security/dossier.');
 }else if(exact&&!verifiedExisting){
  status='BLOCKED_CANONICAL_INCONSISTENT';
  blockerReasons.push('Exact canonical security exists but issuer/dossier identity is incomplete.');
 }else if(verifiedExisting){
  status='EXISTING_SECURITY_REUSE_CANDIDATE';
 }else if(sameIsinOtherListing.length>0){
  status='EXISTING_ISIN_OTHER_LISTING_REVIEW';
  blockerReasons.push('Same ISIN already registered on another listing; issuer re-use requires independent verification.');
 }else if(nameMatches.length>1){
  status='BLOCKED_NAME_AMBIGUOUS';
  blockerReasons.push('Multiple canonical issuer names match; no automatic identity linking.');
 }else if(nameMatches.length===1){
  status='EXISTING_ISSUER_NAME_REVIEW';
  blockerReasons.push('Matching issuer name is only a hint; legal entity linkage is unverified.');
 }else{
  if(!effectiveName)blockerReasons.push('No proposed display name found in Screener or supplied metadata.');
  if(!legacy)blockerReasons.push('No exact legacy Screener ticker/exchange match.');
  if(!evidenceUrl)blockerReasons.push('No issuer/exchange source reference supplied.');
  blockerReasons.push('Legal entity, ISIN and listing evidence has not been authenticated.');
 }
 const metadataSource=suppliedName?'OPERATOR_UNVERIFIED':
  legacy?'LEGACY_SCREENER_UNVERIFIED':'ABSENT';
 const proposedWrites:readonly string[]=verifiedExisting?[]:
  status.startsWith('BLOCKED')?[]:
  ['GOVERNED_REVIEW_ONLY:issuer_security_dossier_crosswalk'];
 // Deterministic idempotent diagnostic identity, NEVER create RPC key.
 const candidateFingerprintSha256=createHash('sha256').update(JSON.stringify({
  schema:'OROTITAN_P0C_V1',isin,ticker,exchange,effectiveName,
  legacyCompanyId:legacy?.id??null,
  canonicalSecurityId:exact?.securityId??null,
  evidenceUrl,
 })).digest('hex');
 return {
  format:'OROTITAN_CANONICAL_IDENTITY_ONBOARDING_CANDIDATE_V1',
  status,isin,ticker,exchange,
  proposedDisplayName:effectiveName,
  tradingCurrencyCandidate:legacy?.currency??null,
  existingIssuerId:exact?.issuerId??(nameMatches.length===1?nameMatches[0].issuerId:null),
  existingSecurityId:exact?.securityId??null,
  existingDossierId:activeDossiers.length===1?activeDossiers[0].dossierId:null,
  legacyCompanyId:legacy?.id??null,
  issuerNameMatchIds:nameMatches.map(i=>i.issuerId),
  evidenceUrl,evidenceStatus:evidenceUrl?'USER_SUPPLIED_POINTER_NOT_FETCHED':'NO_SOURCE_REFERENCE',
  metadataSource,
  needsIndependentLegalAndListingVerification:true,
  needsExchangeCurrencyQuoteUnitVerification:true,
  needsGovernedAtomicIdentityRpc:true,
  constraints:['NO_GLOBAL_ISIN_UNIQUE_INDEX','NO_CANONICAL_IDENTITY_ONBOARDING_RPC'],
  proposedWrites,blockerReasons,candidateFingerprintSha256,
  operationAuthorized:false,
  issuerIdentityAdmitted:false,securityIdentityAdmitted:false,
  researchDossierAdmitted:false,runCreated:false,
  rpcExecuted:false,registryMutated:false,productionTouched:false,
 };
}
