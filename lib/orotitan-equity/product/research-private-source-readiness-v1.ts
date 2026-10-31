/**
 * Fail-closed readiness gate for issuer-neutral private primary-source scans.
 * Only records identity/receipt prerequisites. It never downloads PDF bytes,
 * authenticates issuers, validates claims or writes Registry artifacts.
 */
export type PrivatePdfCandidate=Readonly<{
 id:string;title:string;publicationDate:string;
 expectedSha256:string|null;expectedByteSize:number|null;
 privateStoragePath:string|null;receiptSha256:string|null;
 storedBytesIndependentlyRehashed:boolean;privateReceiptIndependentlyVerified:boolean;
 captureTimestamp:string|null;
}>;
export type PrivatePdfReadiness=Readonly<{
 id:string;state:'POST_CUTOFF_PUBLICATION'|'INVALID_SOURCE_METADATA'|
  'ORIGINAL_BYTES_MISSING'|'RECEIPT_MISSING'|'INTEGRITY_NOT_VERIFIED'|
  'SCAN_ELIGIBLE_PRE_ADMISSION';
 scanAllowed:boolean;historicalBytesAtCutoffCertified:false;
 issuerAuthenticated:false;claimValidated:false;registryAdmitted:false;
 capturedAfterCutoff:boolean|null;
}>;
const sha=(v:unknown):v is string=>
 typeof v==='string'&&/^[a-f0-9]{64}$/.test(v);
const validDay=(s:string):boolean=>
 /^\d{4}-\d{2}-\d{2}$/.test(s)&&
 Number.isFinite(Date.parse(s+'T00:00:00Z'))&&
 new Date(s+'T00:00:00Z').toISOString().slice(0,10)===s;
export function assessPrivatePdfSource(
 c:PrivatePdfCandidate,cutoff:string,
):PrivatePdfReadiness{
 if(!validDay(cutoff))throw Error('RESEARCH_INVALID_CUTOFF');
 let state:PrivatePdfReadiness['state'];
 if(!c||typeof c.id!=='string'||!/^[A-Z0-9_-]{3,80}$/.test(c.id)||
    typeof c.title!=='string'||!c.title.trim()||
    !validDay(c.publicationDate))state='INVALID_SOURCE_METADATA';
 else if(c.publicationDate>cutoff)state='POST_CUTOFF_PUBLICATION';
 else if(!sha(c.expectedSha256)||!Number.isSafeInteger(c.expectedByteSize)||
         c.expectedByteSize===null||c.expectedByteSize<100||
         c.expectedByteSize>40_000_000||
         typeof c.privateStoragePath!=='string'||
         !c.privateStoragePath.trim()||c.privateStoragePath.includes('..')||
         c.privateStoragePath.startsWith('/'))state='ORIGINAL_BYTES_MISSING';
 else if(!sha(c.receiptSha256)||
         c.captureTimestamp===null||
         !Number.isFinite(Date.parse(c.captureTimestamp)))state='RECEIPT_MISSING';
 else if(!c.storedBytesIndependentlyRehashed||
         !c.privateReceiptIndependentlyVerified)state='INTEGRITY_NOT_VERIFIED';
 else state='SCAN_ELIGIBLE_PRE_ADMISSION';
 const capturedAfterCutoff=c?.captureTimestamp&&
   Number.isFinite(Date.parse(c.captureTimestamp))?
   c.captureTimestamp.slice(0,10)>cutoff:null;
 return {id:c?.id??'UNKNOWN',state,
  scanAllowed:state==='SCAN_ELIGIBLE_PRE_ADMISSION',
  historicalBytesAtCutoffCertified:false,
  issuerAuthenticated:false,claimValidated:false,registryAdmitted:false,
  capturedAfterCutoff};
}
