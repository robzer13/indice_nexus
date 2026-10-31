/**
 * ASML Research v3.5 — non-authoritative in-browser reconciliation accumulator.
 * No network, no storage, no Registry access, no admission/certification.
 * Input is the already authenticated, integrity-checked v3.4 scan_pages response.
 */
export type V35Probe=Readonly<{
 claimId:string;numeric:string;derived:boolean;contextHint:string|null;
}>;
export type V35Source=Readonly<{
 id:string;title:string;sourceSha256:string;probes:readonly V35Probe[];
}>;
export type V35NumericCandidate=Readonly<{
 claimId:string;excerpt:string;match:'NUMERIC_TOKEN_ONLY';
}>;
export type V35Page=Readonly<{
 pdfPage:number;textLength:number;textSha256:string;
 candidateMatches:readonly V35NumericCandidate[];
}>;
export type V35Chunk=Readonly<{
 sourceId:string;sourceSha256:string;receiptSha256:string;checkedAt:string;
 sourceDate:string;captureTimestamp:string;
 pdfPageCount:number;startPage:number;endPage:number;
 pages:readonly V35Page[];
}>;
export type V35Observation=Readonly<{
 sourceId:string;sourceSha256:string;pdfPage:number;pageTextSha256:string;
 numeric:string;excerpt:string;
 numericMatch:'NUMERIC_TOKEN_ONLY';
 contextHint:string|null;
 contextHintObservation:'HINT_PRESENT'|'HINT_ABSENT'|'HINT_NOT_CONFIGURED';
}>;
export type V35ClaimRow=Readonly<{
 claimId:string;sourceId:string;sourceSha256:string;numericProbe:string;
 directDocumentLookup:boolean;
 searchStatus:'DERIVED_RECONCILIATION_REQUIRED'|'NUMERIC_CANDIDATES_PRESENT'|
  'NO_NUMERIC_MATCH_IN_EXTRACTED_TEXT'|'PARTIAL_SCAN';
 observations:readonly V35Observation[];
 validated:false;registryAdmitted:false;
}>;
export type V35Coverage=Readonly<{
 sourceId:string;sourceSha256:string;pdfPageCount:number|null;
 scannedPages:number;complete:boolean;unextractableTextPages:number;
}>;
export type V35Matrix=Readonly<{
 complete:boolean;coveredDocuments:number;totalDocuments:number;
 claimCount:number;validatedClaimCount:0;registryAdmitted:false;
 coverage:readonly V35Coverage[];claims:readonly V35ClaimRow[];
}>;

function record(x:unknown):x is Record<string,unknown>{
 return typeof x==='object'&&x!==null&&!Array.isArray(x);
}
function integer(x:unknown):x is number{return Number.isSafeInteger(x);}
function sha(x:unknown):x is string{
 return typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
}
function fail():never{throw Error('V35_SCAN_RESPONSE_INVALID');}

/** Fail closed on changed source identity/hash, truncated pages or unexpected claim IDs. */
export function validateV35Chunk(input:unknown,expected:V35Source,expectedStart:number):V35Chunk{
 if(!record(input)||input.ok!==true||input.mode!=='scan_pages'||
    input.registryAdmitted!==false||input.issuerOriginAuthenticated!==false||
    input.numericMatchesDoNotValidateClaims!==true||
    input.visualPageReviewStillRequired!==true||
    !record(input.source)||!record(input.scan))fail();
 const source=input.source,scan=input.scan;
 if(source.id!==expected.id||source.sourceOriginalHash!==expected.sourceSha256||
    source.integrity!=='STORED_BYTES_SHA256_MATCH'||!sha(source.receiptHash)||
    source.registryAdmitted!==false||source.historicalBytesAtCutoffCertified!==false||
    source.independentlyAuthenticatedPublisher!==false||
    typeof source.sourceDate!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(source.sourceDate)||
    typeof source.captureTimestamp!=='string'||!Number.isFinite(Date.parse(source.captureTimestamp))||
    scan.id!==expected.id||!integer(scan.pdfPageCount)||scan.pdfPageCount<1||
    scan.pdfPageCount>2000||scan.startPage!==expectedStart||
    !integer(scan.endPage)||scan.endPage<expectedStart||
    scan.endPage>Math.min(scan.pdfPageCount,expectedStart+19)||
    !Array.isArray(scan.pageObservations)||
    scan.pageObservations.length!==scan.endPage-expectedStart+1||
    typeof input.checkedAt!=='string'||!Number.isFinite(Date.parse(input.checkedAt)))fail();
 const claims=new Set(expected.probes.map(p=>p.claimId));
 const pages:V35Page[]=[];
 for(let i=0;i<scan.pageObservations.length;i++){
  const page:unknown=scan.pageObservations[i];
  if(!record(page)||page.pdfPage!==expectedStart+i||
     !integer(page.textLength)||page.textLength<0||!sha(page.textSha256)||
     !Array.isArray(page.candidateMatches))fail();
  const matches:V35NumericCandidate[]=[];
  for(const m of page.candidateMatches){
   if(!record(m)||typeof m.claimId!=='string'||!claims.has(m.claimId)||
      typeof m.excerpt!=='string'||m.excerpt.length>300||
      m.match!=='NUMERIC_TOKEN_ONLY')fail();
   matches.push({claimId:m.claimId,excerpt:m.excerpt,match:'NUMERIC_TOKEN_ONLY'});
  }
  pages.push({pdfPage:page.pdfPage,textLength:page.textLength,
    textSha256:page.textSha256,candidateMatches:matches});
 }
 return {sourceId:expected.id,sourceSha256:expected.sourceSha256,
   receiptSha256:source.receiptHash,checkedAt:input.checkedAt,
   sourceDate:source.sourceDate,captureTimestamp:source.captureTimestamp,
   pdfPageCount:scan.pdfPageCount,startPage:expectedStart,
   endPage:scan.endPage,pages};
}
export function buildV35Matrix(sources:readonly V35Source[],chunks:readonly V35Chunk[]):V35Matrix{
 const coverage:V35Coverage[]=[];
 const claims:V35ClaimRow[]=[];
 for(const src of sources){
  const own=chunks.filter(c=>c.sourceId===src.id)
   .slice().sort((a,b)=>a.startPage-b.startPage);
  let expectedPage=1;
  let pageCount:number|null=null;
  let unextractableTextPages=0;
  for(const c of own){
   if(c.sourceSha256!==src.sourceSha256||c.startPage!==expectedPage||
      (pageCount!==null&&c.pdfPageCount!==pageCount))fail();
   pageCount=c.pdfPageCount;
   expectedPage=c.endPage+1;
   unextractableTextPages+=c.pages.filter(p=>p.textLength===0).length;
  }
  const scannedPages=expectedPage-1;
  const complete=pageCount!==null&&scannedPages===pageCount;
  coverage.push({sourceId:src.id,sourceSha256:src.sourceSha256,pdfPageCount:pageCount,
    scannedPages,complete,unextractableTextPages});
  for(const probe of src.probes){
   const observations:V35Observation[]=[];
   if(!probe.derived){
    for(const c of own)for(const page of c.pages)for(const m of page.candidateMatches){
     if(m.claimId!==probe.claimId)continue;
     const hint=probe.contextHint;
     observations.push({
      sourceId:src.id,sourceSha256:src.sourceSha256,
      pdfPage:page.pdfPage,pageTextSha256:page.textSha256,numeric:probe.numeric,
      excerpt:m.excerpt,numericMatch:'NUMERIC_TOKEN_ONLY',
      contextHint:hint,
      contextHintObservation:hint===null?'HINT_NOT_CONFIGURED':
       m.excerpt.toLocaleLowerCase('en-US').includes(hint.toLocaleLowerCase('en-US'))?
        'HINT_PRESENT':'HINT_ABSENT',
     });
    }
   }
   const searchStatus=probe.derived?'DERIVED_RECONCILIATION_REQUIRED':
    observations.length?'NUMERIC_CANDIDATES_PRESENT':
    complete?'NO_NUMERIC_MATCH_IN_EXTRACTED_TEXT':'PARTIAL_SCAN';
   claims.push({claimId:probe.claimId,sourceId:src.id,sourceSha256:src.sourceSha256,
     numericProbe:probe.numeric,directDocumentLookup:!probe.derived,
     searchStatus,observations,validated:false,registryAdmitted:false});
  }
 }
 return {complete:coverage.every(c=>c.complete),coveredDocuments:coverage.filter(c=>c.complete).length,
   totalDocuments:sources.length,claimCount:claims.length,validatedClaimCount:0,
   registryAdmitted:false,coverage,claims};
}
