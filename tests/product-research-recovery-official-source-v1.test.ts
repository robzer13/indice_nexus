import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {
  ASML_Q2_2026_PUBLIC_RELEASE,ASML_Q2_2026_OFFICIAL_PRESENTATION,
  probeAsmlQ2OfficialRelease,retrievePreferredAsmlQ2OfficialSource,
} from '../lib/orotitan-equity/product/research-recovery-official-source-v1';

const q2='<html><title>ASML Q2 2026 financial results</title><main>'
  +'Press release - July 15, 2026. ASML published its 2026 second-quarter results. '
  +'Issuer official statement for the 2026 second quarter. '.repeat(8)+'</main></html>';

function sourceReply(body=q2, status=200, type='text/html', url:string=ASML_Q2_2026_PUBLIC_RELEASE.url) {
  const response=new Response(body,{status,headers:{'content-type':type}});
  Object.defineProperty(response,'url',{value:url});
  return response;
}
test('official pre-cutoff issuer release produces a CONTENT candidate, not admission',async()=>{
  let called=0;
  const fetcher=(async (url:string|URL|Request,options?:RequestInit)=>{
    called++;
    assert.equal(String(url),ASML_Q2_2026_PUBLIC_RELEASE.url);
    assert.equal(options?.redirect,'error');
    assert.equal(options?.credentials,'omit');
    assert.equal(options?.cache,'no-store');
    assert.equal(options?.method,'GET');
    return sourceReply();
  }) as typeof fetch;
  const result=await probeAsmlQ2OfficialRelease(fetcher,'2026-09-21',
    ()=>new Date('2026-10-30T09:00:00Z'));
  assert.equal(called,1);
  assert.equal(result.publicationDate,'2026-07-15');
  assert.equal(result.retrievedAt,'2026-10-30T09:00:00.000Z');
  assert.equal(result.admitted,false);
  assert.equal(result.historicalBytesCertified,false);
  assert.equal(result.contentType,'text/html');
  assert.equal(result.byteSize,Buffer.byteLength(q2));
  assert.equal(result.sha256,createHash('sha256').update(q2).digest('hex'));
});
test('pre-publication, future-cutoff and malformed dates fail BEFORE any network',async()=>{
  let calls=0;
  const f=(async()=>{calls++;return sourceReply();}) as typeof fetch;
  for (const date of ['2026-07-14','2026-10-14','2030-01-01','bad-date']) {
    await assert.rejects(probeAsmlQ2OfficialRelease(f,date),/RECOVERY_SOURCE_TEMPORAL_GATE/);
  }
  assert.equal(calls,0);
});
test('redirect, changed origin, HTTP errors and non-HTML are rejected',async()=>{
  const cases=[
    [sourceReply(q2,302),'RECOVERY_SOURCE_ORIGIN_UNVERIFIED'],
    [sourceReply(q2,200,'text/html','https://example.com/evil'),'RECOVERY_SOURCE_ORIGIN_UNVERIFIED'],
    [sourceReply(q2,200,'application/pdf'),'RECOVERY_SOURCE_CONTENT_TYPE'],
    [sourceReply(q2,404),'RECOVERY_SOURCE_ORIGIN_UNVERIFIED'],
  ] as const;
  for(const [response,reason] of cases) {
    await assert.rejects(probeAsmlQ2OfficialRelease((async()=>response) as typeof fetch),
      new RegExp(reason));
  }
});
test('malformed official contents, oversized bodies or advertised lengths fail closed',async()=>{
  await assert.rejects(probeAsmlQ2OfficialRelease(
    (async()=>sourceReply('<html>ASML 2025 only '+'.'.repeat(400)+'</html>')) as typeof fetch),
    /RECOVERY_SOURCE_IDENTITY_OR_DATE_UNVERIFIED/);
  await assert.rejects(probeAsmlQ2OfficialRelease(
    (async()=>sourceReply(q2+' '.repeat(2_001_000))) as typeof fetch),
    /RECOVERY_SOURCE_TOO_LARGE/);
  const r=sourceReply();
  r.headers.set('content-length','99999999');
  await assert.rejects(probeAsmlQ2OfficialRelease((async()=>r) as typeof fetch),
    /RECOVERY_SOURCE_TOO_LARGE/);
});
test('network failures never create a success-shaped candidate',async()=>{
  await assert.rejects(probeAsmlQ2OfficialRelease(
    (async()=>{throw Error('network unavailable');}) as typeof fetch),/network unavailable/);
});

const pdfBytes=Buffer.concat([
  Buffer.from('%PDF-1.7\n'),
  Buffer.alloc(15000,0x20),
  Buffer.from('\n%%EOF\n'),
]);
function pdfResponse(body:Buffer=pdfBytes,url=ASML_Q2_2026_OFFICIAL_PRESENTATION.url){
  const r=new Response(new Uint8Array(body),{status:200,headers:{'content-type':'application/pdf'}});
  Object.defineProperty(r,'url',{value:url});
  return r;
}
test('403 on investor HTML triggers only the pinned issuer-owned PDF, preserving attribution',async()=>{
  const visited:string[]=[];
  const f=(async(url:string|URL|Request,options?:RequestInit)=>{
    visited.push(String(url));
    assert.equal(options?.redirect,'error');
    return visited.length===1?sourceReply(q2,403):pdfResponse();
  }) as typeof fetch;
  const result=await retrievePreferredAsmlQ2OfficialSource(f,'2026-09-21',
    ()=>new Date('2026-10-30T11:00:00Z'));
  assert.deepEqual(visited,[ASML_Q2_2026_PUBLIC_RELEASE.url,ASML_Q2_2026_OFFICIAL_PRESENTATION.url]);
  assert.equal(result.candidate.contentType,'application/pdf');
  assert.equal(result.candidate.documentId,ASML_Q2_2026_OFFICIAL_PRESENTATION.documentId);
  assert.equal(result.candidate.url,ASML_Q2_2026_OFFICIAL_PRESENTATION.url);
  assert.equal(result.candidate.admitted,false);
  assert.equal(result.candidate.historicalBytesCertified,false);
  assert.equal(result.candidate.sha256,createHash('sha256').update(pdfBytes).digest('hex'));
});
test('PDF fallback rejects truncated, missing signature and redirected content',async()=>{
  const invalid=[
    pdfResponse(Buffer.from('%PDF-1.7 invalid short %%EOF')),
    pdfResponse(Buffer.concat([Buffer.from('BAD-'),Buffer.alloc(11000),Buffer.from('%%EOF')])),
    pdfResponse(pdfBytes,'https://external.example/asset.pdf'),
  ];
  for(const lastResponse of invalid) {
    const f=(async(url:string|URL|Request)=>{
      return String(url)===ASML_Q2_2026_PUBLIC_RELEASE.url
        ? sourceReply(q2,403):lastResponse;
    }) as typeof fetch;
    await assert.rejects(retrievePreferredAsmlQ2OfficialSource(f),
      /RECOVERY_SOURCE_PDF_/);
  }
});
test('PDF fallback cannot bypass pre-cutoff temporal gate or issuer URL origin',async()=>{
  let calls=0;
  const f=(async()=>{calls++;return pdfResponse();}) as typeof fetch;
  await assert.rejects(retrievePreferredAsmlQ2OfficialSource(f,'2026-10-14'),
    /RECOVERY_SOURCE_TEMPORAL_GATE/);
  assert.equal(calls,0);
});
