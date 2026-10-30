import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {ASML_Q3_SEC_SOURCES,ASML_Q3_SOURCE_IDS,retrieveAsmlQ3SecExhibit}
  from '../lib/orotitan-equity/product/asml-q3-official-sources-v1';
const source=ASML_Q3_SEC_SOURCES.SEC_EX_99_3;
const bytes=Buffer.from('<html>ASML 2026 US GAAP consolidated financial statements '+'financial results '.repeat(95)+'</html>');
function reply(data:Buffer=bytes,url=source.url,mime='text/html',status=200) {
 const r=new Response(new Uint8Array(data),{status,headers:{'content-type':mime}});
 Object.defineProperty(r,'url',{value:url});return r;
}
test('three exhibit IDs fixed and none accepts user supplied URLs',()=>{
 assert.deepEqual(ASML_Q3_SOURCE_IDS,['SEC_EX_99_1','SEC_EX_99_2','SEC_EX_99_3']);
 assert.ok(ASML_Q3_SOURCE_IDS.every(x=>ASML_Q3_SEC_SOURCES[x].url.startsWith(
  'https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/')));
});
test('official exhibit successful and sha verified',async()=>{
 const f=(async(url:string|URL|Request,options?:RequestInit)=>{
  assert.equal(String(url),source.url);
  assert.equal(options?.redirect,'error');
  assert.equal(options?.credentials,'omit');
  return reply();
 }) as typeof fetch;
 const result=await retrieveAsmlQ3SecExhibit('SEC_EX_99_3',f,'2026-10-30',
   ()=>new Date('2026-10-30T12:30:00Z'));
 assert.equal(result.candidate.byteSize,bytes.length);
 assert.equal(result.candidate.sha256,createHash('sha256').update(bytes).digest('hex'));
 assert.equal(result.candidate.admitted,false);
 assert.equal(result.candidate.includesLinkedMedia,false);
});
test('cutoff or source identity invalid -> no network',async()=>{
 let calls=0;
 const f=(async()=>{calls++;return reply();}) as typeof fetch;
 await assert.rejects(retrieveAsmlQ3SecExhibit('SEC_EX_99_3',f,'2026-10-29'),/TEMPORAL_GATE/);
 await assert.rejects(retrieveAsmlQ3SecExhibit('https://evil.example' as never,f),/ALLOWLISTED/);
 assert.equal(calls,0);
});
test('transport, mime, truncation, invalid identity are all fail closed',async()=>{
 const variants=[
  reply(bytes,source.url,'text/html',403),
  reply(bytes,'https://fake.sec.gov/filing'),
  reply(bytes,source.url,'application/pdf'),
  reply(Buffer.from('<html>short</html>')),
  reply(Buffer.from('A'.repeat(1200))),
 ];
 for(const response of variants) {
  await assert.rejects(retrieveAsmlQ3SecExhibit('SEC_EX_99_3',
    (async()=>response) as typeof fetch));
 }
});
