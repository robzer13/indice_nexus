import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { validateSourceUrl, checkOriginalResponse, validatePublicationDate } from '../scripts/orotitan-equity/public-source-capture.mjs';

const content=Buffer.from(('<!doctype html><html><head><title>Issuer</title></head><body>Original issuer report fiscal 2025</body></html>').repeat(12));
test('issuer and SEC archive public URL allowlist',()=>{
  assert.equal(validateSourceUrl('https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/R10.htm').issuer,'SEC');
  assert.equal(validateSourceUrl('https://www.asml.com/en/company/governance/shareholders/shareholder-meetings/agm-2026').issuer,'ASML');
});
test('reject SSRF URL, credentials, query, insecure protocol and nonstandard port',()=>{
  const urls=['http://www.sec.gov/Archives/edgar/data/1/2/R1.htm','https://sec.gov/Archives/edgar/data/1/2/R1.htm',
    'https://www.sec.gov.evil.example/Archives/edgar/data/1/2/R1.htm','https://www.sec.gov@evil.example/x',
    'https://www.sec.gov:8443/Archives/edgar/data/1/2/R1.htm','https://127.0.0.1/private','file:///etc/passwd',
    'https://www.sec.gov/Archives/edgar/data/1/2/R1.htm?redirect=1'];
  for(const url of urls) assert.throws(()=>validateSourceUrl(url));
});
test('hashes original bytes without replacing their representation',()=>{
  const x=checkOriginalResponse({status:200,contentType:'text/html; charset=utf-8',contentEncoding:'identity',body:content,expectedMarkers:['issuer report','2025']});
  assert.equal(x.bytes,content.length);
  assert.equal(x.sha256,createHash('sha256').update(content).digest('hex'));
});
test('fail closed on error page, non-html, missing marker and blocked response',()=>{
  assert.throws(()=>checkOriginalResponse({status:403,contentType:'text/html',body:content}));
  assert.throws(()=>checkOriginalResponse({status:200,contentType:'application/pdf',body:content}));
  assert.throws(()=>checkOriginalResponse({status:200,contentType:'text/html',body:Buffer.alloc(999)}));
  assert.throws(()=>checkOriginalResponse({status:200,contentType:'text/html',body:Buffer.from(('<html>ACCESS DENIED').repeat(30))}));
  assert.throws(()=>checkOriginalResponse({status:200,contentType:'text/html',body:content,expectedMarkers:['unrelated financial statement']}));
});
test('publication metadata is declared only, and post-cutoff dates rejected',()=>{
  assert.deepEqual(validatePublicationDate('2026-02-25','2026-09-21'),{state:'DECLARED_UNVERIFIED',date:'2026-02-25'});
  assert.deepEqual(validatePublicationDate(null,'2026-09-21'),{state:'NOT_ASSERTED',date:null});
  assert.throws(()=>validatePublicationDate('2026-09-28','2026-09-21'));
  assert.throws(()=>validatePublicationDate('2026-02-31','2026-09-21'));
});

test('validates issuer-hosted public PDF source and binary body bytes',()=>{
  assert.equal(validateSourceUrl('https://ourbrand.asml.com/m/3136300aa4999bc1/original/2026_01_28_Presentation-Investor-Relations-Q4-2025.pdf').issuer,'ASML_PDF');
  const pdf=Buffer.from('%PDF-1.7\n'+('PDF binary payload not HTML\n').repeat(45)+'%%EOF\n');
  const out=checkOriginalResponse({status:200,contentType:'application/pdf',body:pdf});
  assert.equal(out.sha256,createHash('sha256').update(pdf).digest('hex'));
  assert.throws(()=>checkOriginalResponse({status:200,contentType:'application/pdf',body:content}));
  assert.throws(()=>validateSourceUrl('https://ourbrand.asml.com/private/internals.pdf'));
});
