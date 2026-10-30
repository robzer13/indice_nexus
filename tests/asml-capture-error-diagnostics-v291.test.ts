import test from 'node:test';
import assert from 'node:assert/strict';
import {ASML_V29_CUTOFF,ASML_V29_SOURCES}
  from '../lib/orotitan-equity/product/asml-official-pdf-capture-v29';
import {ASML_CAPTURE_ERROR_LABELS,asmlCaptureErrorLabel,classifyAsmlCaptureError}
  from '../lib/orotitan-equity/product/asml-capture-error-diagnostics-v291';

test('issuer HTTP status failures are classified without exposing raw errors',()=>{
  const statuses:[number,string][]=[
    [401,'SOURCE_ACCESS_DENIED'],[403,'SOURCE_ACCESS_DENIED'],
    [451,'SOURCE_ACCESS_DENIED'],[404,'SOURCE_NOT_FOUND'],
    [410,'SOURCE_NOT_FOUND'],[429,'SOURCE_RATE_LIMITED'],
    [500,'SOURCE_HTTP_FAILURE'],[503,'SOURCE_HTTP_FAILURE'],
  ];
  for(const [status,expected] of statuses){
    const error=new Error('ASML_V29_ISSUER_FETCH_HTTP_'+status);
    assert.equal(classifyAsmlCaptureError(error),expected);
    assert.ok(!asmlCaptureErrorLabel(expected).includes('ASML_V29_'));
  }
});
test('redirect, malformed PDF, size and private storage failures are separated',()=>{
  const cases:[string,string][]=[
    ['ASML_V29_REDIRECT_OUTSIDE_ALLOWLIST','SOURCE_REDIRECT_BLOCKED'],
    ['ASML_V29_TOO_MANY_REDIRECTS','SOURCE_REDIRECT_BLOCKED'],
    ['ASML_V29_UNEXPECTED_MIME','SOURCE_RESPONSE_INVALID'],
    ['ASML_V29_INVALID_PDF','SOURCE_RESPONSE_INVALID'],
    ['ASML_V29_CONTENT_LENGTH_INVALID','SOURCE_SIZE_LIMIT'],
    ['ASML_V29_BODY_EXCEEDS_LIMIT','SOURCE_SIZE_LIMIT'],
    ['ASML_V29_PRIVATE_READBACK_MISMATCH','STORAGE_FAILURE'],
    ['ASML_V29_PRIVATE_BUCKET_REQUIRED','STORAGE_FAILURE'],
    ['ASML_V29_UNKNOWN_SOURCE','SOURCE_SELECTION_INVALID'],
    ['ASML_V29_POST_CUTOFF_SOURCE','SOURCE_POST_CUTOFF'],
  ];
  for(const [raw,expected] of cases)
    assert.equal(classifyAsmlCaptureError(new Error(raw)),expected);
});
test('timeout and fetch errors are classified without transmitting secrets',()=>{
  const timeout=Object.assign(new Error('private endpoint timeout with token'),{name:'TimeoutError'});
  assert.equal(classifyAsmlCaptureError(timeout),'SOURCE_TIMEOUT');
  assert.equal(classifyAsmlCaptureError(new TypeError('fetch failed; secret=abc')),'SOURCE_NETWORK_FAILURE');
  for(const key of Object.keys(ASML_CAPTURE_ERROR_LABELS)){
    const label=asmlCaptureErrorLabel(key);
    assert.equal(typeof label,'string');
    assert.ok(!label.includes('secret=abc'));
    assert.ok(!label.includes('private endpoint'));
  }
});
test('arbitrary request query values and unrecognized errors fail closed',()=>{
  const defaultLabel=ASML_CAPTURE_ERROR_LABELS.CAPTURE_UNCLASSIFIED;
  for(const raw of [null,123,{},'__proto__','<img src=x>','ASML_V29_ISSUER_FETCH_HTTP_403']){
    assert.equal(asmlCaptureErrorLabel(raw),defaultLabel);
  }
  assert.equal(classifyAsmlCaptureError(new Error('https://bad.invalid/secret?token=abc')), 'CAPTURE_UNCLASSIFIED');
  assert.equal(classifyAsmlCaptureError('ASML_V29_PRIVATE_BUCKET_REQUIRED'),'CAPTURE_UNCLASSIFIED');
});
test('TSMC third-quarter source is publisher-hosted and before locked cutoff',()=>{
  const source=ASML_V29_SOURCES.TSMC_2026_Q3_CALL;
  assert.equal(source.id,'TSMC_2026_Q3_CALL');
  assert.equal(source.originHost,'investor.tsmc.com');
  assert.ok(source.url.startsWith('https://investor.tsmc.com/english/encrypt/files/encrypt_file/reports/2026-10/'));
  assert.ok(source.publicationDate<=ASML_V29_CUTOFF);
});
