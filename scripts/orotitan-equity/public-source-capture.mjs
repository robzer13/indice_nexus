#!/usr/bin/env node
/* Capture public issuer/SEC HTTP entity bytes. NON-AUTHORITATIVE.
 * This tool never registers OroTitan Registry evidence or modifies Supabase.
 * SEC fair-access requests require a real monitored SEC_EDGAR_CONTACT.
 */
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import https from 'node:https';
import path from 'node:path';

export const MAX_BYTES = 30000000;
export function validateSourceUrl(input) {
  let u;
  try { u = new URL(input); } catch { throw Error('SOURCE_URL_INVALID'); }
  if (u.protocol !== 'https:' || u.port || u.username || u.password || u.search || u.hash) throw Error('SOURCE_URL_UNSAFE');
  if (u.hostname === 'www.sec.gov') {
    if (!/^\/Archives\/edgar\/data\/\d+\/\d+\/[A-Za-z0-9_.-]+$/.test(u.pathname)) throw Error('SEC_ARCHIVE_PATH_INVALID');
    return {url:u.href,issuer:'SEC',hostname:u.hostname};
  }
  if (u.hostname === 'www.asml.com') {
    if (!/^\/(en\/)?(company|investors|news)\//.test(u.pathname)) throw Error('ISSUER_PATH_INVALID');
    return {url:u.href,issuer:'ASML',hostname:u.hostname};
  }
  throw Error('HOST_NOT_ALLOWED');
}
export function checkOriginalResponse({status,contentType,contentEncoding,body,expectedMarkers=[]}) {
  if(status!==200) throw Error('NON_200_HTTP_RESPONSE');
  if(!/^text\/html(?:\s*;|$)/i.test(contentType??'')) throw Error('NON_HTML_CONTENT_TYPE');
  if(contentEncoding && contentEncoding.toLowerCase()!=='identity') throw Error('ENCODING_NOT_IDENTITY');
  if(!Buffer.isBuffer(body)||body.length<256||body.length>MAX_BYTES) throw Error('INVALID_DOCUMENT_SIZE');
  const sample=body.subarray(0,Math.min(body.length,1000000)).toString('utf8').toLowerCase();
  if(!/<(?:!doctype\s+html|html)(?:\s|>)/i.test(sample)) throw Error('NOT_HTML_DOCUMENT');
  if(/captcha|request rate threshold exceeded|access denied|<title>\s*(?:error|403)|automated tool/i.test(sample)) throw Error('BLOCK_OR_ERROR_RESPONSE');
  for(const marker of expectedMarkers)
    if(typeof marker!=='string'||marker.length<3||!sample.includes(marker.toLowerCase())) throw Error('EXPECTED_MARKER_MISSING');
  return {bytes:body.length,sha256:createHash('sha256').update(body).digest('hex')};
}
export function validatePublicationDate(date,cutoff) {
  if(!date) return {state:'NOT_ASSERTED',date:null};
  if(!/^\d{4}-\d{2}-\d{2}$/.test(date)||Number.isNaN(Date.parse(date+'T00:00:00Z'))||new Date(date+'T00:00:00Z').toISOString().slice(0,10)!==date) throw Error('BAD_PUBLICATION_DATE');
  if(cutoff && (!/^\d{4}-\d{2}-\d{2}$/.test(cutoff)||date>cutoff)) throw Error('POST_CUTOFF_OR_BAD_CUTOFF');
  return {state:'DECLARED_UNVERIFIED',date};
}
export function downloadHttpsOriginal(input,userAgent) {
  const source=validateSourceUrl(input);
  if(source.issuer==='SEC'&&(!userAgent||!/[^@\s]+@[^@\s]+\.[^@\s]+/.test(userAgent))) throw Error('SEC_REQUIRES_REAL_MONITORED_CONTACT');
  return new Promise((resolve,reject)=>{
    const req=https.get(source.url,{
      headers:{'user-agent':userAgent||'OroTitan-Public-Source-Capture/1.0','accept':'text/html','accept-encoding':'identity'},
      timeout:20000,maxHeaderSize:16000,
    },resp=>{
      if(resp.statusCode!==200) {resp.resume();reject(Error('HTTP_STATUS_'+resp.statusCode));return;}
      const chunks=[];let count=0;
      resp.on('data',chunk=>{count+=chunk.length;if(count>MAX_BYTES){req.destroy(Error('DOCUMENT_TOO_LARGE'));return;}chunks.push(chunk);});
      resp.on('end',()=>{try{
        const body=Buffer.concat(chunks);
        const validation=checkOriginalResponse({status:resp.statusCode,contentType:resp.headers['content-type'],contentEncoding:resp.headers['content-encoding'],body});
        resolve({source,body,validation,contentType:resp.headers['content-type'],etag:resp.headers.etag||null,lastModified:resp.headers['last-modified']||null});
      }catch(error){reject(error);}});
      resp.on('error',reject);
    });
    req.on('timeout',()=>req.destroy(Error('HTTPS_TIMEOUT')));
    req.on('error',reject);
  });
}
export async function capture({url,outDir,contact,declaredPublicationDate=null,cutoff=null}) {
  const {source,body,validation,contentType,etag,lastModified}=await downloadHttpsOriginal(url,contact);
  const declared=validatePublicationDate(declaredPublicationDate,cutoff);
  await mkdir(outDir,{recursive:true});
  const filename=validation.sha256+'.html';
  const filepath=path.join(outDir,filename);
  try {await writeFile(filepath,body,{flag:'wx',mode:0o600});}
  catch(error) {
    if(error?.code!=='EEXIST') throw error;
    if(!(await readFile(filepath)).equals(body)) throw Error('EXISTING_FILE_CONTENT_CONFLICT');
  }
  const receipt={
    schema:'OROTITAN_PUBLIC_BYTES_CAPTURE_V1_UNATTESTED',
    original_url:source.url,host:source.hostname,downloaded_at_utc:new Date().toISOString(),
    sha256:validation.sha256,bytes:validation.bytes,body_file:filename,
    content_type:contentType,response_etag:etag,response_last_modified:lastModified,
    declared_publication_date:declared.date,publication_date_evidence:declared.state,cutoff_if_supplied:cutoff,
    registry_admitted:false,registry_attestation_receipt:null,
    authority_state:'CAPTURE_ONLY_UNATTESTED',historical_byte_identity_at_cutoff:'NOT_PROVEN'
  };
  await writeFile(path.join(outDir,validation.sha256+'.receipt.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
  return receipt;
}
if(process.argv[1]&&path.resolve(process.argv[1])===path.resolve(new URL(import.meta.url).pathname)) {
  if(!process.env.SOURCE_URL) {console.error('SOURCE_URL required');process.exitCode=2;}
  else capture({
    url:process.env.SOURCE_URL,outDir:process.env.SOURCE_CAPTURE_DIR||'source-capture-output',
    contact:process.env.SEC_EDGAR_CONTACT,declaredPublicationDate:process.env.SOURCE_PUBLICATION_DATE||null,
    cutoff:process.env.DATA_CUTOFF||null
  }).then(receipt=>console.log(JSON.stringify({sha256:receipt.sha256,bytes:receipt.bytes,authority:receipt.authority_state})))
    .catch(error=>{console.error('SOURCE_CAPTURE_FAILED: '+error.message);process.exitCode=1;});
}
