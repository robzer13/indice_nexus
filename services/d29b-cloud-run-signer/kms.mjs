/** Private Cloud Run KMS transport. No keys or credentials in files/env.
 * Cloud Run ADC access tokens come from its attached, least-privilege service
 * identity via the instance metadata endpoint, never a JSON key.
 */
import { createPublicKey, verify } from "node:crypto";
import { KEY_VERSION } from "./policy.mjs";

const META_URL="http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";
const KMS_URL="https://cloudkms.googleapis.com/v1/"+KEY_VERSION+":asymmetricSign";
const B64=/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
export class KmsTransportError extends Error {
  constructor(code){super(code);this.code=code;}
}
const reject=code=>{throw new KmsTransportError(code);};
export function crc32c(bytes) {
  let crc=0xffffffff;
  for(const byte of bytes) {
    crc^=byte;
    for(let i=0;i<8;i++)crc=(crc>>>1)^((crc&1)?0x82f63b78:0);
  }
  return (crc^0xffffffff)>>>0;
}
async function readLimitedJson(response,maxBytes) {
  if(!response.ok)reject("UPSTREAM_STATUS");
  if(response.redirected)reject("UPSTREAM_REDIRECT");
  const contentType=response.headers?.get?.("content-type")??"";
  if(!contentType.includes("json"))reject("UPSTREAM_CONTENT_TYPE");
  if(!response.body)reject("UPSTREAM_EMPTY_BODY");
  const chunks=[];let size=0;
  for await(const chunk of response.body) {
    const buf=Buffer.from(chunk);size+=buf.length;
    if(size>maxBytes)reject("UPSTREAM_TOO_LARGE");
    chunks.push(buf);
  }
  try{return JSON.parse(Buffer.concat(chunks).toString("utf8"));}
  catch{reject("UPSTREAM_INVALID_JSON");}
}
export async function signWithCloudKms(message,policy,transport=fetch) {
  if(!Buffer.isBuffer(message) || message.length<30 || message.length>25000)
    reject("MESSAGE_LENGTH");
  let publicKey;
  try{
    publicKey=createPublicKey(policy.publicKeyPem);
    if(publicKey.asymmetricKeyType!=="ed25519")reject("KEY_NOT_ED25519");
  }catch{reject("INVALID_PINNED_PUBLIC_KEY");}

  let tokenBody;
  try{
    const meta=await transport(META_URL,{
      method:"GET",headers:{"Metadata-Flavor":"Google"},
      redirect:"error",signal:AbortSignal.timeout(2500),
    });
    tokenBody=await readLimitedJson(meta,4096);
  }catch{reject("METADATA_UNAVAILABLE");}
  const accessToken=tokenBody?.access_token;
  if(typeof accessToken!=="string" || accessToken.length<40 ||
    accessToken.length>8192 || !/^[A-Za-z0-9._~+\/-]+$/.test(accessToken))
    reject("METADATA_TOKEN_INVALID");

  let body;
  try{
    const response=await transport(KMS_URL,{
      method:"POST",
      headers:{"Authorization":"Bearer "+accessToken,
        "Content-Type":"application/json"},
      body:JSON.stringify({data:message.toString("base64"),
        dataCrc32c:String(crc32c(message))}),
      redirect:"error",signal:AbortSignal.timeout(8000),
    });
    body=await readLimitedJson(response,4096);
  }catch{reject("KMS_REQUEST_FAILED");}
  if(!body || typeof body!=="object" ||
    body.name!==policy.keyVersion || body.verifiedDataCrc32c!==true ||
    body.protectionLevel!=="SOFTWARE" ||
    typeof body.signature!=="string" || body.signature.length!==88 ||
    !B64.test(body.signature))reject("KMS_RESPONSE_NOT_ADMITTED");
  const signature=Buffer.from(body.signature,"base64");
  if(signature.length!==64 ||
    signature.toString("base64")!==body.signature ||
    typeof body.signatureCrc32c!=="string" ||
    !/^(?:0|[1-9][0-9]{0,9})$/.test(body.signatureCrc32c) ||
    Number(body.signatureCrc32c)>0xffffffff ||
    crc32c(signature)!==Number(body.signatureCrc32c) ||
    !verify(null,message,publicKey,signature))
    reject("KMS_SIGNATURE_INTEGRITY_INVALID");
  return signature;
}
