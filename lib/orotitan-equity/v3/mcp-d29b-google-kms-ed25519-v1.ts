/**
 * D29b Google Cloud KMS Ed25519 signing adapter — REVIEW ONLY.
 * Only usable in the separately authenticated D2 issuer runtime.
 * Never import this module into OAuth MCP or browser code.
 *
 * IAM token is provided by workload identity in the trusted runtime;
 * no service-account JSON key, refresh token, HMAC or private key is accepted.
 * The issuer independently verifies the returned signature with pinned SPKI.
 */
import type { D29bSigningAdapter } from "./mcp-d29b-d2-issuer-server";
import { crc32c } from "./mcp-d29b-crc32c-v1";

export type KmsFetchResponse = {
  ok:boolean;
  status:number;
  json:()=>Promise<unknown>;
};
export type D29bKmsTransport={
  accessToken:()=>Promise<string>;
  request:(url:string,request:{
    method:"POST";
    headers:Record<string,string>;
    body:string;
    signal?:AbortSignal;
  })=>Promise<KmsFetchResponse>;
};
export class D29bKmsError extends Error {
  constructor(readonly code:string){super(code);}
}
function deny(code:string):never {throw new D29bKmsError(code);}
const VERSION_PATH=/^projects\/[a-z][a-z0-9-]{4,62}\/locations\/[a-z0-9-]+\/keyRings\/[A-Za-z0-9_-]{1,63}\/cryptoKeys\/[A-Za-z0-9_-]{1,63}\/cryptoKeyVersions\/[1-9][0-9]{0,8}$/;
const B64=/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
const CRC32C_DECIMAL=/^(?:0|[1-9][0-9]{0,9})$/;

export function createD29bGoogleKmsEd25519Adapter(
  input:{keyId:string;cryptoKeyVersion:string},
  transport:D29bKmsTransport,
):D29bSigningAdapter {
  if(!/^[A-Za-z0-9_-]{1,64}$/.test(input.keyId) ||
    !VERSION_PATH.test(input.cryptoKeyVersion) ||
    typeof transport?.accessToken!=="function" ||
    typeof transport?.request!=="function")
    deny("D29B_KMS_CONFIGURATION_INVALID");
  const endpoint="https://cloudkms.googleapis.com/v1/"+
    input.cryptoKeyVersion+":asymmetricSign";
  return {
    keyId:input.keyId,
    sign:async(message:Uint8Array):Promise<Uint8Array>=>{
      if(!(message instanceof Uint8Array) ||
        message.byteLength<30 || message.byteLength>25000)
        deny("D29B_KMS_MESSAGE_INVALID");
      // Ed25519 is PureEdDSA: sign RAW message bytes, never SHA256 digest.
      const accessToken=await transport.accessToken();
      if(typeof accessToken!=="string" || accessToken.length<40 ||
        accessToken.length>8192 || !/^[A-Za-z0-9._~+\/-]+$/.test(accessToken))
        deny("D29B_KMS_IDENTITY_UNAVAILABLE");
      // Google KMS verifies the raw data checksum and returns one for the
      // signature. Fail closed on network timeout or integrity mismatch.
      // This does not retry: the issuer owns backoff and retry budget.
      const expectedName=input.cryptoKeyVersion;
      const raw=Buffer.from(message);
      let response:KmsFetchResponse;
      try {
        response=await transport.request(endpoint,{
          method:"POST",
          headers:{
            Authorization:"Bearer "+accessToken,
            "Content-Type":"application/json",
          },
          body:JSON.stringify({
            data:raw.toString("base64"),dataCrc32c:String(crc32c(raw)),
          }),
          signal:AbortSignal.timeout(8000),
        });
      } catch {deny("D29B_KMS_REQUEST_FAILED");}
      if(!response.ok || response.status!==200)
        deny("D29B_KMS_SIGNATURE_UNAVAILABLE");
      let body:unknown;
      try {body=await response.json();}
      catch {deny("D29B_KMS_RESPONSE_INVALID");}
      if(!body || typeof body!=="object" || Array.isArray(body) ||
        !("signature" in body) ||
        typeof body.signature!=="string" ||
        body.signature.length!==88 || !B64.test(body.signature))
        deny("D29B_KMS_SIGNATURE_INVALID");
      const responseData=body as Record<string,unknown>;
      // Pin the exact key version and software protection level. The KMS
      // response MUST acknowledge the data CRC32C check we requested.
      if(responseData.name!==expectedName ||
        responseData.verifiedDataCrc32c!==true ||
        responseData.protectionLevel!=="SOFTWARE")
        deny("D29B_KMS_KEY_OR_REQUEST_INTEGRITY_INVALID");
      const encoded=responseData.signature as string;
      const signature=Buffer.from(encoded,"base64");
      if(signature.length!==64 || signature.toString("base64")!==encoded)
        deny("D29B_KMS_SIGNATURE_INVALID");
      const sigCrc=responseData.signatureCrc32c;
      if(typeof sigCrc!=="string" || !CRC32C_DECIMAL.test(sigCrc) ||
        Number(sigCrc)>0xffffffff ||
        crc32c(signature)!==Number(sigCrc))
        deny("D29B_KMS_SIGNATURE_INTEGRITY_INVALID");
      return signature;
    },
  };
}
