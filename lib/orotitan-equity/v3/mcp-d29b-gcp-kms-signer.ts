/**
 * D29b REVIEW-ONLY: Google Cloud KMS EC_SIGN_ED25519 (SOFTWARE) signing adapter.
 * No credentials, private keys, GCP IAM roles, or endpoints are installed here.
 * The admin runtime must inject an IAM-authorized short-lived access token.
 * This adapter is not imported by the public ChatGPT/OAuth MCP route.
 *
 * Protocol: Google Cloud KMS cryptoKeyVersions.asymmetricSign (raw `data`,
 * dataCrc32c, verifiedDataCrc32c, signatureCrc32c).
 * https://cloud.google.com/kms/docs/reference/rest/v1/projects.locations.keyRings.cryptoKeys.cryptoKeyVersions/asymmetricSign
 */
import { createPublicKey, verify } from "node:crypto";
import type { D29bSigningAdapter } from "./mcp-d29b-d2-issuer-server";

const VERSION = /^projects\/[a-z][a-z0-9-]{4,28}\/(?:locations\/[a-z0-9-]+)\/keyRings\/[A-Za-z0-9_-]+\/cryptoKeys\/[A-Za-z0-9_-]+\/cryptoKeyVersions\/[1-9][0-9]*$/;
const KEY_ID=/^[A-Za-z0-9_-]{1,64}$/;
const MAX_MESSAGE_BYTES=24000;

export class D29bKmsError extends Error {
  constructor(readonly code:string){super(code);}
}
function fail(code:string):never {throw new D29bKmsError(code);}
export function crc32c(bytes:Uint8Array):number {
  let crc=0xffffffff;
  for(const byte of bytes) {
    crc^=byte;
    for(let i=0;i<8;i++) crc=(crc>>>1)^((crc&1)?0x82f63b78:0);
  }
  return (crc^0xffffffff)>>>0;
}
type KmsResponse={
  signature?:unknown;
  signatureCrc32c?:unknown;
  verifiedDataCrc32c?:unknown;
  name?:unknown;
  protectionLevel?:unknown;
};
type KmsFetch=(
  input:string,init:{method:string;headers:Record<string,string>;body:string},
)=>Promise<{ok:boolean;status:number;json:()=>Promise<unknown>}>;
export type GcpD29bSignerOptions={
  keyId:string;versionName:string;pinnedPublicKeySpkiPem:string;
  /** Returns a short-lived IAM access token, e.g. via workload identity. */
  accessToken:()=>Promise<string>;
  /** Server-only HTTPS implementation; injectable for deterministic tests. */
  fetcher:KmsFetch;
};

/** The code uses Ed25519, which Google Cloud KMS currently offers under
 * SOFTWARE protection only. This is a managed KMS key, not Cloud HSM.
 */
export function createGcpD29bEd25519Signer(
  cfg:GcpD29bSignerOptions,
):D29bSigningAdapter {
  if(!KEY_ID.test(cfg.keyId) || !VERSION.test(cfg.versionName) ||
    cfg.pinnedPublicKeySpkiPem.length>2000) fail("D29B_GCP_KMS_CONFIGURATION_INVALID");
  let pinnedKey:ReturnType<typeof createPublicKey>;
  try {
    pinnedKey=createPublicKey(cfg.pinnedPublicKeySpkiPem);
    if(pinnedKey.asymmetricKeyType!=="ed25519") fail("D29B_GCP_KMS_KEY_NOT_ED25519");
  } catch {fail("D29B_GCP_KMS_KEY_NOT_ED25519");}
  const url="https://cloudkms.googleapis.com/v1/"+cfg.versionName+":asymmetricSign";
  return {
    keyId:cfg.keyId,
    sign:async(data:Uint8Array):Promise<Uint8Array>=>{
      if(!(data instanceof Uint8Array) || !data.byteLength ||
        data.byteLength>MAX_MESSAGE_BYTES) fail("D29B_GCP_KMS_DATA_INVALID");
      const token=await cfg.accessToken();
      if(!token || token.length<16 || token.length>9000 ||
        !/^[A-Za-z0-9._~-]+$/.test(token)) fail("D29B_GCP_KMS_IDENTITY_UNAVAILABLE");
      const message=Buffer.from(data);
      // Raw data, NOT SHA256 digest: Ed25519 PureEdDSA signs the exact bytes.
      let response;
      try {
        response=await cfg.fetcher(url,{
          method:"POST",
          headers:{
            Authorization:"Bearer "+token,
            "Content-Type":"application/json",
          },
          body:JSON.stringify({
            data:message.toString("base64"),
            dataCrc32c:String(crc32c(message)),
          }),
        });
      } catch {fail("D29B_GCP_KMS_NETWORK_UNAVAILABLE");}
      if(!response.ok) fail("D29B_GCP_KMS_SIGN_REFUSED");
      let value:unknown;
      try {value=await response.json();}catch{fail("D29B_GCP_KMS_RESPONSE_INVALID");}
      if(!value || typeof value!=="object" || Array.isArray(value))
        fail("D29B_GCP_KMS_RESPONSE_INVALID");
      const parsed=value as KmsResponse;
      if(parsed.name!==cfg.versionName ||
        parsed.verifiedDataCrc32c!==true ||
        parsed.protectionLevel!=="SOFTWARE" ||
        typeof parsed.signature!=="string" ||
        typeof parsed.signatureCrc32c!=="string" ||
        !/^[A-Za-z0-9+/]+={0,2}$/.test(parsed.signature))
        fail("D29B_GCP_KMS_RESPONSE_UNTRUSTED");
      const signature=Buffer.from(parsed.signature,"base64");
      if(signature.length!==64 ||
        signature.toString("base64")!==parsed.signature ||
        String(crc32c(signature))!==parsed.signatureCrc32c)
        fail("D29B_GCP_KMS_SIGNATURE_CORRUPTED");
      if(!verify(null,message,pinnedKey,signature))
        fail("D29B_GCP_KMS_PINNED_PUBLIC_KEY_MISMATCH");
      return new Uint8Array(signature);
    },
  };
}
