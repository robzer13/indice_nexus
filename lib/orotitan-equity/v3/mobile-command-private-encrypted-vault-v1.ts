/** D34 REVIEW ONLY. AES-256-GCM envelope for a D31 HMAC-sealed capability.
 * The encryption key stays on the trusted application server. PostgreSQL
 * stores nonce/ciphertext/tag/keyVersion only, never plaintext/token/key.
 */
import { createCipheriv, createDecipheriv, createHash, randomBytes, timingSafeEqual } from "node:crypto";
import type { CommandKind, MobileActor } from "./mobile-command-intents-candidate-v1";
import type { ServerPrivateCapability } from "./mobile-command-oauth-capabilities-v1";

export type D34VaultContext = Readonly<{
  previewId: string; kind: CommandKind; actor: MobileActor;
  fingerprintSha256: string;
}>;
export type D34EncryptedEnvelope = Readonly<{
  keyVersion: string; nonce: string; ciphertext: string; authTag: string;
}>;
export class D34VaultError extends Error {
  constructor(readonly code: string) {super(code);}
}
const UUID=/^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i;
const HEX64=/^[a-f0-9]{64}$/;
const B64=/^[A-Za-z0-9+/]+={0,2}$/;

function validateContext(c:D34VaultContext) {
  if(!c|| !UUID.test(c.previewId) || !HEX64.test(c.fingerprintSha256) ||
    (c.kind!=="add_company"&&c.kind!=="request_research") ||
    !c.actor || !c.actor.subject || !c.actor.clientId ||
    c.actor.resource!=="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp") {
      throw new D34VaultError("D34_CONTEXT_INVALID");
  }
}
function assertKey(key:Uint8Array) {
  if(!(key instanceof Uint8Array)||key.length!==32)
    throw new D34VaultError("D34_VAULT_AES_KEY_REQUIRED");
}
function aad(c:D34VaultContext):Buffer{
  // Stable array layout binds AES-GCM to actor, client, action, exact preview.
  return Buffer.from(JSON.stringify(["orotitan.d34.vault.v1",c.previewId,
    c.kind,c.actor.subject,c.actor.clientId,c.actor.resource,
    c.fingerprintSha256]),"utf8");
}
function canonicalB64(value:string,expectedSize?:number):Buffer{
  if(typeof value!=="string"||value.length>6000||!B64.test(value))
    throw new D34VaultError("D34_ENVELOPE_FORMAT_DENIED");
  const b=Buffer.from(value,"base64");
  if(b.toString("base64")!==value ||
    (expectedSize!==undefined&&b.length!==expectedSize))
    throw new D34VaultError("D34_ENVELOPE_FORMAT_DENIED");
  return b;
}
export function sealD34ServerCapability(args:{
  token:ServerPrivateCapability;context:D34VaultContext;
  key:Uint8Array;keyVersion:string;
}):Readonly<{envelope:D34EncryptedEnvelope;capabilitySha256:string}>{
  validateContext(args.context);assertKey(args.key);
  if(!/^[a-z0-9_-]{1,40}$/.test(args.keyVersion)||
    !args.token?.sealed||args.token.sealed.length>2200)
    throw new D34VaultError("D34_ENCRYPTION_INPUT_DENIED");
  const iv=randomBytes(12);
  const cipher=createCipheriv("aes-256-gcm",args.key,iv);
  cipher.setAAD(aad(args.context));
  const bytes=Buffer.concat([cipher.update(args.token.sealed,"utf8"),cipher.final()]);
  const tag=cipher.getAuthTag();
  return {
    envelope:{
      keyVersion:args.keyVersion,
      nonce:iv.toString("base64"),
      ciphertext:bytes.toString("base64"),
      authTag:tag.toString("base64"),
    },
    capabilitySha256:createHash("sha256").update(args.token.sealed).digest("hex"),
  };
}
export function openD34ServerCapability(args:{
  envelope:D34EncryptedEnvelope;context:D34VaultContext;
  key:Uint8Array;expectedSha256:string;expectedKeyVersion:string;
}):ServerPrivateCapability{
  validateContext(args.context);assertKey(args.key);
  if(!args.envelope||args.envelope.keyVersion!==args.expectedKeyVersion||
    !HEX64.test(args.expectedSha256))
    throw new D34VaultError("D34_KEY_VERSION_OR_DIGEST_DENIED");
  const iv=canonicalB64(args.envelope.nonce,12);
  const bytes=canonicalB64(args.envelope.ciphertext);
  const tag=canonicalB64(args.envelope.authTag,16);
  if(bytes.length<50||bytes.length>4000)
    throw new D34VaultError("D34_CIPHERTEXT_LENGTH_DENIED");
  try{
    const decipher=createDecipheriv("aes-256-gcm",args.key,iv);
    decipher.setAAD(aad(args.context));decipher.setAuthTag(tag);
    const decrypted=Buffer.concat([decipher.update(bytes),decipher.final()]);
    const plaintext=decrypted.toString("utf8");
    if(Buffer.from(plaintext,"utf8").compare(decrypted)!==0)
      throw new D34VaultError("D34_NONUTF8_DENIED");
    const calculated=createHash("sha256").update(decrypted).digest();
    const expected=Buffer.from(args.expectedSha256,"hex");
    if(expected.length!==calculated.length||!timingSafeEqual(expected,calculated))
      throw new D34VaultError("D34_CAPABILITY_DIGEST_DENIED");
    if(!/^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{43}$/.test(plaintext))
      throw new D34VaultError("D34_CAPABILITY_FORMAT_DENIED");
    return {sealed:plaintext};
  }catch(e){
    if(e instanceof D34VaultError)throw e;
    throw new D34VaultError("D34_DECRYPTION_DENIED");
  }
}
