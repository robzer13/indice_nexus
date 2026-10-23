/**
 * D38 / draft-only: session-bound attestations for sensitive preview/vault/queue
 * operations. Not exported as MCP tools and not connected to production.
 *
 * Server policy MUST be independent of token verification and check the
 * precise preview and operation against canonical private data.
 */
import {createHash, createHmac, randomBytes} from "node:crypto";
import type {JWTVerifyGetKey} from "jose";
import {verifyMcpOAuthBearer,type McpOAuthConfig} from "./mcp-oauth-v1";
import type {MobileActor} from "./mobile-command-intents-candidate-v1";

export type D38Operation =
  "create_preview"|"approve_vault"|"read_vault"|
  "confirm_company"|"confirm_research";
export type D38Proof = Readonly<{
  claims:Readonly<{
    version:"d38v1";operation:D38Operation;previewId:string;
    subject:string;clientId:string;resource:string;sessionId:string;
    bindingSha256:string;issuedAt:string;expiresAt:string;nonce:string;
  }>;macSha256:string;
}>;
export class D38AuthorizationError extends Error {
 constructor(readonly code:string){super(code);}
}
const RESOURCE="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const UUID=/^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i;
const SHA=/^[a-f0-9]{64}$/;
const SUBJECT=/^[a-zA-Z0-9_.:@|-]{4,160}$/;
const CLIENT=/^[a-zA-Z0-9_-]{6,160}$/;
const ZERO_SHA="0".repeat(64);
type Claims=D38Proof["claims"];
function canonical(c:Claims):string{
 return ["orotitan.d38.bound.v1",c.version,c.operation,c.previewId,
  c.subject,c.clientId,c.resource,c.sessionId,c.bindingSha256,
  c.issuedAt,c.expiresAt,c.nonce].join("\n");
}
export function d38Digest(value:string):string{
 return createHash("sha256").update(value,"utf8").digest("hex");
}
export function d38PreviewBinding(kind:"add_company"|"request_research",
 fingerprintSha256:string,expiresMs:number):string{
 if(!SHA.test(fingerprintSha256)||!Number.isSafeInteger(expiresMs)||
   expiresMs<1_700_000_000_000)throw new D38AuthorizationError("D38_PREVIEW_BINDING_INVALID");
 return d38Digest([kind,fingerprintSha256,String(expiresMs)].join("\n"));
}
export function d38VaultBinding(input:{
 kind:"add_company"|"request_research";fingerprintSha256:string;
 capabilitySha256:string;keyVersion:string;
 nonce:Uint8Array;ciphertext:Uint8Array;authTag:Uint8Array;
}):string {
 if(!SHA.test(input.fingerprintSha256)||!SHA.test(input.capabilitySha256)||
  !/^[a-z0-9_-]{1,40}$/.test(input.keyVersion)||
  input.nonce.length!==12||input.authTag.length!==16||
  input.ciphertext.length<50||input.ciphertext.length>4000)
   throw new D38AuthorizationError("D38_VAULT_BINDING_INVALID");
 return d38Digest([input.kind,input.fingerprintSha256,
   input.capabilitySha256,input.keyVersion,
   Buffer.from(input.nonce).toString("hex"),
   Buffer.from(input.ciphertext).toString("hex"),
   Buffer.from(input.authTag).toString("hex")].join("\n"));
}
export function d38CapabilityBinding(sealed:string):string{
 if(typeof sealed!=="string"||sealed.length>2200||
   !/^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{43}$/.test(sealed))
  throw new D38AuthorizationError("D38_CAPABILITY_BINDING_INVALID");
 return d38Digest(sealed);
}
export interface D38SessionPolicy {
 isSessionActive(query:{subject:string;sessionId:string}):Promise<boolean>;
 allows(query:{actor:MobileActor;sessionId:string;operation:D38Operation;
  previewId:string;bindingSha256:string}):Promise<boolean>;
}
export async function issueD38BoundProof(args:{
 bearer:string;oauth:McpOAuthConfig;keys:JWTVerifyGetKey;
 operation:D38Operation;previewId:string;bindingSha256:string;
 privateMacKey:Uint8Array;policy:D38SessionPolicy;nowMs?:number;
}):Promise<D38Proof> {
 const now=args.nowMs??Date.now();
 if(!Number.isSafeInteger(now)||now<1_700_000_000_000||
   !(args.privateMacKey instanceof Uint8Array)||
   args.privateMacKey.length!==32||args.oauth.audience!==RESOURCE||
   !UUID.test(args.previewId)||!SHA.test(args.bindingSha256)||
   !["create_preview","approve_vault","read_vault",
    "confirm_company","confirm_research"].includes(args.operation)||
   (args.operation==="read_vault" && args.bindingSha256!==ZERO_SHA)){
   throw new D38AuthorizationError("D38_INPUT_DENIED");
 }
 const verified=await verifyMcpOAuthBearer(args.bearer,args.oauth,args.keys);
 if(!SUBJECT.test(verified.subject)||!CLIENT.test(verified.clientId)||
   !verified.sessionId||!UUID.test(verified.sessionId)||
   !verified.scopes.has("orotitan.read"))
  throw new D38AuthorizationError("D38_VERIFIED_SESSION_REQUIRED");
 const actor:MobileActor={subject:verified.subject,
  clientId:verified.clientId,resource:RESOURCE};
 if(!await args.policy.isSessionActive({
   subject:verified.subject,sessionId:verified.sessionId})||
   !await args.policy.allows({actor,sessionId:verified.sessionId,
     operation:args.operation,previewId:args.previewId,
     bindingSha256:args.bindingSha256}))
    throw new D38AuthorizationError("D38_REVOKED_OR_POLICY_DENIED");
 const sec=Math.floor(now/1000);
 const claims:Claims={version:"d38v1",operation:args.operation,
   previewId:args.previewId,subject:actor.subject,clientId:actor.clientId,
   resource:actor.resource,sessionId:verified.sessionId,
   bindingSha256:args.bindingSha256,issuedAt:String(sec),
   expiresAt:String(sec+30),nonce:randomBytes(16).toString("hex")};
 return {claims,macSha256:createHmac("sha256",args.privateMacKey)
  .update(canonical(claims),"utf8").digest("hex")};
}
