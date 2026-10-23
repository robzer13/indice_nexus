/**
 * D37: review-only server attestation of an ALREADY VERIFIED OAuth session.
 * This is a boundary candidate, NOT an MCP write tool or a hosted credential.
 * The MAC key is never accepted from the browser/model or stored in the JWT.
 * SQL must verify MAC, exact scope, short TTL and unique nonce atomically.
 */
import {createHmac, randomBytes} from "node:crypto";
import type {JWTVerifyGetKey} from "jose";
import {
  verifyMcpOAuthBearer,type McpOAuthConfig,
} from "./mcp-oauth-v1";
import type {MobileActor} from "./mobile-command-intents-candidate-v1";

export type D37Operation = "read_preview"|"issue_research_scope";
export type D37Claims = Readonly<{
  version:"d37v1"; operation:D37Operation; previewId:string;
  subject:string; clientId:string;resource:string; sessionId:string;
  issuerId:string; contractSha256:string; cutoff:string;
  issuedAt:string; expiresAt:string; nonce:string;
}>;
export type D37ServerProof = Readonly<{claims:D37Claims;macSha256:string}>;
export class D37AuthorizationError extends Error {
  constructor(readonly code:string){super(code);}
}
const RESOURCE="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const UUID=/^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i;
const SHA=/^[a-f0-9]{64}$/;
const SUBJECT=/^[a-zA-Z0-9_.:@|-]{4,160}$/;
const CLIENT=/^[a-zA-Z0-9_-]{6,160}$/;
const ZERO_UUID="00000000-0000-0000-0000-000000000000";
const ZERO_SHA="0".repeat(64);
const DEFAULT_CUTOFF="1970-01-01";
function canonical(c:D37Claims):string {
 return [
 "orotitan.d37.session.v1",c.version,c.operation,c.previewId,
 c.subject,c.clientId,c.resource,c.sessionId,c.issuerId,
 c.contractSha256,c.cutoff,c.issuedAt,c.expiresAt,c.nonce,
 ].join("\n");
}
export function d37CanonicalProofBody(c:D37Claims):string {
 // Exported for independent test-vector checks; no secrets are included.
 return canonical(c);
}
export interface D37IndependentSessionPolicy {
  // Lookup in trusted server context; must verify session is still ACTIVE.
  isSessionActive(value:{subject:string;sessionId:string}):Promise<boolean>;
  // Independent authorization for the exact preview/operation/scope.
  allows(value:{
    actor:MobileActor;sessionId:string;operation:D37Operation;
    previewId:string;issuerId:string;contractSha256:string;cutoff:string;
  }):Promise<boolean>;
}
export async function issueD37OAuthSessionProof(args:{
  bearer:string; oauth:McpOAuthConfig;keys:JWTVerifyGetKey;
  operation:D37Operation;previewId:string;
  issuerId?:string;contractSha256?:string;cutoff?:string;
  sessionPolicy:D37IndependentSessionPolicy;
  privateMacKey:Uint8Array;nowMs?:number;
}):Promise<D37ServerProof> {
  const now=args.nowMs??Date.now();
  if(!Number.isSafeInteger(now)||now<1_700_000_000_000 ||
    !(args.privateMacKey instanceof Uint8Array) ||
    args.privateMacKey.byteLength!==32 ||
    args.oauth.audience!==RESOURCE || !UUID.test(args.previewId)) {
    throw new D37AuthorizationError("D37_CONFIGURATION_OR_PREVIEW_DENIED");
  }
  if(args.operation!=="read_preview"&&args.operation!=="issue_research_scope"){
    throw new D37AuthorizationError("D37_OPERATION_DENIED");
  }
  const id=await verifyMcpOAuthBearer(args.bearer,args.oauth,args.keys);
  if(!SUBJECT.test(id.subject)||!CLIENT.test(id.clientId)||
    !id.scopes.has("orotitan.read")||!id.sessionId||
    !UUID.test(id.sessionId)) {
    throw new D37AuthorizationError("D37_VERIFIED_SESSION_REQUIRED");
  }
  const scope=args.operation==="read_preview" ? {
    issuerId:ZERO_UUID,contractSha256:ZERO_SHA,cutoff:DEFAULT_CUTOFF,
  }:{
    issuerId:args.issuerId??"",contractSha256:args.contractSha256??"",
    cutoff:args.cutoff??"",
  };
  if(!UUID.test(scope.issuerId)||!SHA.test(scope.contractSha256)||
    !/^\d{4}-\d{2}-\d{2}$/.test(scope.cutoff)) {
    throw new D37AuthorizationError("D37_SCOPE_DENIED");
  }
  const parsed=new Date(scope.cutoff+"T00:00:00Z");
  if(Number.isNaN(parsed.getTime()) ||
     parsed.toISOString().slice(0,10)!==scope.cutoff) {
    throw new D37AuthorizationError("D37_CUTOFF_INVALID");
  }
  const actor:MobileActor={
    subject:id.subject,clientId:id.clientId,resource:RESOURCE,
  };
  if(!await args.sessionPolicy.isSessionActive({
    subject:id.subject,sessionId:id.sessionId
  }) || !await args.sessionPolicy.allows({
    actor,sessionId:id.sessionId,operation:args.operation,
    previewId:args.previewId,...scope,
  })) {
    throw new D37AuthorizationError("D37_INDEPENDENT_SESSION_OR_SCOPE_DENIED");
  }
  const nowSeconds=Math.floor(now/1000);
  const claims:D37Claims={
    version:"d37v1",operation:args.operation,previewId:args.previewId,
    subject:id.subject,clientId:id.clientId,resource:RESOURCE,
    sessionId:id.sessionId, ...scope,
    issuedAt:String(nowSeconds),expiresAt:String(nowSeconds+30),
    nonce:randomBytes(16).toString("hex"),
  };
  return {claims,macSha256:createHmac("sha256",args.privateMacKey)
    .update(canonical(claims),"utf8").digest("hex")};
}
