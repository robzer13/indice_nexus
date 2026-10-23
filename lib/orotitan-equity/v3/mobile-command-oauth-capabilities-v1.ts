/**
 * D31 / DRAFT: independent authorization for ChatGPT mobile command INTENTS.
 *
 * Never exported as an MCP tool or imported by live routes. Neither signed
 * OAuth identity nor this server-private capability is Registry write authority.
 *
 * OAuth proves subject/client/resource. An independent server policy must then
 * approve the exact command. The preview is fetched from trusted private state;
 * never accept a model-supplied actor, draft, grant or HMAC signing key.
 *
 * The sealed capability MUST remain on the server (not returned to ChatGPT).
 * Persistent storage + transactional replay checks are separate release gates.
 */
import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import type { JWTVerifyGetKey } from "jose";
import {
  verifyMcpOAuthBearer, type McpOAuthConfig, type McpOAuthIdentity,
} from "./mcp-oauth-v1";
import {
  confirmMobileCommand, type CommandKind, type CommandReceipt,
  type DraftCommand, type MobileActor, type MobileCommandPrivatePort,
  MOBILE_PREVIEW_TTL_MS, MobileCommandError,
} from "./mobile-command-intents-candidate-v1";

const PURPOSE = "orotitan.mobile.intent.authorize.v1";
const HEX64 = /^[a-f0-9]{64}$/;
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/;
const HEX32 = /^[a-f0-9]{32}$/;
const B64 = /^[A-Za-z0-9_-]+$/;
const RESOURCE = "https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
type CapabilityData = {
  purpose: typeof PURPOSE; version: 1; previewId: string;
  kind: CommandKind; fingerprintSha256: string;
  subject: string; clientId: string; resource: typeof RESOURCE;
  issuedAtMs: number; expiresAtMs: number; nonce: string;
};
export type ServerPrivateCapability = Readonly<{sealed: string}>;
export type TrustedCommandAuthorization = Readonly<{
  actor: MobileActor; kind: CommandKind; previewId: string;
  fingerprintSha256: string;
}>;
export interface IndependentCommandPolicy {
  /** Server policy, NOT user_metadata, OIDC email/profile, or LLM text. */
  allows(command: TrustedCommandAuthorization): Promise<boolean>;
  /** Read from a private preview store, not from model-supplied fields. */
  readPreview(previewId: string): Promise<DraftCommand | null>;
}

export class CommandCapabilityError extends Error {
  constructor(readonly code: string) { super(code); }
}

function assertKey(key: Uint8Array) {
  if (!(key instanceof Uint8Array) || key.byteLength < 32 ||
    key.byteLength > 256) {
    throw new CommandCapabilityError("D31_SIGNING_KEY_NOT_CONFIGURED");
  }
}
function assertNow(now: number) {
  if (!Number.isSafeInteger(now) || now < 1_700_000_000_000) {
    throw new CommandCapabilityError("D31_TIME_INVALID");
  }
}
function actorFromVerified(identity: McpOAuthIdentity, cfg: McpOAuthConfig): MobileActor {
  if (cfg.audience !== RESOURCE ||
    !cfg.allowedSubjects.has(identity.subject) ||
    !cfg.allowedClients.has(identity.clientId) ||
    !identity.scopes.has("orotitan.read")) {
    throw new CommandCapabilityError("D31_OAUTH_IDENTITY_DENIED");
  }
  return {
    subject: identity.subject, clientId: identity.clientId,
    resource: RESOURCE,
  };
}
function isKind(value: unknown): value is CommandKind {
  return value === "add_company" || value === "request_research";
}
function sameActor(a: MobileActor, b: MobileActor) {
  return a.subject === b.subject && a.clientId === b.clientId &&
    a.resource === b.resource;
}
function validatePreview(p: DraftCommand | null,
  previewId: string, actor: MobileActor, now: number,
): DraftCommand {
  if (!p || !UUID.test(previewId) || p.previewId !== previewId ||
    !isKind(p.kind) || !sameActor(p.actor, actor) ||
    !HEX64.test(p.fingerprintSha256) ||
    !Number.isSafeInteger(p.expiresAtMs) ||
    now >= p.expiresAtMs || p.expiresAtMs - now > MOBILE_PREVIEW_TTL_MS) {
    throw new CommandCapabilityError("D31_PREVIEW_DENIED_OR_EXPIRED");
  }
  return p;
}
function hmac(body: string, key: Uint8Array): Buffer {
  return createHmac("sha256", key)
    .update(PURPOSE + "\x00" + body, "utf8").digest();
}
function encode(payload: CapabilityData, key: Uint8Array): ServerPrivateCapability {
  const body = Buffer.from(JSON.stringify(payload), "utf8").toString("base64url");
  return {sealed: body + "." + hmac(body, key).toString("base64url")};
}
function open(token: ServerPrivateCapability, key: Uint8Array, now: number): CapabilityData {
  const sealed = token?.sealed;
  if (!sealed || sealed.length > 2200) {
    throw new CommandCapabilityError("D31_CAPABILITY_INVALID");
  }
  const parts = sealed.split(".");
  if (parts.length !== 2 || parts[0].length > 1700 ||
    !B64.test(parts[0]) || !B64.test(parts[1])) {
    throw new CommandCapabilityError("D31_CAPABILITY_INVALID");
  }
  const supplied = Buffer.from(parts[1], "base64url");
  // Base64URL's unused trailing bits may admit distinct textual spellings
  // for identical decoded bytes. Reject every noncanonical representation.
  if (Buffer.from(parts[0], "base64url").toString("base64url") !== parts[0] ||
    supplied.toString("base64url") !== parts[1]) {
    throw new CommandCapabilityError("D31_CAPABILITY_SIGNATURE_DENIED");
  }
  const expected = hmac(parts[0],key);
  if (supplied.byteLength !== expected.byteLength ||
    !timingSafeEqual(supplied, expected)) {
    throw new CommandCapabilityError("D31_CAPABILITY_SIGNATURE_DENIED");
  }
  let p: unknown;
  try { p = JSON.parse(Buffer.from(parts[0], "base64url").toString("utf8")); }
  catch { throw new CommandCapabilityError("D31_CAPABILITY_INVALID"); }
  if (!p || typeof p !== "object" || Array.isArray(p)) {
    throw new CommandCapabilityError("D31_CAPABILITY_INVALID");
  }
  const d = p as Record<string,unknown>;
  if (Object.keys(d).sort().join(",") !==
    ["clientId","expiresAtMs","fingerprintSha256","issuedAtMs","kind",
      "nonce","previewId","purpose","resource","subject","version"].sort().join(",") ||
    d.purpose !== PURPOSE || d.version !== 1 || !isKind(d.kind) ||
    typeof d.previewId !== "string" || !UUID.test(d.previewId) ||
    typeof d.fingerprintSha256 !== "string" || !HEX64.test(d.fingerprintSha256) ||
    typeof d.subject !== "string" || typeof d.clientId !== "string" ||
    d.resource !== RESOURCE || typeof d.nonce !== "string" ||
    !HEX32.test(d.nonce) || !Number.isSafeInteger(d.issuedAtMs) ||
    !Number.isSafeInteger(d.expiresAtMs) ||
    (d.expiresAtMs as number) <= (d.issuedAtMs as number) ||
    (d.expiresAtMs as number) - (d.issuedAtMs as number) > MOBILE_PREVIEW_TTL_MS ||
    now < (d.issuedAtMs as number) || now >= (d.expiresAtMs as number)) {
    throw new CommandCapabilityError("D31_CAPABILITY_EXPIRED_OR_INVALID");
  }
  return d as CapabilityData;
}
async function authorize(policy: IndependentCommandPolicy,
  preview: DraftCommand, actor: MobileActor,
) {
  if (!await policy.allows({
    actor, kind: preview.kind, previewId: preview.previewId,
    fingerprintSha256: preview.fingerprintSha256,
  })) throw new CommandCapabilityError("D31_INDEPENDENT_AUTHORIZATION_DENIED");
}

/**
 * The preview is loaded from a private server store, then approved against a
 * separate server-side policy. This token is NOT a custom OAuth write scope.
 * The caller must never echo it to the model or expose it in logs.
 */
export async function issueServerPrivateCommandCapability(args: {
  bearer: string; oauth: McpOAuthConfig; keys: JWTVerifyGetKey;
  previewId: string; policy: IndependentCommandPolicy;
  signingKey: Uint8Array; nowMs?: number;
}): Promise<ServerPrivateCapability> {
  assertKey(args.signingKey);
  const now = args.nowMs ?? Date.now();
  assertNow(now);
  const verified = await verifyMcpOAuthBearer(args.bearer,args.oauth,args.keys);
  const actor = actorFromVerified(verified,args.oauth);
  const preview = validatePreview(
    await args.policy.readPreview(args.previewId),args.previewId,actor,now);
  await authorize(args.policy,preview,actor);
  return encode({
    purpose:PURPOSE,version:1,previewId:preview.previewId,
    kind:preview.kind,fingerprintSha256:preview.fingerprintSha256,
    subject:actor.subject,clientId:actor.clientId,resource:RESOURCE,
    issuedAtMs:now,expiresAtMs:Math.min(preview.expiresAtMs,now+MOBILE_PREVIEW_TTL_MS),
    nonce:randomBytes(16).toString("hex"),
  },args.signingKey);
}

/** Revalidate JWT, independent policy, private preview and capability at
 * confirmation time. A trusted durable confirmation port (D30) is still
 * mandatory; this module never accesses the Registry or Supabase directly.
 */
export async function confirmWithServerPrivateCapability(args: {
  bearer: string; oauth: McpOAuthConfig; keys: JWTVerifyGetKey;
  token: ServerPrivateCapability; signingKey: Uint8Array;
  previewId: string; kind: CommandKind; fingerprintSha256: string;
  confirmation: "CONFIRM"; policy: IndependentCommandPolicy;
  queue: MobileCommandPrivatePort; nowMs?: number;
}): Promise<CommandReceipt> {
  assertKey(args.signingKey);
  const now = args.nowMs ?? Date.now();
  assertNow(now);
  const verified = await verifyMcpOAuthBearer(args.bearer,args.oauth,args.keys);
  const actor = actorFromVerified(verified,args.oauth);
  const cap = open(args.token,args.signingKey,now);
  if (cap.subject !== actor.subject || cap.clientId !== actor.clientId ||
    cap.resource !== actor.resource || cap.previewId !== args.previewId ||
    cap.kind !== args.kind || cap.fingerprintSha256 !== args.fingerprintSha256) {
    throw new CommandCapabilityError("D31_CAPABILITY_BINDING_DENIED");
  }
  const preview = validatePreview(
    await args.policy.readPreview(args.previewId),args.previewId,actor,now);
  if (preview.kind !== args.kind ||
    preview.fingerprintSha256 !== args.fingerprintSha256 ||
    preview.expiresAtMs !== cap.expiresAtMs) {
    throw new CommandCapabilityError("D31_PREVIEW_CHANGED");
  }
  await authorize(args.policy,preview,actor);
  try {
    return await confirmMobileCommand(args.kind,{
      previewId:args.previewId,fingerprintSha256:args.fingerprintSha256,
      confirmation:args.confirmation,
    },actor,args.queue,now);
  } catch(e) {
    if (e instanceof MobileCommandError) throw e;
    throw new CommandCapabilityError("D31_PRIVATE_QUEUE_UNAVAILABLE");
  }
}
