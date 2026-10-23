/**
 * D33 / REVIEW ONLY: a server-only D31 -> D32 seam, not a deployed MCP tool.
 *
 * Independent OAuth verification and authorization precede a narrow private
 * database approval. Confirmations reuse D31 checks, then ONLY the sealed
 * server-held capability is sent to the D32 EXECUTE-only submitter procedure.
 * No public client or model may call these functions with arbitrary identities.
 */
import { createHash } from "node:crypto";
import type { JWTVerifyGetKey } from "jose";
import { verifyMcpOAuthBearer, type McpOAuthConfig } from "./mcp-oauth-v1";
import {
  issueServerPrivateCommandCapability, confirmWithServerPrivateCapability,
  CommandCapabilityError, type IndependentCommandPolicy,
  type ServerPrivateCapability,
} from "./mobile-command-oauth-capabilities-v1";
import type {
  CommandKind, CommandReceipt, MobileActor, MobileCommandPrivatePort,
} from "./mobile-command-intents-candidate-v1";

type ApprovedRow = Readonly<{
  requestId: string; kind: CommandKind;
  state: "REQUESTED_NOT_EXECUTED"; duplicate: boolean;
}>;
export interface D33PrivateDatabaseProcedures {
  /** The private approval issuer role only. Credentials and HMAC stay server-side. */
  approve(input: {
    previewId: string; kind: CommandKind; actor: MobileActor;
    fingerprintSha256: string; capabilitySha256: string;
  }): Promise<boolean>;
  /** The separate, EXECUTE-only private submitter role. */
  confirm(input: {previewId: string; sealedCapability: string}): Promise<ApprovedRow>;
}
type Context = {
  bearer: string; oauth: McpOAuthConfig; keys: JWTVerifyGetKey;
  previewId: string; policy: IndependentCommandPolicy;
  signingKey: Uint8Array; db: D33PrivateDatabaseProcedures;
  nowMs?: number;
};
const RESOURCE = "https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export async function issueAndRegisterD33PrivateApproval(
  args: Context,
): Promise<ServerPrivateCapability> {
  // Existing D31 verifies signed issuer/audience/subject/client and policy.
  const token = await issueServerPrivateCommandCapability(args);
  // Revalidate, so an unverified model-provided 'actor' is never passed to PG.
  const verified = await verifyMcpOAuthBearer(args.bearer, args.oauth, args.keys);
  const actor: MobileActor = {
    subject: verified.subject, clientId: verified.clientId,
    resource: args.oauth.audience,
  };
  const preview = await args.policy.readPreview(args.previewId);
  if (!preview || !UUID.test(preview.previewId) ||
    preview.previewId !== args.previewId ||
    preview.actor.subject !== actor.subject ||
    preview.actor.clientId !== actor.clientId ||
    preview.actor.resource !== actor.resource ||
    actor.resource !== RESOURCE ||
    !(await args.policy.allows({
      actor, previewId: preview.previewId, kind: preview.kind,
      fingerprintSha256: preview.fingerprintSha256,
    }))) {
    throw new CommandCapabilityError("D33_APPROVAL_CONTEXT_CHANGED");
  }
  const capabilitySha256 = createHash("sha256").update(token.sealed).digest("hex");
  try {
    await args.db.approve({
      previewId: preview.previewId, kind: preview.kind,
      actor, fingerprintSha256: preview.fingerprintSha256, capabilitySha256,
    });
  } catch {
    // No secret or database exception strings in model-visible failures.
    throw new CommandCapabilityError("D33_PRIVATE_APPROVAL_DENIED");
  }
  // SERVER-ONLY; never include token in MCP tool response.
  return token;
}

/** In D33 the D30 queue port is closed over the server-held secret. */
export async function confirmD33AuthorizedRequest(args: Context & {
  token: ServerPrivateCapability;
  kind: CommandKind; fingerprintSha256: string;
  confirmation: "CONFIRM";
  baseQueue: Omit<MobileCommandPrivatePort,"atomicConfirm">;
}): Promise<CommandReceipt> {
  const privateQueue: MobileCommandPrivatePort = {
    authorize: actor => args.baseQueue.authorize(actor),
    lookupCompany: c => args.baseQueue.lookupCompany(c),
    lookupResearchTarget: id => args.baseQueue.lookupResearchTarget(id),
    storePreview: p => args.baseQueue.storePreview(p),
    atomicConfirm: async ({previewId, kind, fingerprintSha256}) => {
      if (previewId !== args.previewId || kind !== args.kind ||
        fingerprintSha256 !== args.fingerprintSha256) {
        throw new CommandCapabilityError("D33_CONFIRM_BINDING_DENIED");
      }
      const row = await args.db.confirm({
        previewId, sealedCapability: args.token.sealed,
      });
      if (!row || !UUID.test(row.requestId) || row.kind !== kind ||
        row.state !== "REQUESTED_NOT_EXECUTED" ||
        typeof row.duplicate !== "boolean") {
        throw new CommandCapabilityError("D33_PRIVATE_RECEIPT_INVALID");
      }
      return {
        requestId:row.requestId,kind:row.kind,
        fingerprintSha256,state:row.state,duplicate:row.duplicate,
        confirmed:true,registryRunId:null,registryMutationPerformed:false,
        companyInserted:false,researchStarted:false,
      };
    },
  };
  return confirmWithServerPrivateCapability({
    bearer:args.bearer,oauth:args.oauth,keys:args.keys,
    token:args.token,signingKey:args.signingKey,
    previewId:args.previewId,kind:args.kind,
    fingerprintSha256:args.fingerprintSha256,
    confirmation:args.confirmation,policy:args.policy,
    queue:privateQueue,nowMs:args.nowMs,
  });
}
