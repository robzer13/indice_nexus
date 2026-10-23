/**
 * D30 — mobile / ChatGPT command-intent candidate. REVIEW ONLY.
 *
 * This file is NOT imported by the live MCP route. It deliberately cannot
 * write the Screener, canonical issuer/security tables, or Registry runs.
 * "add_company" means a CONFIRMED ONBOARDING REQUEST, not a new issuer.
 * "request_research" means a CONFIRMED QUEUE REQUEST, not execution.
 *
 * The authenticated MCP server must supply the actor. Never trust actor IDs
 * supplied by model arguments, browser content, user_metadata or filenames.
 */
import { createHash, randomUUID } from "node:crypto";

export class MobileCommandError extends Error {
  constructor(readonly code: string) { super(code); }
}
export type MobileActor = Readonly<{
  subject: string; clientId: string; resource: string;
}>;
export type CompanyCandidate = Readonly<{
  name: string; ticker: string; exchange: string; isin: string;
}>;
export type ResearchTarget = Readonly<{
  issuerId: string; displayName: string; ticker: string; exchange: string;
  isin: string; eligibleForRequest: boolean;
}>;
export type CommandKind = "add_company" | "request_research";
export type DraftCommand = Readonly<{
  previewId: string; kind: CommandKind; actor: MobileActor;
  payload: Readonly<Record<string, string>>;
  fingerprintSha256: string; expiresAtMs: number;
}>;
export type CommandReceipt = Readonly<{
  requestId: string; kind: CommandKind; fingerprintSha256: string;
  state: "REQUESTED_NOT_EXECUTED";
  confirmed: true; registryRunId: null; registryMutationPerformed: false;
  companyInserted: false; researchStarted: false; duplicate: boolean;
}>;

/** This is an authorization/data boundary, not a general purpose DAO.
 * The production replacement must atomically recheck actor, identity, stale
 * preview, duplicate issuer and request eligibility under an audited DB role.
 */
export interface MobileCommandPrivatePort {
  authorize(actor: MobileActor): Promise<boolean>;
  lookupCompany(candidate: CompanyCandidate): Promise<
    "NEW_UNAMBIGUOUS" | "ALREADY_TRACKED" | "AMBIGUOUS"
  >;
  lookupResearchTarget(issuerId: string): Promise<ResearchTarget | null>;
  storePreview(preview: DraftCommand): Promise<void>;
  atomicConfirm(args: {
    previewId: string; actor: MobileActor;
    fingerprintSha256: string; nowMs: number;
  }): Promise<CommandReceipt>;
}

export const MOBILE_PREVIEW_TTL_MS = 2 * 60_000;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const ISIN = /^[A-Z]{2}[A-Z0-9]{9}[0-9]$/;
const RESOURCE = "https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const HASH = /^[0-9a-f]{64}$/;

function assertActor(actor: MobileActor): void {
  if (!actor || typeof actor.subject !== "string" ||
    !/^[a-zA-Z0-9_.:@|-]{4,160}$/.test(actor.subject) ||
    typeof actor.clientId !== "string" ||
    !/^[a-zA-Z0-9_-]{6,160}$/.test(actor.clientId) ||
    actor.resource !== RESOURCE) {
    throw new MobileCommandError("MOBILE_ACTOR_SCOPE_DENIED");
  }
}
function assertTime(now: number): void {
  if (!Number.isSafeInteger(now) || now < 1_700_000_000_000) {
    throw new MobileCommandError("MOBILE_CLOCK_INVALID");
  }
}
async function authorize(port: MobileCommandPrivatePort, actor: MobileActor) {
  assertActor(actor);
  if (!await port.authorize(actor)) {
    throw new MobileCommandError("MOBILE_COMMAND_NOT_AUTHORIZED");
  }
}
function fingerprint(kind: CommandKind, payload: Readonly<Record<string, string>>) {
  // Canonical field ordering, not a caller-provided hash.
  const pairs = Object.keys(payload).sort().map(k => [k, payload[k]]);
  return createHash("sha256").update(JSON.stringify([kind, pairs])).digest("hex");
}
function normalizeCompany(candidate: CompanyCandidate): CompanyCandidate {
  if (!candidate || typeof candidate.name !== "string" ||
    typeof candidate.ticker !== "string" ||
    typeof candidate.exchange !== "string" ||
    typeof candidate.isin !== "string") {
    throw new MobileCommandError("MOBILE_COMPANY_IDENTITY_INCOMPLETE");
  }
  const name = candidate.name.trim().replace(/\s+/g, " ");
  const ticker = candidate.ticker.trim().toUpperCase();
  const exchange = candidate.exchange.trim().toUpperCase();
  const isin = candidate.isin.trim().toUpperCase();
  if (name.length < 2 || name.length > 160 ||
    !/^[A-Z0-9][A-Z0-9.\-]{0,19}$/.test(ticker) ||
    !/^[A-Z0-9][A-Z0-9_\-]{1,39}$/.test(exchange) ||
    !ISIN.test(isin)) {
    throw new MobileCommandError("MOBILE_COMPANY_IDENTITY_INVALID");
  }
  return { name, ticker, exchange, isin };
}
function previewPayload(kind: CommandKind, payload: Readonly<Record<string, string>>,
  actor: MobileActor, nowMs: number): DraftCommand {
  return {
    previewId: randomUUID(), kind,
    actor: {subject: actor.subject, clientId: actor.clientId, resource: actor.resource},
    payload: Object.freeze({...payload}),
    fingerprintSha256: fingerprint(kind, payload),
    expiresAtMs: nowMs + MOBILE_PREVIEW_TTL_MS,
  };
}
function assertReceipt(r: CommandReceipt, kind: CommandKind, sha: string): void {
  if (!r || !UUID.test(r.requestId) || r.kind !== kind ||
    r.fingerprintSha256 !== sha || !HASH.test(r.fingerprintSha256) ||
    r.state !== "REQUESTED_NOT_EXECUTED" || r.confirmed !== true ||
    r.registryRunId !== null || r.registryMutationPerformed !== false ||
    r.companyInserted !== false || r.researchStarted !== false ||
    typeof r.duplicate !== "boolean") {
    throw new MobileCommandError("MOBILE_UNTRUSTED_CONFIRM_RESULT");
  }
}

/** Tool candidate: preview_add_company. No write or request is created. */
export async function previewAddCompany(
  input: CompanyCandidate, actor: MobileActor,
  port: MobileCommandPrivatePort, nowMs: number = Date.now(),
): Promise<DraftCommand> {
  assertTime(nowMs);
  await authorize(port, actor);
  const normalized = normalizeCompany(input);
  const status = await port.lookupCompany(normalized);
  if (status === "AMBIGUOUS") {
    throw new MobileCommandError("MOBILE_IDENTITY_NEEDS_REVIEW");
  }
  if (status === "ALREADY_TRACKED") {
    throw new MobileCommandError("MOBILE_COMPANY_ALREADY_TRACKED");
  }
  if (status !== "NEW_UNAMBIGUOUS") {
    throw new MobileCommandError("MOBILE_IDENTITY_LOOKUP_UNAVAILABLE");
  }
  const preview = previewPayload("add_company", normalized, actor, nowMs);
  await port.storePreview(preview);
  return preview;
}

/** Tool candidate: preview_request_research. A queue intent, not Registry. */
export async function previewRequestResearch(
  issuerId: string, actor: MobileActor,
  port: MobileCommandPrivatePort, nowMs: number = Date.now(),
): Promise<DraftCommand> {
  assertTime(nowMs);
  await authorize(port, actor);
  if (typeof issuerId !== "string" || !UUID.test(issuerId)) {
    throw new MobileCommandError("MOBILE_ISSUER_ID_INVALID");
  }
  const resolved = await port.lookupResearchTarget(issuerId);
  if (!resolved || resolved.issuerId.toLowerCase() !== issuerId.toLowerCase() ||
    !resolved.eligibleForRequest) {
    throw new MobileCommandError("MOBILE_RESEARCH_NOT_ELIGIBLE");
  }
  const company = normalizeCompany({
    name: resolved.displayName, ticker: resolved.ticker,
    exchange: resolved.exchange, isin: resolved.isin,
  });
  const preview = previewPayload("request_research",
    {issuerId: resolved.issuerId.toLowerCase(), ...company}, actor, nowMs);
  await port.storePreview(preview);
  return preview;
}

/**
 * Tool candidates add_company and request_research only call this after the
 * operator explicitly confirms a fresh preview AND its fingerprint.
 * Backend must apply all checks again atomically at confirmation time.
 */
export async function confirmMobileCommand(
  kind: CommandKind,
  args: { previewId: string; fingerprintSha256: string; confirmation: "CONFIRM" },
  actor: MobileActor, port: MobileCommandPrivatePort,
  nowMs: number = Date.now(),
): Promise<CommandReceipt> {
  assertTime(nowMs);
  await authorize(port, actor);
  if (!args || args.confirmation !== "CONFIRM" ||
    !UUID.test(args.previewId) || !HASH.test(args.fingerprintSha256)) {
    throw new MobileCommandError("MOBILE_EXPLICIT_CONFIRMATION_REQUIRED");
  }
  // Caller does not send issuer status, Registry state or "authority".
  const result = await port.atomicConfirm({
    previewId: args.previewId, actor,
    fingerprintSha256: args.fingerprintSha256, nowMs,
  });
  assertReceipt(result, kind, args.fingerprintSha256);
  return result;
}
