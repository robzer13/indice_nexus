/**
 * Isolated remote MCP access policy. No admin-cookie delegation and no
 * Supabase service-role credentials in the MCP response.
 *
 * Prototype uses a server-only bearer secret; no ChatGPT OAuth identity flow
 * is implied. Disabled until security review and a supported auth connection.
 */
import { createHash, timingSafeEqual } from "node:crypto";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function safeCompare(actual: string, expected: string): boolean {
  const a = createHash("sha256").update(actual).digest();
  const b = createHash("sha256").update(expected).digest();
  return timingSafeEqual(a, b);
}

export function mcpEnabled(env: NodeJS.ProcessEnv): boolean {
  return env.OROTITAN_MCP_ENABLED === "true";
}

export function mcpWriteEnabled(_env: NodeJS.ProcessEnv): boolean {
  // Read-only OAuth pilot: Supabase does not support custom OAuth resource
  // scopes yet. Never infer write authority from OIDC email/profile scopes.
  // A separately reviewed capability mechanism is required before release.
  return false;
}

export function allowedMcpRuns(env: NodeJS.ProcessEnv): ReadonlySet<string> {
  const list = env.OROTITAN_MCP_ALLOWED_RUN_IDS?.split(",").map(x => x.trim().toLowerCase()) ?? [];
  if (list.length === 0 || list.length > 25 || list.some(x => !UUID.test(x))) return new Set();
  if (new Set(list).size !== list.length) return new Set();
  return new Set(list);
}

export function authorizeMcpBearer(
  authorization: string | null,
  env: NodeJS.ProcessEnv,
): boolean {
  if (!mcpEnabled(env) || allowedMcpRuns(env).size === 0) return false;
  const expected = env.OROTITAN_MCP_READ_TOKEN;
  if (typeof expected !== "string" || expected.length < 48 ||
    expected.length > 256 || !/^Bearer [^\s]{48,256}$/.test(authorization ?? "")) return false;
  return safeCompare(authorization!.slice(7), expected);
}

export function requireMcpRun(runId: string, env: NodeJS.ProcessEnv): string {
  const normalized = runId.toLowerCase();
  if (!UUID.test(normalized) || !allowedMcpRuns(env).has(normalized)) {
    throw new McpBridgeError("RUN_NOT_ALLOWED");
  }
  return normalized;
}

export class McpBridgeError extends Error {
  constructor(public readonly code: string) { super(code); }
}
