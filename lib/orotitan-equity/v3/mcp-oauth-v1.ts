/**
 * MCP OAuth 2.1 resource-server verification.
 * OAuth authorization endpoints, PKCE, consent and token issuance belong to
 * an independent identity provider, NOT to OroTitan or its admin cookie.
 * Never accept arbitrary JWT issuers, JWKS URLs, subject IDs or audiences.
 */
import { createRemoteJWKSet, jwtVerify, type JWTPayload, type JWTVerifyGetKey } from "jose";

export type McpOAuthConfig = {
  issuer: string;
  audience: string;
  tokenAudience: "authenticated";
  jwksUri: string;
  resourceMetadataUrl: string;
  allowedSubjects: ReadonlySet<string>;
  allowedClients: ReadonlySet<string>;
};
export type McpOAuthIdentity = {
  subject: string;
  scopes: ReadonlySet<string>;
};

export class McpOAuthError extends Error {
  constructor(public readonly code: string) { super(code); }
}

export const MCP_SCOPE_READ = "orotitan.read";
export const MCP_SCOPE_WRITE = "orotitan.write";

function strictHttps(raw: string | undefined): URL | null {
  if (!raw || raw.length > 700) return null;
  try {
    const url = new URL(raw);
    if (url.protocol !== "https:" || url.username || url.password || url.search ||
      url.hash || !url.hostname || url.hostname === "localhost" ||
      url.hostname.endsWith(".local") || url.hostname === "127.0.0.1" ||
      url.hostname === "0.0.0.0") return null;
    return url;
  } catch { return null; }
}

export function mcpOAuthConfig(env: NodeJS.ProcessEnv): McpOAuthConfig | null {
  if (env.OROTITAN_MCP_ENABLED !== "true" ||
    env.OROTITAN_MCP_AUTH_MODE !== "OAUTH") return null;

  const origin = strictHttps(env.OROTITAN_MCP_PUBLIC_ORIGIN);
  const issuer = strictHttps(env.OROTITAN_MCP_OAUTH_ISSUER);
  const jwks = strictHttps(env.OROTITAN_MCP_OAUTH_JWKS_URI);
  const subjectsRaw = env.OROTITAN_MCP_OAUTH_ALLOWED_SUBJECTS;
  const subjects = subjectsRaw?.split(",").map(s => s.trim()) ?? [];
  const clients = env.OROTITAN_MCP_OAUTH_ALLOWED_CLIENT_IDS?.split(",").map(s => s.trim()) ?? [];
  if (!origin || !issuer || !jwks || origin.pathname !== "/" ||
    issuer.pathname.endsWith("//") || jwks.origin !== issuer.origin ||
    subjects.length < 1 || subjects.length > 10 ||
    subjects.some(s => s.length < 4 || s.length > 160 || !/^[a-zA-Z0-9_.:@|-]+$/.test(s)) ||
    new Set(subjects).size !== subjects.length ||
    clients.length < 1 || clients.length > 10 ||
    clients.some(s => s.length < 6 || s.length > 160 || !/^[a-zA-Z0-9_-]+$/.test(s)) ||
    new Set(clients).size !== clients.length) return null;

  // Audience/resource identifier is fixed by the host, never a model input.
  const resource = origin.origin + "/api/orotitan/mcp";
  return {
    issuer: issuer.href.replace(/\/$/, ""),
    audience: resource,
    tokenAudience: "authenticated",
    jwksUri: jwks.href,
    resourceMetadataUrl: origin.origin + "/.well-known/oauth-protected-resource",
    allowedSubjects: new Set(subjects),
    allowedClients: new Set(clients),
  };
}

export function isOAuthPayloadAdmitted(payload: JWTPayload, config: McpOAuthConfig): boolean {
  const clientId = payload.client_id;
  return typeof payload.sub === "string" &&
    config.allowedSubjects.has(payload.sub) &&
    typeof clientId === "string" &&
    config.allowedClients.has(clientId) &&
    // ChatGPT requires resource-bound access tokens. Supabase's default
    // aud="authenticated" is not sufficient to prove the token was minted
    // for this MCP resource. A reviewed, client-specific access-token hook
    // must add this signed claim before we can activate the bridge.
    payload.resource === config.audience &&
    payload.role === "authenticated" &&
    typeof payload.exp === "number" && typeof payload.iat === "number";
}

// Never construct a remote JWKS URL from token-controlled jku/x5u/jwk headers.
const remoteKeysets = new Map<string, ReturnType<typeof createRemoteJWKSet>>();
function getRemoteKeyset(uri: string) {
  let keyset = remoteKeysets.get(uri);
  if (!keyset) {
    keyset = createRemoteJWKSet(new URL(uri), {
      timeoutDuration: 3000,
      cacheMaxAge: 60_000,
      cooldownDuration: 5_000,
    });
    remoteKeysets.set(uri, keyset);
  }
  return keyset;
}

/** No bearer fallback when OAUTH is chosen. */
export async function verifyMcpOAuthBearer(
  authorization: string | null,
  config: McpOAuthConfig,
  keys: JWTVerifyGetKey = getRemoteKeyset(config.jwksUri),
): Promise<McpOAuthIdentity> {
  if (!authorization || !/^Bearer [A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/.test(authorization) ||
    authorization.length > 9_000) throw new McpOAuthError("OAUTH_TOKEN_MISSING_OR_INVALID");
  const jwt = authorization.slice(7);
  try {
    const { payload } = await jwtVerify(jwt, keys, {
      issuer: config.issuer, audience: config.tokenAudience,
      algorithms: ["RS256","ES256"], requiredClaims: ["iss","aud","sub","exp","iat"],
      maxTokenAge: "1h", clockTolerance: "5s",
    });
    if (!isOAuthPayloadAdmitted(payload,config)) {
      throw new McpOAuthError("OAUTH_IDENTITY_OR_SCOPE_DENIED");
    }
    return {subject: payload.sub!, scopes: new Set([MCP_SCOPE_READ])};
  } catch (error) {
    if (error instanceof McpOAuthError) throw error;
    throw new McpOAuthError("OAUTH_VERIFICATION_FAILED");
  }
}

export function oauthResourceMetadata(config: McpOAuthConfig) {
  return {
    resource: config.audience,
    authorization_servers: [config.issuer],
    scopes_supported: ["email", "profile"],
  };
}

export function oauthChallenge(config: McpOAuthConfig): string {
  return `Bearer resource_metadata="${config.resourceMetadataUrl}"`;
}
