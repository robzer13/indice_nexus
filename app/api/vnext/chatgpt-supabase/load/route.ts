import type { LoadRequest, OperationFailure } from "@/lib/orotitan-equity/post-c7/chatgpt-supabase-bridge";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type ContextTier = NonNullable<LoadRequest["requested_context_tiers"]>[number];

const ALLOWED_QUERY_PARAMS = new Set(["issuer_query", "run_id", "tiers"]);
const CONTEXT_TIERS = new Set<ContextTier>(["L0", "L1", "L2", "L3"]);
const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function json(body: unknown, status = 200): Response {
  return Response.json(body, {
    status,
    headers: {
      "cache-control": "no-store",
    },
  });
}

function requireSingleParam(url: URL, name: string): string | null {
  const values = url.searchParams.getAll(name);
  if (values.length > 1) {
    throw new Error(`${name} must be provided at most once`);
  }
  return values[0] ?? null;
}

function parseContextTiers(raw: string | null): ContextTier[] | undefined {
  if (raw === null) return undefined;
  if (raw.trim().length === 0) {
    throw new Error("tiers must not be empty when provided");
  }

  const tiers = raw.split(",").map((value) => value.trim());
  const unique = new Set(tiers);
  if (unique.size !== tiers.length) {
    throw new Error("tiers must not contain duplicates");
  }

  for (const tier of tiers) {
    if (!CONTEXT_TIERS.has(tier as ContextTier)) {
      throw new Error(`unsupported context tier: ${tier}`);
    }
  }

  return tiers as ContextTier[];
}

export function parseLoadRequestFromUrl(url: URL): LoadRequest {
  for (const key of url.searchParams.keys()) {
    if (!ALLOWED_QUERY_PARAMS.has(key)) {
      throw new Error(`unsupported query parameter: ${key}`);
    }
  }

  const issuerQuery = requireSingleParam(url, "issuer_query")?.trim() ?? "";
  if (issuerQuery.length < 1 || issuerQuery.length > 256) {
    throw new Error("issuer_query must contain 1 to 256 characters");
  }

  const rawRunId = requireSingleParam(url, "run_id");
  const runId = rawRunId === null ? null : rawRunId.trim();
  if (runId !== null && !UUID_PATTERN.test(runId)) {
    throw new Error("run_id must be a UUID");
  }

  const tiers = parseContextTiers(requireSingleParam(url, "tiers"));

  return {
    contract_version: "0.1.0",
    operation: "LOAD",
    issuer_query: issuerQuery,
    ...(runId !== null ? { run_id: runId } : {}),
    ...(tiers !== undefined ? { requested_context_tiers: tiers } : {}),
  };
}

function failureStatus(result: OperationFailure): number {
  if (result.error_class === "NOT_FOUND") return 404;
  if (result.error_class === "CONTRACT_VIOLATION") return 400;
  if (result.error_class === "INFRASTRUCTURE") return 503;
  return 409;
}

function serverSetupFailure(): OperationFailure {
  return {
    contract_version: "0.1.0",
    operation: "OPERATION_FAILURE",
    error_class: "INFRASTRUCTURE",
    message: "LOAD bridge server initialization failed",
    retry_without_reload_allowed: false,
  };
}

export async function GET(request: Request): Promise<Response> {
  if (process.env.VERCEL_ENV !== "preview") {
    return json({ error: "OROTITAN_BRIDGE_LOAD_PREVIEW_ONLY" }, 403);
  }

  // Vercel Deployment Protection validates this Trusted Sources OIDC header
  // before the request reaches the function. This local guard fails closed if
  // the header is absent and never exposes or logs the token.
  const oidcToken = request.headers.get("x-vercel-trusted-oidc-idp-token");
  if (!oidcToken || oidcToken.trim().length === 0) {
    return json({ error: "OROTITAN_BRIDGE_LOAD_TRUSTED_OIDC_REQUIRED" }, 401);
  }

  let loadRequest: LoadRequest;
  try {
    loadRequest = parseLoadRequestFromUrl(new URL(request.url));
  } catch (error) {
    return json(
      {
        error: "OROTITAN_BRIDGE_LOAD_REQUEST_INVALID",
        message: error instanceof Error ? error.message : "invalid LOAD request",
      },
      400,
    );
  }

  let result;
  try {
    const { executeServerControlledOperation } = await import(
      "@/lib/orotitan-equity/post-c7/chatgpt-supabase-bridge-server"
    );
    result = await executeServerControlledOperation(loadRequest);
  } catch {
    const failure = serverSetupFailure();
    return json(failure, failureStatus(failure));
  }

  if (result.operation === "LOAD_RESULT") {
    return json(result);
  }
  if (result.operation === "OPERATION_FAILURE") {
    return json(result, failureStatus(result));
  }

  return json({ error: "OROTITAN_BRIDGE_LOAD_NON_READ_RESULT_REJECTED" }, 500);
}
