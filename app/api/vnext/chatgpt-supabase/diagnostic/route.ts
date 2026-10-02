import { lookup } from "node:dns/promises";
import { connect as tlsConnect } from "node:tls";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const EXPECTED_SUPABASE_HOST = "cugpgtzygqqlxetyeven.supabase.co";
const STEP_TIMEOUT_MS = 5_000;

type DiagnosticClassification =
  | "OK"
  | "ENV_MISSING"
  | "URL_INVALID"
  | "URL_UNEXPECTED_HOST"
  | "DNS_FAILURE"
  | "TLS_FAILURE"
  | "HTTP_TRANSPORT_FAILURE"
  | "AUTH_FAILURE"
  | "POSTGREST_FAILURE";

type DiagnosticStep = {
  status: "OK" | "FAILED" | "NOT_RUN";
  detail?: string;
  http_status?: number;
};

export type SupabaseRuntimeDiagnostic = {
  diagnostic_version: "0.1.0";
  mutation_allowed: false;
  expected_host: string;
  classification: DiagnosticClassification;
  environment: DiagnosticStep & {
    supabase_url_present: boolean;
    service_role_present: boolean;
  };
  url: DiagnosticStep;
  dns: DiagnosticStep;
  tls: DiagnosticStep;
  http_transport: DiagnosticStep;
  auth_rest: DiagnosticStep;
};

type LookupResult = {
  family: number;
};

type DiagnosticDependencies = {
  lookupHost: (hostname: string) => Promise<LookupResult>;
  probeTls: (hostname: string) => Promise<{ protocol: string | null }>;
  fetchHttp: typeof fetch;
};

type DiagnosticEnvironment = {
  NEXT_PUBLIC_SUPABASE_URL?: string;
  SUPABASE_SERVICE_ROLE_KEY?: string;
};

function json(body: unknown, status = 200): Response {
  return Response.json(body, {
    status,
    headers: {
      "cache-control": "no-store",
    },
  });
}

function notRun(): DiagnosticStep {
  return { status: "NOT_RUN" };
}

function safeErrorCode(error: unknown): string {
  if (typeof error === "object" && error !== null && "code" in error) {
    const code = (error as { code?: unknown }).code;
    if (typeof code === "string" && /^[A-Z0-9_]+$/.test(code)) return code;
  }
  if (error instanceof Error && /TIMEOUT/.test(error.message)) return "TIMEOUT";
  return "UNKNOWN";
}

function baseline(
  supabaseUrlPresent: boolean,
  serviceRolePresent: boolean,
  classification: DiagnosticClassification,
): SupabaseRuntimeDiagnostic {
  return {
    diagnostic_version: "0.1.0",
    mutation_allowed: false,
    expected_host: EXPECTED_SUPABASE_HOST,
    classification,
    environment: {
      status: supabaseUrlPresent && serviceRolePresent ? "OK" : "FAILED",
      supabase_url_present: supabaseUrlPresent,
      service_role_present: serviceRolePresent,
    },
    url: notRun(),
    dns: notRun(),
    tls: notRun(),
    http_transport: notRun(),
    auth_rest: notRun(),
  };
}

async function withTimeout<T>(promise: Promise<T>, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error(`${label}_TIMEOUT`)), STEP_TIMEOUT_MS);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function defaultTlsProbe(hostname: string): Promise<{ protocol: string | null }> {
  return new Promise((resolve, reject) => {
    const socket = tlsConnect({
      host: hostname,
      port: 443,
      servername: hostname,
      rejectUnauthorized: true,
    });

    const timer = setTimeout(() => {
      socket.destroy();
      reject(new Error("TLS_TIMEOUT"));
    }, STEP_TIMEOUT_MS);

    socket.once("secureConnect", () => {
      clearTimeout(timer);
      const protocol = socket.getProtocol();
      socket.end();
      resolve({ protocol });
    });

    socket.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
  });
}

const defaultDependencies: DiagnosticDependencies = {
  lookupHost: async (hostname) => {
    const result = await lookup(hostname);
    return { family: result.family };
  },
  probeTls: defaultTlsProbe,
  fetchHttp: fetch,
};

export async function runSupabaseRuntimeDiagnostic(
  env: DiagnosticEnvironment,
  dependencies: DiagnosticDependencies = defaultDependencies,
): Promise<SupabaseRuntimeDiagnostic> {
  const rawUrl = env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceRole = env.SUPABASE_SERVICE_ROLE_KEY;
  const result = baseline(Boolean(rawUrl), Boolean(serviceRole), "ENV_MISSING");

  if (!rawUrl || !serviceRole) return result;

  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    result.classification = "URL_INVALID";
    result.url = { status: "FAILED", detail: "INVALID_URL" };
    return result;
  }

  if (
    parsed.protocol !== "https:" ||
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.port !== "" ||
    parsed.pathname !== "/" ||
    parsed.search !== "" ||
    parsed.hash !== ""
  ) {
    result.classification = "URL_INVALID";
    result.url = { status: "FAILED", detail: "EXPECTED_HTTPS_ORIGIN_ONLY" };
    return result;
  }

  if (parsed.hostname !== EXPECTED_SUPABASE_HOST) {
    result.classification = "URL_UNEXPECTED_HOST";
    result.url = { status: "FAILED", detail: "UNEXPECTED_HOST" };
    return result;
  }

  result.url = { status: "OK", detail: "EXPECTED_HOST" };

  try {
    const dnsResult = await withTimeout(dependencies.lookupHost(parsed.hostname), "DNS");
    result.dns = {
      status: "OK",
      detail: dnsResult.family === 6 ? "IPV6" : dnsResult.family === 4 ? "IPV4" : "RESOLVED",
    };
  } catch (error) {
    result.classification = "DNS_FAILURE";
    result.dns = { status: "FAILED", detail: safeErrorCode(error) };
    return result;
  }

  try {
    const tlsResult = await dependencies.probeTls(parsed.hostname);
    result.tls = { status: "OK", detail: tlsResult.protocol ?? "NEGOTIATED" };
  } catch (error) {
    result.classification = "TLS_FAILURE";
    result.tls = { status: "FAILED", detail: safeErrorCode(error) };
    return result;
  }

  try {
    const transportResponse = await dependencies.fetchHttp(`${parsed.origin}/rest/v1/`, {
      method: "GET",
      headers: { accept: "application/json" },
      cache: "no-store",
      signal: AbortSignal.timeout(STEP_TIMEOUT_MS),
    });
    result.http_transport = {
      status: "OK",
      http_status: transportResponse.status,
    };
  } catch (error) {
    result.classification = "HTTP_TRANSPORT_FAILURE";
    result.http_transport = { status: "FAILED", detail: safeErrorCode(error) };
    return result;
  }

  try {
    const authResponse = await dependencies.fetchHttp(
      `${parsed.origin}/rest/v1/issuers?select=issuer_id&limit=1`,
      {
        method: "GET",
        headers: {
          accept: "application/json",
          apikey: serviceRole,
          authorization: `Bearer ${serviceRole}`,
        },
        cache: "no-store",
        signal: AbortSignal.timeout(STEP_TIMEOUT_MS),
      },
    );

    result.auth_rest = {
      status: authResponse.ok ? "OK" : "FAILED",
      http_status: authResponse.status,
    };

    if (authResponse.ok) {
      result.classification = "OK";
      return result;
    }
    if (authResponse.status === 401) {
      result.classification = "AUTH_FAILURE";
      return result;
    }

    result.classification = "POSTGREST_FAILURE";
    return result;
  } catch (error) {
    result.classification = "HTTP_TRANSPORT_FAILURE";
    result.auth_rest = { status: "FAILED", detail: safeErrorCode(error) };
    return result;
  }
}

export async function GET(): Promise<Response> {
  if (process.env.VERCEL_ENV !== "preview") {
    return json({ error: "OROTITAN_SUPABASE_DIAGNOSTIC_PREVIEW_ONLY" }, 403);
  }

  const result = await runSupabaseRuntimeDiagnostic({
    NEXT_PUBLIC_SUPABASE_URL: process.env.NEXT_PUBLIC_SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY,
  });
  return json(result);
}
