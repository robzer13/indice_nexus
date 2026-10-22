import { requireAdminSession } from "@/lib/auth/admin-session";
import { createServerSupabaseClient } from "@/lib/supabase/server";
import { assertProductionUrl } from "@/lib/orotitan-equity/v3/pilotage-trusted-loader-server";
import {
  IntakeError, MANUAL_INTAKE_ALLOWED_RUN_STATUSES,
  MANUAL_INTAKE_MAX_BYTES, MANUAL_INTAKE_MAX_FILES_PER_STAGE,
  MANUAL_INTAKE_BUCKET, intakePrefix, intakeObjectPath,
  isRunId, parseIntakeObject, validateIntakeBytes,
} from "@/lib/orotitan-equity/v3/manual-intake-v3-1";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function reply(body: Record<string, unknown>, status = 200): Response {
  return Response.json(body, { status, headers: {
    "cache-control": "no-store", "x-content-type-options": "nosniff",
  } });
}

type Client = ReturnType<typeof createServerSupabaseClient>;

async function authorizedClient(): Promise<Client | null> {
  try { await requireAdminSession(); assertProductionUrl(); }
  catch { return null; }
  return createServerSupabaseClient();
}

async function findRun(client: Client, runId: string) {
  const { data, error } = await client.from("orotitan_runs")
    .select("run_id,current_stage,run_status,state_version").eq("run_id", runId).maybeSingle();
  if (error) throw new IntakeError("INTAKE_RUN_QUERY_FAILED");
  if (!data) throw new IntakeError("INTAKE_RUN_NOT_FOUND");
  return data as {run_id: string; current_stage: string | null; run_status: string; state_version: number};
}

function outcome(err: unknown): Response {
  const code = err instanceof IntakeError ? err.code : "INTAKE_UNAVAILABLE";
  const status = code === "INTAKE_RUN_NOT_FOUND" ? 404 :
    code === "INTAKE_DISABLED" ? 403 :
    code === "INTAKE_LIMIT_REACHED" || code === "INTAKE_RUN_CHANGED_REFRESH" ? 409 :
    code.startsWith("INTAKE_FILE") || code.startsWith("INTAKE_JSON") ||
    code.startsWith("INTAKE_NOT_UTF8") || code === "INTAKE_IDENTITY_INVALID" ?
      400 : 503;
  return reply({ ok: false, code }, status);
}

function isEnabled(): boolean {
  // Must be explicitly enabled after separate production approval.
  return process.env.OROTITAN_MANUAL_INTAKE_ENABLED === "true";
}

async function objectsInStage(client: Client, prefix: string) {
  const { data, error } = await client.storage.from(MANUAL_INTAKE_BUCKET).list(prefix, {
    limit: MANUAL_INTAKE_MAX_FILES_PER_STAGE + 1,
    offset: 0, sortBy: { column: "name", order: "asc" },
  });
  if (error) throw new IntakeError("INTAKE_LIST_UNAVAILABLE");
  return data ?? [];
}

/** Lists only quarantine uploads. This is NOT an authoritative Registry view. */
export async function GET(request: Request): Promise<Response> {
  const client = await authorizedClient();
  if (!client) return reply({ ok: false, code: "UNAUTHORIZED" }, 401);
  const runId = new URL(request.url).searchParams.get("run");
  if (!isRunId(runId)) return reply({ ok: false, code: "INTAKE_IDENTITY_INVALID" }, 400);
  try {
    const run = await findRun(client, runId);
    const prefix = intakePrefix(run.run_id, run.current_stage ?? "");
    const objects = await objectsInStage(client, prefix);
    const files = objects.map(item => parseIntakeObject(item, run.current_stage!))
      .filter((item): item is NonNullable<typeof item> => item !== null);
    return reply({
      ok: true, enabled: isEnabled(), currentStage: run.current_stage,
      runStateVersion: run.state_version,
      files, limit: MANUAL_INTAKE_MAX_FILES_PER_STAGE,
      authority: "QUARANTINE_ONLY_NOT_REGISTRY",
      nextAction: "VERIFY_AND_REGISTER_SEPARATELY",
    });
  } catch (err) { return outcome(err); }
}

/** No artifact registration, stage transition, attestation or publication here. */
export async function POST(request: Request): Promise<Response> {
  const client = await authorizedClient();
  if (!client) return reply({ ok: false, code: "UNAUTHORIZED" }, 401);
  if (!isEnabled()) return outcome(new IntakeError("INTAKE_DISABLED"));
  const origin = request.headers.get("origin");
  if (!origin || origin !== new URL(request.url).origin) {
    return reply({ ok: false, code: "INTAKE_ORIGIN_REJECTED" }, 403);
  }
  const length = request.headers.get("content-length");
  if (length && (Number(length) > MANUAL_INTAKE_MAX_BYTES + 50_000 ||
    !Number.isSafeInteger(Number(length)))) {
    return reply({ ok: false, code: "INTAKE_REQUEST_TOO_LARGE" }, 413);
  }
  if (!request.headers.get("content-type")?.toLowerCase().startsWith("multipart/form-data;")) {
    return reply({ ok: false, code: "INTAKE_MULTIPART_REQUIRED" }, 415);
  }

  let form: FormData;
  try { form = await request.formData(); }
  catch { return reply({ ok: false, code: "INTAKE_MULTIPART_INVALID" }, 400); }
  const runId = form.get("run");
  const expectedStage = form.get("expectedStage");
  const expectedRunVersion = Number(form.get("expectedRunStateVersion"));
  const files = form.getAll("file");
  if (!isRunId(runId) || typeof expectedStage !== "string" ||
    !Number.isSafeInteger(expectedRunVersion) || expectedRunVersion < 1 ||
    files.length !== 1 || !(files[0] instanceof File)) {
    return reply({ ok: false, code: "INTAKE_REQUEST_INVALID" }, 400);
  }
  const upload = files[0];
  if (upload.size === 0 || upload.size > MANUAL_INTAKE_MAX_BYTES) {
    return reply({ ok: false, code: "INTAKE_FILE_SIZE_INVALID" }, 400);
  }
  try {
    const run = await findRun(client, runId);
    if (run.current_stage !== expectedStage || run.state_version !== expectedRunVersion) {
      throw new IntakeError("INTAKE_RUN_CHANGED_REFRESH");
    }
    if (!MANUAL_INTAKE_ALLOWED_RUN_STATUSES.includes(
      run.run_status as typeof MANUAL_INTAKE_ALLOWED_RUN_STATUSES[number])) {
      throw new IntakeError("INTAKE_RUN_NOT_EDITABLE");
    }
    const prefix = intakePrefix(run.run_id, run.current_stage ?? "");
    // The validation is server-side; client-side checks are only UX.
    const file = validateIntakeBytes(upload.name, new Uint8Array(await upload.arrayBuffer()));
    const list = await objectsInStage(client, prefix);
    const objectName = intakeObjectPath(run.run_id, run.current_stage!, file);
    const basename = objectName.slice(prefix.length + 1);
    if (list.some(item => item.name === basename)) {
      return reply({ ok: true, duplicate: true, status: "RECEIVED_NOT_VERIFIED",
        filename: file.fileName, sha256: file.sha256 });
    }
    if (list.length >= MANUAL_INTAKE_MAX_FILES_PER_STAGE) {
      throw new IntakeError("INTAKE_LIMIT_REACHED");
    }
    const { error } = await client.storage.from(MANUAL_INTAKE_BUCKET)
      .upload(objectName, file.content, {
        contentType: file.contentType, upsert: false, cacheControl: "0",
      });
    if (error) {
      // Do not reinterpret a storage conflict as content verification.
      throw new IntakeError("INTAKE_STORAGE_UPLOAD_FAILED");
    }
    return reply({
      ok: true, duplicate: false, status: "RECEIVED_NOT_VERIFIED",
      filename: file.fileName, sha256: file.sha256, sizeBytes: file.sizeBytes,
      authority: "QUARANTINE_ONLY_NOT_REGISTRY",
      nextAction: "VERIFY_AND_REGISTER_SEPARATELY",
    }, 201);
  } catch (err) { return outcome(err); }
}
