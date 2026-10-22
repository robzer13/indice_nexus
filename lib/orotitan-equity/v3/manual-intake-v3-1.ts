/**
 * Manual ChatGPT work-product intake. Files in this namespace are quarantine
 * objects ONLY, never Registry artifacts or persistence attestations.
 * No authority is inferred from the filename, its SHA or the upload receipt.
 */
import { createHash } from "node:crypto";

export const MANUAL_INTAKE_BUCKET = "orotitan-text-artifacts-v1";
export const MANUAL_INTAKE_PREFIX = "manual-intake/v1";
export const MANUAL_INTAKE_MAX_FILES_PER_REQUEST = 3;
export const MANUAL_INTAKE_MAX_BYTES = 1_000_000;
export const MANUAL_INTAKE_MAX_FILES_PER_STAGE = 100;
export const MANUAL_INTAKE_ALLOWED_STAGES = ["RESEARCH", "DEEP_DIVE", "INTEGRATION"] as const;
export const MANUAL_INTAKE_ALLOWED_RUN_STATUSES = ["ACTIVE", "PAUSED", "BLOCKED"] as const;

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const HEX = /^[0-9a-f]{64}$/;
const ALLOWED = new Set(["md", "txt", "json", "csv"]);

export class IntakeError extends Error {
  constructor(readonly code: string) { super(code); }
}

export type ManualIntakeFile = {
  fileName: string;
  content: Uint8Array;
  contentType: string;
  sizeBytes: number;
  sha256: string;
};

export type IntakeListing = {
  filename: string;
  sha256: string;
  sizeBytes: number | null;
  stage: string;
  status: "RECEIVED_NOT_VERIFIED";
};

export function isRunId(value: unknown): value is string {
  return typeof value === "string" && UUID.test(value);
}

export function intakePrefix(runId: string, stage: string): string {
  if (!isRunId(runId) || !MANUAL_INTAKE_ALLOWED_STAGES.includes(
    stage as typeof MANUAL_INTAKE_ALLOWED_STAGES[number])) {
    throw new IntakeError("INTAKE_IDENTITY_INVALID");
  }
  return `${MANUAL_INTAKE_PREFIX}/${runId.toLowerCase()}/${stage}`;
}

export function validateIntakeBytes(filename: string, data: Uint8Array): ManualIntakeFile {
  if (typeof filename !== "string" || filename.length === 0 || filename.length > 160) {
    throw new IntakeError("INTAKE_FILENAME_INVALID");
  }
  const basename = filename.split(/[\\/]/).at(-1) ?? "";
  const ext = basename.split(".").at(-1)?.toLowerCase() ?? "";
  if (!ALLOWED.has(ext)) throw new IntakeError("INTAKE_FILETYPE_NOT_ALLOWED");
  if (data.byteLength === 0 || data.byteLength > MANUAL_INTAKE_MAX_BYTES) {
    throw new IntakeError("INTAKE_FILE_SIZE_INVALID");
  }
  // No binary data or malformed UTF-8 in the text-artifact bucket.
  let decoded: string;
  try { decoded = new TextDecoder("utf-8", { fatal: true }).decode(data); }
  catch { throw new IntakeError("INTAKE_NOT_UTF8_TEXT"); }
  if (decoded.includes("\0")) throw new IntakeError("INTAKE_NOT_UTF8_TEXT");
  if (ext === "json") {
    try {
      const parsed: unknown = JSON.parse(decoded);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("NOT_AN_OBJECT");
      }
    } catch { throw new IntakeError("INTAKE_JSON_INVALID"); }
  }
  const safeName = basename.normalize("NFKD")
    .replace(/[^a-zA-Z0-9._-]/g, "_").replace(/^\.+/, "_").slice(0, 100);
  const sha256 = createHash("sha256").update(data).digest("hex");
  return {
    fileName: safeName,
    content: data,
    contentType: ext === "json" ? "application/json" :
      ext === "csv" ? "text/csv" : "text/plain",
    sizeBytes: data.byteLength,
    sha256,
  };
}

export function intakeObjectPath(runId: string, stage: string, file: ManualIntakeFile): string {
  return `${intakePrefix(runId,stage)}/${file.sha256}__${file.fileName}`;
}

export function parseIntakeObject(
  object: { name: string; metadata?: { size?: unknown } | null }, stage: string,
): IntakeListing | null {
  const split = object.name.indexOf("__");
  if (split !== 64 || !HEX.test(object.name.slice(0, split))) return null;
  const filename = object.name.slice(split + 2);
  if (!filename || filename.length > 100 || !/^[a-zA-Z0-9._-]+$/.test(filename)) return null;
  const n = object.metadata?.size;
  return {
    filename, sha256: object.name.slice(0, split),
    sizeBytes: typeof n === "number" && Number.isSafeInteger(n) && n >= 0 ? n : null,
    stage, status: "RECEIVED_NOT_VERIFIED",
  };
}
