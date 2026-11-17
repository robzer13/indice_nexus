/**
 * C08: exact pinned V3.1 loader after a successful LOCAL PG17 COMMIT.
 * No Supabase client, HTTP, network, service-role, or secrets. PSQL loopback only.
 * C03 candidate Git locators are SIMULATED and have NO remote authority.
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import {
  loadTrustedSchneiderShadowSnapshot,
  type ArtifactRow,
  type RunRow,
  type StageRow,
  type IdentityRow,
  type TrustedPilotagePort,
} from "../../c08-candidate-v31/lib/orotitan-equity/v3/pilotage-trusted-loader-v3-1";

const RUN = "db00ce77-04eb-492b-a76b-164ecbe9ccfe";
const CANDIDATE = new Set([
  "SCORING_INPUTS:1",
  "FUNDAMENTALS_LOCK:2",
  "DEEP_DIVE_STAGE_MANIFEST:2",
]);

function check(value: unknown): asserts value {
  if (!value) throw new Error("C08 fail-closed");
}
function safeDb(sql: string): unknown[] {
  check(process.env.PGHOST === "127.0.0.1");
  check(process.env.PGPORT === "5432");
  check(process.env.PGUSER === "postgres");
  check(process.env.PGSSLMODE === "disable");
  const db = process.argv[2];
  check(/^orotitan_registry_v1_[0-9]+$/.test(db ?? ""));
  const env = {
    PATH: process.env.PATH || "/usr/bin:/bin",
    HOME: process.env.HOME || "/tmp",
    LANG: "C.UTF-8",
    PGHOST: "127.0.0.1", PGPORT: "5432", PGUSER: "postgres",
    PGPASSWORD: "postgres", PGSSLMODE: "disable", PGDATABASE: db,
  };
  // Never permit psql to print private errors in GitHub Actions public logs.
  const out = execFileSync("psql",
    ["-X", "-w", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-d", db, "-c", sql],
    { env, timeout: 30000, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  return out.split("\n").filter(line => line.trim()).map(line => JSON.parse(line));
}
function sha(data: Buffer, alg: "sha256" | "sha1"): string {
  return createHash(alg).update(data).digest("hex");
}
function immutableBytes(row: ArtifactRow, root: string): Uint8Array {
  const type = row.artifact_type;
  const version = row.version;
  check(/^[A-Z][A-Z0-9_]+$/.test(type));
  check(Number.isSafeInteger(version) && version >= 1 && version <= 100);
  check(row.run_id === RUN);
  const name = type + "__v" + String(version).padStart(3, "0") + ".json";
  const localCandidate = CANDIDATE.has(type + ":" + String(version));
  const path = localCandidate ? join(root, name) :
    join(root, "historical", row.stage_code.toLowerCase(), name);
  const buf = readFileSync(path);
  check(buf.length <= 300000 && buf.length === row.size_bytes);
  check(sha(buf, "sha256") === row.content_sha256);
  const gitHeader = Buffer.from("blob " + buf.length + "\0");
  check(createHash("sha1").update(gitHeader).update(buf).digest("hex") === row.github_blob_sha);
  return new Uint8Array(buf);
}
async function run(): Promise<void> {
  const root = resolve(process.argv[3] || "");
  check(root.endsWith("/orotitan-schneider-c07-private"));
  const runRows = safeDb("SELECT row_to_json(t)::text FROM (SELECT * FROM public.orotitan_runs WHERE run_id='" + RUN + "'::uuid) t;") as RunRow[];
  const stages = safeDb("SELECT row_to_json(t)::text FROM (SELECT * FROM public.orotitan_run_stages WHERE run_id='" + RUN + "'::uuid ORDER BY stage_code) t;") as StageRow[];
  const artifacts = safeDb("SELECT row_to_json(t)::text FROM (SELECT * FROM public.orotitan_artifacts WHERE run_id='" + RUN + "'::uuid ORDER BY stage_code,artifact_id,version) t;") as ArtifactRow[];
  check(runRows.length === 1 && runRows[0].state_version === 8);
  check(stages.length === 2 && artifacts.length === 31);
  const port: TrustedPilotagePort = {
    async getRun() { return runRows[0] ?? null; },
    async getStages() { return stages; },
    async getArtifacts() { return artifacts; },
    async getIssuer(issuerId: string): Promise<IdentityRow | null> {
      check(/^[0-9a-f-]{36}$/.test(issuerId));
      const rows = safeDb(
        "SELECT row_to_json(t)::text FROM (SELECT issuer_id,display_name,legal_name FROM public.issuers WHERE issuer_id='" +
        issuerId + "'::uuid) t;"
      ) as IdentityRow[];
      return rows.length === 1 ? rows[0] : null;
    },
    async readImmutableBytes(row: ArtifactRow) {
      return immutableBytes(row, root);
    },
  };
  const snapshot = await loadTrustedSchneiderShadowSnapshot({
    port,
    environment: {
      projectRef: "awgsurdyvsyolcgpnygh",
      projectName: "orotitan-vnext-shadow",
      region: "eu-west-3",
      status: "ACTIVE_HEALTHY",
    },
    runId: RUN,
    expectedRunStateVersion: 8,
    expectedCurrentStageStateVersion: 3,
  });
  check(snapshot.stages.DEEP_DIVE?.manifest?.version === 2);
  check(snapshot.stages.DEEP_DIVE?.gates.READY_FOR_VALUATION === true);
  process.stdout.write("C08 LOCAL_POSTCOMMIT_TRUSTED_LOADER=PASS\n");
}
run().catch(() => {
  // Never print exception text: it can contain Registry metadata or private paths.
  process.stderr.write("BLOCKED_C08_POSTCOMMIT_TRUSTED_LOADER\n");
  process.exitCode = 1;
});
