import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import { createHash } from "node:crypto";
import { METHOD_V2_AUTHORITY_SET_SHA256 } from "../lib/orotitan-equity/method-v2/authority";

const read = (path: string) => readFileSync(path, "utf8").replace(/\r/g, "");
const migrationPath = "migrations/20261005195139_orotitan_registry_v1_13_method_generation.sql";
const migration = read(migrationPath);
const historicalCreate = read("migrations/20260921_orotitan_registry_v1_10_methodology_successor_cas.sql");
const predecessorPath = "migrations/20260923_orotitan_registry_v1_12_valuation_date_alignment_successor.sql";
const predecessor = read(predecessorPath);

test("Restored production predecessor has exact historical bytes and precedes Method Generation", () => {
  const bytes = readFileSync(predecessorPath);
  assert.equal(createHash("sha1").update(`blob ${bytes.length}\0`).update(bytes).digest("hex"),
    "350cf25c07df7eaa53ee6b8865daeb8b72a04432");
  const runner = read("tests/postgres/run-registry-v1-migration-tests.sh");
  assert.ok(runner.includes(predecessorPath));
  assert.ok(runner.indexOf('-f "$valuation_date_alignment"') < runner.indexOf('-f "$registry13"'));
  assert.ok(runner.indexOf('-f "$registry13"') < runner.indexOf('-f "$registry14"'));
  assert.ok(runner.includes("0985bfebba7e3b2885cd457c5f662565"));
  assert.equal((runner.match(/-f "\$valuation_date_verify"/g) ?? []).length, 2);
});

test("Registry identity uses the exact frozen Method-V2 V1.1 authority", () => {
  assert.equal(METHOD_V2_AUTHORITY_SET_SHA256, "1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2");
  assert.ok(migration.includes(METHOD_V2_AUTHORITY_SET_SHA256));
  assert.deepEqual(readdirSync("migrations").filter(name => name.includes("v1_13_method_generation")), [migrationPath.split("/").at(-1)]);
});

test("Legacy create and successor admission remain byte-equivalent except cross-generation replay rejection", () => {
  const functions = (sql: string) => [...sql.matchAll(/create or replace function public\.(create_orotitan_(?:run|methodology_successor_run))\([\s\S]*?\$function\$;/g)].map(match => match[0]);
  const guard = "\n    if v_existing.methodology_generation is distinct from 'METHOD_V1' then\n      raise exception 'IDEMPOTENCY_CONFLICT: legacy route requires METHOD_V1' using errcode = '23514';\n    end if;";
  assert.equal(functions(migration).length, 2);
  assert.deepEqual(functions(migration).map(fn => fn.replace(guard, "")),
    [functions(historicalCreate)[0], functions(predecessor)[0]]);
});

test("Migration materializes Registry identity without snapshot propagation or runtime activation", () => {
  assert.doesNotMatch(migration, /(?:alter table|insert into|update) public\.research_snapshots/i);
  assert.doesNotMatch(migration, /grant\s/i);
  for (const name of ["create_orotitan_method_v2_run", "create_orotitan_method_v2_successor_run"]) {
    assert.match(migration, new RegExp(`revoke all on function public\\.${name}\\([\\s\\S]*?\\) from public, anon, authenticated, service_role;`));
  }
  assert.match(migration, /add column methodology_generation text not null default 'METHOD_V1'/);
});
