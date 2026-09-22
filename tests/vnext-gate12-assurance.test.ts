import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

import {
  HANDOFF_GATE_BY_STAGE,
  RUN_STATUSES,
  STAGE_LIFECYCLES,
} from "../runtime/vnext/state-machine";
import { RECOVERY_CLASSES } from "../runtime/vnext/recovery-engine";
import {
  VNEXT_SHADOW_PROJECT_REF,
  assertVNextShadowSupabaseUrl,
} from "../runtime/vnext/environment";

type GoldenFixture = {
  fixture_schema_version: string;
  name: string;
  frozen_runtime: {
    run_statuses: string[];
    stage_lifecycles: string[];
    handoff_gate_by_stage: Record<string, string>;
    recovery_classes: string[];
    environment: {
      shadow_project_ref: string;
      production_project_ref: string;
      publication_enabled: boolean;
    };
  };
  hash_fixture: {
    exact_utf8_text: string;
    sha256: string;
  };
  required_assurance_files: string[];
};

function readGolden(): GoldenFixture {
  const path = resolve(
    process.cwd(),
    "tests/vnext/fixtures/gate12-golden-v0.1.json",
  );
  return JSON.parse(readFileSync(path, "utf8")) as GoldenFixture;
}

test("Gate 12 golden fixture pins deterministic runtime vocabularies", () => {
  const golden = readGolden();

  assert.equal(golden.fixture_schema_version, "0.1.0");
  assert.deepEqual(RUN_STATUSES, golden.frozen_runtime.run_statuses);
  assert.deepEqual(
    STAGE_LIFECYCLES,
    golden.frozen_runtime.stage_lifecycles,
  );
  assert.deepEqual(
    HANDOFF_GATE_BY_STAGE,
    golden.frozen_runtime.handoff_gate_by_stage,
  );
  assert.deepEqual(
    RECOVERY_CLASSES,
    golden.frozen_runtime.recovery_classes,
  );
});

test("Gate 12 golden hash fixture detects exact-byte drift", () => {
  const golden = readGolden();
  const actual = createHash("sha256")
    .update(Buffer.from(golden.hash_fixture.exact_utf8_text, "utf8"))
    .digest("hex");

  assert.equal(actual, golden.hash_fixture.sha256);
});

test("Gate 12 golden environment remains shadow-only and production rejects", () => {
  const golden = readGolden();

  assert.equal(
    VNEXT_SHADOW_PROJECT_REF,
    golden.frozen_runtime.environment.shadow_project_ref,
  );
  assert.equal(golden.frozen_runtime.environment.publication_enabled, false);

  assert.throws(
    () =>
      assertVNextShadowSupabaseUrl(
        `https://${golden.frozen_runtime.environment.production_project_ref}.supabase.co`,
      ),
    /production Supabase is forbidden/,
  );
});

test("Gate 12 required deterministic assurance files all exist", () => {
  const golden = readGolden();

  for (const relativePath of golden.required_assurance_files) {
    assert.equal(
      existsSync(resolve(process.cwd(), relativePath)),
      true,
      `missing assurance file: ${relativePath}`,
    );
  }
});

test("Gate 12 CI cannot silently omit pull-request deterministic assurance", () => {
  const workflow = readFileSync(
    resolve(process.cwd(), ".github/workflows/vnext-ci.yml"),
    "utf8",
  );

  assert.match(workflow, /pull_request:/);
  assert.match(workflow, /- vnext/);
  assert.match(workflow, /verify-vnext:/);
  assert.match(workflow, /Validate VNext baseline manifest/);
  assert.match(workflow, /npm run lint/);
  assert.match(workflow, /npm run typecheck/);
  assert.match(workflow, /npm test/);
  assert.match(workflow, /npm run test:postgres/);
  assert.match(workflow, /npm run build/);
});

test("Gate 12 package scripts keep unit, contract and PostgreSQL assurance wired", () => {
  const packageJson = JSON.parse(
    readFileSync(resolve(process.cwd(), "package.json"), "utf8"),
  ) as {
    scripts?: Record<string, string>;
  };

  assert.equal(packageJson.scripts?.test, "tsx --test tests/*.test.ts");
  assert.match(packageJson.scripts?.["test:postgres"] ?? "", /migration-tests/);
  assert.equal(packageJson.scripts?.typecheck, "tsc --noEmit");
  assert.match(packageJson.scripts?.lint ?? "", /--max-warnings=0/);
});
