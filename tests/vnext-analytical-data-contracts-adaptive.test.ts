import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import Ajv2020 from "ajv/dist/2020";
import addFormats from "ajv-formats";

const files = [
  "orotitan-analytical-common.schema.v0.1.json",
  "company-economic-dna.schema.v0.1.json",
  "overlay-selection.schema.v0.1.json",
  "cycle-analysis.schema.v0.1.json",
  "technology-map.schema.v0.1.json",
  "industry-structure.schema.v0.1.json",
  "causal-graph.schema.v0.1.json",
  "serial-acquirer-capital-deployment.schema.v0.1.json",
] as const;

function load(name: string) {
  return JSON.parse(
    readFileSync(
      new URL(`../schemas/vnext/data-contracts/${name}`, import.meta.url),
      "utf8",
    ),
  );
}

test("V2 adaptive analytical schemas compile together", () => {
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);

  for (const name of files) ajv.addSchema(load(name));
  for (const name of files) {
    assert.ok(ajv.getSchema(load(name).$id), `missing schema: ${name}`);
  }
});

test("cycle regime vocabulary preserves explicit UNKNOWN", () => {
  const common = load("orotitan-analytical-common.schema.v0.1.json");
  assert.deepEqual(common.$defs.cycleRegime.enum, [
    "PEAK",
    "LATE_CYCLE",
    "MID_CYCLE",
    "DOWNTURN",
    "TROUGH",
    "EARLY_RECOVERY",
    "EXPANSION",
    "UNKNOWN",
  ]);
});

test("causal edges preserve unsupported and not-assessable states", () => {
  const common = load("orotitan-analytical-common.schema.v0.1.json");
  assert.deepEqual(common.$defs.causalEdgeStatus.enum, [
    "SUPPORTED",
    "MIXED",
    "NOT_SUPPORTED",
    "LOW_CONFIDENCE",
    "NOT_ASSESSABLE",
  ]);
});

test("serial-acquirer contract reuses frozen capital-seasoning states", () => {
  const schema = load("serial-acquirer-capital-deployment.schema.v0.1.json");
  const seasoning =
    schema.properties.acquisition_cohorts.items.properties.seasoning_state.enum;

  assert.deepEqual(seasoning, [
    "COMMITTED",
    "DEPLOYED",
    "IN_SERVICE",
    "RAMPING",
    "STABILIZING",
    "SEASONED",
    "UNKNOWN",
  ]);
});

test("adaptive V2 support contracts contain no scoring or OroTitan terminal authority", () => {
  for (const name of files.slice(1)) {
    const raw = JSON.stringify(load(name));
    assert.equal(raw.includes("OQS"), false, name);
    assert.equal(raw.includes("OVS"), false, name);
    assert.equal(raw.includes("INVESTMENT_SCORE"), false, name);
    assert.equal(raw.includes("OROTITAN_STATUS"), false, name);
  }
});
