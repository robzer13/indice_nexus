import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import Ajv2020 from "ajv/dist/2020";
import addFormats from "ajv-formats";

import { INITIAL_DD_INPUT_BLOCKS } from "../runtime/vnext/analytical-data-contracts";

const schemaFiles = [
  "orotitan-analytical-common.schema.v0.1.json",
  "research-source-manifest.schema.v0.1.json",
  "evidence-ledger.schema.v0.1.json",
  "conflict-ledger.schema.v0.1.json",
  "calculation-ledger.schema.v0.1.json",
  "material-assumption-register.schema.v0.1.json",
  "material-research-hypothesis-register.schema.v0.1.json",
  "research-gap-register.schema.v0.1.json",
  "dd-input-sufficiency-record.schema.v0.1.json",
  "analysis-input-lock.schema.v0.1.json",
  "analytical-block-output.schema.v0.1.json",
] as const;

function loadSchema(name: string) {
  return JSON.parse(
    readFileSync(
      new URL(`../schemas/vnext/data-contracts/${name}`, import.meta.url),
      "utf8",
    ),
  );
}

test("all V2 core analytical schemas compile together under Draft 2020-12", () => {
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);

  for (const name of schemaFiles) {
    ajv.addSchema(loadSchema(name));
  }

  for (const name of schemaFiles) {
    const schema = loadSchema(name);
    assert.ok(ajv.getSchema(schema.$id), `schema not registered: ${name}`);
  }
});

test("frozen epistemic and DD sufficiency vocabularies remain exact", () => {
  const common = loadSchema("orotitan-analytical-common.schema.v0.1.json");

  assert.deepEqual(common.$defs.epistemicType.enum, [
    "REPORTED",
    "CALCULATED",
    "CONSENSUS",
    "ESTIMATE",
    "ASSUMPTION",
    "UNKNOWN",
  ]);

  assert.deepEqual(common.$defs.ddInputStatus.enum, [
    "SUFFICIENT",
    "INSUFFICIENT",
    "NOT_APPLICABLE",
  ]);

  assert.equal(
    common.$defs.ddInputStatus.enum.includes("PARTIALLY_SUFFICIENT"),
    false,
  );
});

test("initial and full refresh retain exactly the 11 frozen DD input blocks", () => {
  const schema = loadSchema("dd-input-sufficiency-record.schema.v0.1.json");
  const enumValues = schema.properties.blocks.items.properties.block_id.enum;

  assert.deepEqual(enumValues, [...INITIAL_DD_INPUT_BLOCKS]);
});

test("assumption register structurally requires ASSUMPTION epistemic type", () => {
  const schema = loadSchema(
    "material-assumption-register.schema.v0.1.json",
  );
  assert.equal(
    schema.properties.assumptions.items.properties.epistemic_type.const,
    "ASSUMPTION",
  );
});


test("analytical block vocabulary is frozen for Process Engine routing", () => {
  const common = loadSchema("orotitan-analytical-common.schema.v0.1.json");
  const block = loadSchema("analytical-block-output.schema.v0.1.json");

  assert.deepEqual(common.$defs.blockCode.enum, [
    "BUSINESS_MODEL",
    "ECONOMIC_QUALITY",
    "INDUSTRY_STRUCTURE",
    "TECHNOLOGY",
    "CYCLICALITY",
    "MOAT",
    "RUNWAY",
    "RETURN_QUALITY",
    "FCF_FORENSIC",
    "CAPITAL_ALLOCATION",
    "MANAGEMENT_GOVERNANCE",
    "OUTSIDE_VIEW",
    "RISK_RESILIENCE",
    "RED_TEAM",
    "VALUATION",
    "CROSS_BLOCK_RECONCILIATION",
  ]);

  assert.equal(
    block.properties.block_id.$ref,
    "./orotitan-analytical-common.v0.1.json#/$defs/blockCode",
  );
});

test("French-first stays a product-layer rule and machine enums remain canonical", () => {
  const direction = JSON.parse(
    readFileSync(
      new URL(
        "../calibration/vnext/OROTITAN_POST_C7_PRODUCT_DIRECTION_ANALYTICAL_ENGINE_V2_001.json",
        import.meta.url,
      ),
      "utf8",
    ),
  );
  const common = loadSchema("orotitan-analytical-common.schema.v0.1.json");

  assert.equal(direction.product_requirements.french_first_ui, true);
  assert.ok(common.$defs.ddInputStatus.enum.includes("INSUFFICIENT"));
  assert.equal(common.$defs.ddInputStatus.enum.includes("INSUFFISANT"), false);
});
