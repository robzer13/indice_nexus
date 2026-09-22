import assert from "node:assert/strict";
import test from "node:test";

import Ajv2020 from "ajv/dist/2020";

import schema from "../schemas/vnext/state-model-v0.1.schema.json";

const ajv = new Ajv2020({ allErrors: true, strict: true });
const validate = ajv.compile(schema);

test("VNext state-vector schema accepts all eight required dimensions", () => {
  const valid = {
    runtime_state: "READY",
    stage_state: "IN_PROGRESS",
    analytical_state: "PROVISIONALLY_STABLE",
    evidence_state: "PARTIAL_BUT_DECISIONABLE",
    valuation_reliability: "MEDIUM",
    price_condition: "AT_OR_BELOW_REQUIRED_RETURN_PRICE",
    decision_state: "INVESTABLE_NOW",
    audit_status: "PASS",
  };

  assert.equal(validate(valid), true, JSON.stringify(validate.errors));
});

test("VNext state-vector schema rejects missing dimensions", () => {
  const invalid = {
    runtime_state: "READY",
    stage_state: "IN_PROGRESS",
  };

  assert.equal(validate(invalid), false);
  assert.ok(validate.errors?.some((error) => error.keyword === "required"));
});

test("VNext state-vector schema rejects unknown vocabulary", () => {
  const invalid = {
    runtime_state: "BROKEN",
    stage_state: "IN_PROGRESS",
    analytical_state: "LOCKED",
    evidence_state: "SUFFICIENT",
    valuation_reliability: "HIGH",
    price_condition: "AT_OR_BELOW_STRONG_RETURN_PRICE",
    decision_state: "WAIT_FOR_PRICE",
    audit_status: "PASS",
  };

  assert.equal(validate(invalid), false);
  assert.ok(validate.errors?.some((error) => error.keyword === "enum"));
});
