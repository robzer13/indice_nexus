import assert from "node:assert/strict";
import test from "node:test";

import moduleContract from "../contracts/orotitan-equity/vnext/modules/VALUATION_ASSUMPTION_INTEGRITY.module-contract.v0.1.json";
import fixtures from "./fixtures/vnext/valuation-assumption-integrity.v0.1.json";

import {
  assertValuationAssumptionIntegrityFinalizable,
  evaluateValuationAssumptionIntegrity,
  type ValuationAssumptionIntegrityInput,
} from "../runtime/vnext/modules/valuation-assumption-integrity";
import { assertValidModuleContract } from "../runtime/vnext/module-contract";

test("Valuation Assumption Integrity module contract is valid under frozen Gate 7 schema", () => {
  assert.doesNotThrow(() => assertValidModuleContract(moduleContract));
});

for (const fixture of fixtures.cases) {
  test(`Valuation Assumption Integrity fixture: ${fixture.id}`, () => {
    const output = evaluateValuationAssumptionIntegrity(
      fixture.input as ValuationAssumptionIntegrityInput,
    );

    assert.equal(output.finalizable, fixture.expected.finalizable);

    if (
      "validationCodes" in fixture.expected &&
      Array.isArray(fixture.expected.validationCodes)
    ) {
      assert.deepEqual(
        output.validationCodes,
        fixture.expected.validationCodes,
      );
    }

    if ("containsValidation" in fixture.expected) {
      const expected = fixture.expected.containsValidation;
      const codes = Array.isArray(expected) ? expected : [expected];

      for (const code of codes) {
        if (typeof code === "string") {
          assert.equal(output.validationCodes.includes(code), true);
        }
      }
    }

    if (
      "limitations" in fixture.expected &&
      Array.isArray(fixture.expected.limitations)
    ) {
      assert.deepEqual(output.limitations, fixture.expected.limitations);
    }

    if (
      "containsLimitation" in fixture.expected &&
      typeof fixture.expected.containsLimitation === "string"
    ) {
      assert.equal(
        output.limitations.includes(
          fixture.expected.containsLimitation,
        ),
        true,
      );
    }
  });
}

test("FCFF must use WACC", () => {
  const output = evaluateValuationAssumptionIntegrity(
    fixtures.cases[1].input as ValuationAssumptionIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "FCFF_MUST_BE_DISCOUNTED_AT_WACC",
    ),
    true,
  );
});

test("UNKNOWN Owner Earnings cannot support a precise Owner Earnings DCF", () => {
  const output = evaluateValuationAssumptionIntegrity(
    fixtures.cases[2].input as ValuationAssumptionIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "OWNER_EARNINGS_UNKNOWN_CANNOT_SUPPORT_PRECISE_OWNER_EARNINGS_DCF",
    ),
    true,
  );
});

test("runway horizon cannot mechanically determine explicit forecast years", () => {
  const output = evaluateValuationAssumptionIntegrity(
    fixtures.cases[3].input as ValuationAssumptionIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "RUNWAY_HORIZON_CANNOT_MECHANICALLY_SET_FORECAST_YEARS",
    ),
    true,
  );
});

test("growth must reconcile to reinvestment and expected marginal return", () => {
  const output = evaluateValuationAssumptionIntegrity(
    fixtures.cases[4].input as ValuationAssumptionIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "GROWTH_REINVESTMENT_MARGINAL_RETURN_MUST_RECONCILE",
    ),
    true,
  );
});

test("margin expansion and optionality cannot be counted again in terminal value", () => {
  const output = evaluateValuationAssumptionIntegrity(
    fixtures.cases[5].input as ValuationAssumptionIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "MARGIN_EXPANSION_DOUBLE_COUNTED_IN_TERMINAL_VALUE",
    ),
    true,
  );
  assert.equal(
    output.validationCodes.includes(
      "OPTIONALITY_DOUBLE_COUNTED_BETWEEN_BASE_AND_TERMINAL",
    ),
    true,
  );
});

test("terminal growth must reconcile with reinvestment and return on new capital", () => {
  const output = evaluateValuationAssumptionIntegrity(
    fixtures.cases[6].input as ValuationAssumptionIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "TERMINAL_GROWTH_REINVESTMENT_RETURN_MUST_RECONCILE",
    ),
    true,
  );
});

test("material future M&A requires target pool, capital, capacity and return support", () => {
  const output = evaluateValuationAssumptionIntegrity(
    fixtures.cases[7].input as ValuationAssumptionIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "MATERIAL_FUTURE_MNA_REQUIRES_TARGET_POOL_CAPITAL_CAPACITY_AND_RETURN_SUPPORT",
    ),
    true,
  );
});

test("material assumptions retain exact identity, variable, value/range, epistemic type, rationale, sensitivity and use", () => {
  const output = evaluateValuationAssumptionIntegrity(
    fixtures.cases[0].input as ValuationAssumptionIntegrityInput,
  );

  assert.deepEqual(output.materialAssumptionIds, [
    "A-REV-1",
    "A-SHARES-1",
    "A-TERM-1",
    "A-WACC-1",
  ]);
});

test("assumption leakage and critical placeholders block finalization", () => {
  const output = evaluateValuationAssumptionIntegrity(
    fixtures.cases[8].input as ValuationAssumptionIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "ASSUMPTION_LEAKAGE_INTO_FACTUAL_NARRATIVE",
    ),
    true,
  );
  assert.equal(
    output.validationCodes.includes(
      "CRITICAL_PLACEHOLDER_BLOCKS_VALUATION_ASSUMPTION_INTEGRITY",
    ),
    true,
  );
});

test("non-critical unresolved placeholder remains an explicit limitation", () => {
  const output = evaluateValuationAssumptionIntegrity(
    fixtures.cases[10].input as ValuationAssumptionIntegrityInput,
  );

  assert.equal(output.finalizable, true);
  assert.equal(
    output.limitations.includes(
      "NON_CRITICAL_PLACEHOLDER_MUST_REMAIN_UNKNOWN_LIMITATION",
    ),
    true,
  );
});

test("module does not emit valuation score, intrinsic value, reliability or Elite status", () => {
  const output = evaluateValuationAssumptionIntegrity(
    fixtures.cases[0].input as ValuationAssumptionIntegrityInput,
  );

  for (const forbidden of [
    "intrinsicValue",
    "ovs",
    "investmentScore",
    "valuationReliability",
    "valuationElite",
  ]) {
    assert.equal(
      Object.prototype.hasOwnProperty.call(output, forbidden),
      false,
    );
  }
});

test("valid output passes finalization assertion and invalid output fails closed", () => {
  const valid = evaluateValuationAssumptionIntegrity(
    fixtures.cases[0].input as ValuationAssumptionIntegrityInput,
  );
  const invalid = evaluateValuationAssumptionIntegrity(
    fixtures.cases[1].input as ValuationAssumptionIntegrityInput,
  );

  assert.doesNotThrow(() =>
    assertValuationAssumptionIntegrityFinalizable(valid),
  );

  assert.throws(
    () =>
      assertValuationAssumptionIntegrityFinalizable(invalid),
    /VNEXT_VALUATION_ASSUMPTION_INTEGRITY_NOT_FINALIZABLE/,
  );
});
