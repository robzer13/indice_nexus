import assert from "node:assert/strict";
import test from "node:test";

import moduleContract from "../contracts/orotitan-equity/vnext/modules/VALUATION_DIAGNOSTIC_INTEGRITY.module-contract.v0.1.json";
import fixtures from "./fixtures/vnext/valuation-diagnostic-integrity.v0.1.json";

import {
  assertValuationDiagnosticIntegrityFinalizable,
  evaluateValuationDiagnosticIntegrity,
  selectNDiagnostic,
  type ValuationDiagnosticIntegrityInput,
} from "../runtime/vnext/modules/valuation-diagnostic-integrity";
import { assertValidModuleContract } from "../runtime/vnext/module-contract";

test("Valuation Diagnostic Integrity module contract is valid under frozen Gate 7 schema", () => {
  assert.doesNotThrow(() => assertValidModuleContract(moduleContract));
});

for (const fixture of fixtures.cases) {
  test(`Valuation Diagnostic Integrity fixture: ${fixture.id}`, () => {
    const output = evaluateValuationDiagnosticIntegrity(
      fixture.input as ValuationDiagnosticIntegrityInput,
    );

    assert.equal(output.finalizable, fixture.expected.finalizable);

    if (
      "selectedNBasis" in fixture.expected &&
      typeof fixture.expected.selectedNBasis === "string"
    ) {
      assert.equal(
        output.selectedNBasis,
        fixture.expected.selectedNBasis,
      );
    }

    if (
      "numericOvsPermittedByNSelection" in fixture.expected &&
      typeof fixture.expected.numericOvsPermittedByNSelection ===
        "boolean"
    ) {
      assert.equal(
        output.numericOvsPermittedByNSelection,
        fixture.expected.numericOvsPermittedByNSelection,
      );
    }

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

    if ("containsLimitation" in fixture.expected) {
      const expected = fixture.expected.containsLimitation;
      const codes = Array.isArray(expected) ? expected : [expected];

      for (const code of codes) {
        if (typeof code === "string") {
          assert.equal(output.limitations.includes(code), true);
        }
      }
    }
  });
}

test("valid numeric Mature Normalization deterministically precedes valid Same-Multiple", () => {
  const input =
    fixtures.cases[0].input as ValuationDiagnosticIntegrityInput;

  assert.equal(
    selectNDiagnostic(
      input.matureNormalization,
      input.sameMultiple,
    ),
    "MATURE_NORMALIZATION_RETURN",
  );
});

test("Same-Multiple fallback is allowed only when Mature Normalization is legitimately non-numeric", () => {
  const input =
    fixtures.cases[1].input as ValuationDiagnosticIntegrityInput;

  assert.equal(
    selectNDiagnostic(
      input.matureNormalization,
      input.sameMultiple,
    ),
    "NO_MULTIPLE_EXPANSION_RETURN",
  );
});

test("invalid Mature Normalization never authorizes Same-Multiple fallback", () => {
  const input =
    fixtures.cases[2].input as ValuationDiagnosticIntegrityInput;

  assert.equal(
    selectNDiagnostic(
      input.matureNormalization,
      input.sameMultiple,
    ),
    "INVALID",
  );
});

test("no numeric N diagnostic keeps numeric OVS prohibited rather than inventing a value", () => {
  const output = evaluateValuationDiagnosticIntegrity(
    fixtures.cases[3].input as ValuationDiagnosticIntegrityInput,
  );

  assert.equal(output.selectedNBasis, "NOT_AVAILABLE");
  assert.equal(output.numericOvsPermittedByNSelection, false);
});

test("Reverse DCF solves one material variable and rejects underdetermined multi-variable solving", () => {
  const output = evaluateValuationDiagnosticIntegrity(
    fixtures.cases[4].input as ValuationDiagnosticIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "REVERSE_DCF_MUST_SOLVE_ONE_MATERIAL_VARIABLE",
    ),
    true,
  );
});

test("expected shareholder return uses exact IRR when interim cash flows are modeled", () => {
  const output = evaluateValuationDiagnosticIntegrity(
    fixtures.cases[5].input as ValuationDiagnosticIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "INTERIM_CASH_FLOWS_REQUIRE_EXACT_IRR",
    ),
    true,
  );
  assert.equal(
    output.validationCodes.includes(
      "NAIVE_EXPECTED_RETURN_ADDITION_FORBIDDEN",
    ),
    true,
  );
});

test("10-year expected return requires economic-horizon support", () => {
  const output = evaluateValuationDiagnosticIntegrity(
    fixtures.cases[6].input as ValuationDiagnosticIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "TEN_YEAR_EXPECTED_RETURN_REQUIRES_ECONOMIC_HORIZON_SUPPORT",
    ),
    true,
  );
});

test("normalized multiple cross-check requires economic comparability", () => {
  const output = evaluateValuationDiagnosticIntegrity(
    fixtures.cases[7].input as ValuationDiagnosticIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "NORMALIZED_MULTIPLE_REQUIRES_COMPARABILITY",
    ),
    true,
  );
});

test("cyclical valuation requires mid-cycle normalization when triggered", () => {
  const output = evaluateValuationDiagnosticIntegrity(
    fixtures.cases[8].input as ValuationDiagnosticIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "CYCLICAL_VALUATION_REQUIRES_MID_CYCLE_NORMALIZATION",
    ),
    true,
  );
});

test("valuation arithmetic and monotonicity checks fail closed", () => {
  const output = evaluateValuationDiagnosticIntegrity(
    fixtures.cases[9].input as ValuationDiagnosticIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "REQUIRED_MATH_CHECK_NOT_PASS:MONOTONICITY_DISCOUNT_RATE",
    ),
    true,
  );
});

test("sensitivity focuses on two to four dominant variables when valuation is assessable", () => {
  const output = evaluateValuationDiagnosticIntegrity(
    fixtures.cases[10].input as ValuationDiagnosticIntegrityInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "VALUE_MOST_SENSITIVE_TO_REQUIRES_TWO_TO_FOUR_VARIABLES",
    ),
    true,
  );
});

test("NOT_ASSESSABLE valuation reliability remains a valid explicit state but prohibits numeric OVS", () => {
  const output = evaluateValuationDiagnosticIntegrity(
    fixtures.cases[11].input as ValuationDiagnosticIntegrityInput,
  );

  assert.equal(output.finalizable, true);
  assert.equal(output.valuationReliability, "NOT_ASSESSABLE");
  assert.equal(output.numericOvsPermittedByNSelection, false);
});

test("module does not emit OVS, Investment Score, valuation Elite or intrinsic value", () => {
  const output = evaluateValuationDiagnosticIntegrity(
    fixtures.cases[0].input as ValuationDiagnosticIntegrityInput,
  );

  for (const forbidden of [
    "ovs",
    "investmentScore",
    "valuationElite",
    "intrinsicValue",
  ]) {
    assert.equal(
      Object.prototype.hasOwnProperty.call(output, forbidden),
      false,
    );
  }
});

test("valid output passes finalization assertion and invalid output fails closed", () => {
  const valid = evaluateValuationDiagnosticIntegrity(
    fixtures.cases[0].input as ValuationDiagnosticIntegrityInput,
  );
  const invalid = evaluateValuationDiagnosticIntegrity(
    fixtures.cases[2].input as ValuationDiagnosticIntegrityInput,
  );

  assert.doesNotThrow(() =>
    assertValuationDiagnosticIntegrityFinalizable(valid),
  );

  assert.throws(
    () =>
      assertValuationDiagnosticIntegrityFinalizable(invalid),
    /VNEXT_VALUATION_DIAGNOSTIC_INTEGRITY_NOT_FINALIZABLE/,
  );
});
