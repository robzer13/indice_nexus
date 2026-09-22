import assert from "node:assert/strict";
import test from "node:test";

import moduleContract from "../contracts/orotitan-equity/vnext/modules/FINANCING_CONSISTENCY.module-contract.v0.1.json";
import fixtures from "./fixtures/vnext/financing-consistency.v0.1.json";

import {
  assertFinancingConsistencyFinalizable,
  evaluateFinancingConsistency,
  type FinancingConsistencyInput,
} from "../runtime/vnext/modules/financing-consistency";
import { assertValidModuleContract } from "../runtime/vnext/module-contract";

test("Financing Consistency module contract is valid under frozen Gate 7 schema", () => {
  assert.doesNotThrow(() => assertValidModuleContract(moduleContract));
});

for (const fixture of fixtures.cases) {
  test(`Financing Consistency fixture: ${fixture.id}`, () => {
    const output = evaluateFinancingConsistency(
      fixture.input as FinancingConsistencyInput,
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

test("material supplier finance with financing substance requires debt-like reconciliation", () => {
  const output = evaluateFinancingConsistency(
    fixtures.cases[1].input as FinancingConsistencyInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "MATERIAL_SUPPLIER_FINANCE_REQUIRES_DEBT_LIKE_RECONCILIATION",
    ),
    true,
  );
});

test("material factoring cannot improve CFO without financing-transfer reconciliation", () => {
  const output = evaluateFinancingConsistency(
    fixtures.cases[2].input as FinancingConsistencyInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "MATERIAL_FACTORING_REQUIRES_FINANCING_TRANSFER_RECONCILIATION",
    ),
    true,
  );
});

test("material leases require consistent cash, ROIC and EV treatment", () => {
  const output = evaluateFinancingConsistency(
    fixtures.cases[3].input as FinancingConsistencyInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "MATERIAL_LEASES_REQUIRE_CASH_ROIC_EV_CONSISTENCY",
    ),
    true,
  );
});

test("buybacks require both a funding bridge and share-count reconciliation", () => {
  const output = evaluateFinancingConsistency(
    fixtures.cases[4].input as FinancingConsistencyInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "BUYBACKS_REQUIRE_FINANCING_BRIDGE",
    ),
    true,
  );
  assert.equal(
    output.validationCodes.includes(
      "BUYBACKS_REQUIRE_SHARE_COUNT_RECONCILIATION",
    ),
    true,
  );
});

test("dividends must remain consistent with retained-capital needs", () => {
  const output = evaluateFinancingConsistency(
    fixtures.cases[5].input as FinancingConsistencyInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "DIVIDENDS_MUST_RECONCILE_WITH_RETAINED_CAPITAL_NEEDS",
    ),
    true,
  );
});

test("UNKNOWN financing materiality remains explicit as a limitation", () => {
  const output = evaluateFinancingConsistency(
    fixtures.cases[8].input as FinancingConsistencyInput,
  );

  assert.equal(output.finalizable, true);
  assert.equal(
    output.limitations.includes(
      "SUPPLIER_FINANCE_MATERIALITY_UNKNOWN",
    ),
    true,
  );
});

test("Financing Consistency does not create a leverage, capital-allocation or resilience score", () => {
  const output = evaluateFinancingConsistency(
    fixtures.cases[0].input as FinancingConsistencyInput,
  );

  assert.equal(
    Object.prototype.hasOwnProperty.call(output, "leverageScore"),
    false,
  );
  assert.equal(
    Object.prototype.hasOwnProperty.call(
      output,
      "capitalAllocationScore",
    ),
    false,
  );
  assert.equal(
    Object.prototype.hasOwnProperty.call(output, "resilienceScore"),
    false,
  );
});

test("valid output passes finalization assertion and invalid output fails closed", () => {
  const valid = evaluateFinancingConsistency(
    fixtures.cases[0].input as FinancingConsistencyInput,
  );
  const invalid = evaluateFinancingConsistency(
    fixtures.cases[1].input as FinancingConsistencyInput,
  );

  assert.doesNotThrow(() =>
    assertFinancingConsistencyFinalizable(valid),
  );

  assert.throws(
    () => assertFinancingConsistencyFinalizable(invalid),
    /VNEXT_FINANCING_CONSISTENCY_NOT_FINALIZABLE/,
  );
});
