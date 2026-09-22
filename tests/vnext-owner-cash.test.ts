import assert from "node:assert/strict";
import test from "node:test";

import moduleContract from "../contracts/orotitan-equity/vnext/modules/OWNER_CASH.module-contract.v0.1.json";
import fixtures from "./fixtures/vnext/owner-cash.v0.1.json";

import {
  assertOwnerCashFinalizable,
  evaluateOwnerCash,
  type OwnerCashInput,
} from "../runtime/vnext/modules/owner-cash";
import { assertValidModuleContract } from "../runtime/vnext/module-contract";

test("Owner Cash module contract is valid under frozen Gate 7 schema", () => {
  assert.doesNotThrow(() => assertValidModuleContract(moduleContract));
});

for (const fixture of fixtures.cases) {
  test(`Owner Cash fixture: ${fixture.id}`, () => {
    const output = evaluateOwnerCash(
      fixture.input as OwnerCashInput,
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

test("a point Owner Earnings estimate requires resolved material maintenance components", () => {
  const output = evaluateOwnerCash(
    fixtures.cases[2].input as OwnerCashInput,
  );

  assert.equal(output.finalizable, false);
  assert.equal(
    output.validationCodes.includes(
      "OWNER_EARNINGS_POINT_REQUIRES_RESOLVED_MAINTENANCE",
    ),
    true,
  );
});

test("a range may preserve maintenance uncertainty without false precision", () => {
  const output = evaluateOwnerCash(
    fixtures.cases[1].input as OwnerCashInput,
  );

  assert.equal(output.ownerEarningsForm, "RANGE");
  assert.equal(output.finalizable, true);
  assert.equal(
    output.unknownMaintenanceComponents.includes(
      "INTANGIBLE_INVESTMENT",
    ),
    true,
  );
});

test("maintenance reinvestment cannot be subtracted twice after total cash capex", () => {
  const base = {
    ...(fixtures.cases[0].input as OwnerCashInput),
    maintenanceDoubleCountedAgainstTotalCapexFcf: true,
  };

  const output = evaluateOwnerCash(base);

  assert.equal(
    output.validationCodes.includes(
      "MAINTENANCE_REINVESTMENT_DOUBLE_COUNTED",
    ),
    true,
  );
});

test("working-capital movement already embedded in CFO cannot be subtracted twice", () => {
  const output = evaluateOwnerCash(
    fixtures.cases[8].input as OwnerCashInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "WORKING_CAPITAL_DOUBLE_COUNTED",
    ),
    true,
  );
});

test("material SBC receives one economic penalty, not zero or two", () => {
  const output = evaluateOwnerCash(
    fixtures.cases[4].input as OwnerCashInput,
  );

  assert.equal(output.finalizable, false);
  assert.equal(
    output.validationCodes.includes("SBC_DOUBLE_PENALTY_FORBIDDEN"),
    true,
  );
  assert.equal(
    output.validationCodes.includes(
      "MATERIAL_SBC_REQUIRES_ONE_PENALTY",
    ),
    true,
  );
});

test("acquisitions stay outside Standardized FCF and material acquisition deployment is separated from organic cash", () => {
  const output = evaluateOwnerCash(
    fixtures.cases[5].input as OwnerCashInput,
  );

  assert.equal(
    output.validationCodes.includes(
      "ACQUISITIONS_NOT_SUBTRACTED_FROM_STANDARDIZED_FCF_BY_DEFAULT",
    ),
    true,
  );
  assert.equal(
    output.validationCodes.includes(
      "MATERIAL_ACQUISITION_CAPITAL_REQUIRES_ORGANIC_CASH_SEPARATION",
    ),
    true,
  );
});

test("per-share economic cash requires economic diluted shares", () => {
  const base = {
    ...(fixtures.cases[0].input as OwnerCashInput),
    perShareUsesEconomicDilutedShares: {
      state: "NO" as const,
      rationale:
        "Per-share cash currently uses a basic share count rather than economic diluted shares.",
      evidenceIds: ["E-PS-NO"],
      assumptionIds: [],
    },
  };

  const output = evaluateOwnerCash(base);

  assert.equal(
    output.validationCodes.includes(
      "PER_SHARE_CASH_REQUIRES_ECONOMIC_DILUTED_SHARES",
    ),
    true,
  );
});

test("sector-valid distributable-capital framework may replace industrial FCF and Owner Earnings", () => {
  const output = evaluateOwnerCash(
    fixtures.cases[6].input as OwnerCashInput,
  );

  assert.equal(output.cashFramework, "SECTOR_DISTRIBUTABLE_CAPITAL");
  assert.equal(output.ownerEarningsForm, "NOT_APPLICABLE");
  assert.equal(output.finalizable, true);
});

test("Owner Cash does not emit Cash Economics score or elite status", () => {
  const output = evaluateOwnerCash(
    fixtures.cases[0].input as OwnerCashInput,
  );

  assert.equal(
    Object.prototype.hasOwnProperty.call(output, "cashEconomicsScore"),
    false,
  );
  assert.equal(
    Object.prototype.hasOwnProperty.call(output, "cashEconomicsElite"),
    false,
  );
});

test("valid output passes finalization assertion and invalid output fails closed", () => {
  const valid = evaluateOwnerCash(
    fixtures.cases[0].input as OwnerCashInput,
  );
  const invalid = evaluateOwnerCash(
    fixtures.cases[2].input as OwnerCashInput,
  );

  assert.doesNotThrow(() => assertOwnerCashFinalizable(valid));
  assert.throws(
    () => assertOwnerCashFinalizable(invalid),
    /VNEXT_OWNER_CASH_NOT_FINALIZABLE/,
  );
});
