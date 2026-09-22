import assert from "node:assert/strict";
import test from "node:test";

import moduleContract from "../contracts/orotitan-equity/vnext/modules/RETURN_NORMALIZATION.module-contract.v0.1.json";
import fixtures from "./fixtures/vnext/return-normalization.v0.1.json";

import {
  assertReturnNormalizationFinalizable,
  deriveRoiicInterpretability,
  evaluateReturnNormalization,
  type ReturnNormalizationInput,
} from "../runtime/vnext/modules/return-normalization";
import { assertValidModuleContract } from "../runtime/vnext/module-contract";

test("Return Normalization module contract is valid under frozen Gate 7 schema", () => {
  assert.doesNotThrow(() => assertValidModuleContract(moduleContract));
});

for (const fixture of fixtures.cases) {
  test(`Return Normalization fixture: ${fixture.id}`, () => {
    const result = evaluateReturnNormalization(
      fixture.input as ReturnNormalizationInput,
    );

    assert.equal(result.finalizable, fixture.expected.finalizable);
    assert.equal(
      result.roiicInterpretability,
      fixture.expected.roiicInterpretability,
    );
    assert.deepEqual(
      result.validationCodes,
      fixture.expected.validationCodes,
    );
  });
}

test("3-year ROIIC passes the interpretability gate when all frozen conditions hold", () => {
  const input = fixtures.cases[0].input as ReturnNormalizationInput;

  assert.equal(
    deriveRoiicInterpretability(input.roiic),
    "INTERPRETABLE",
  );
});

test("near-zero delta invested capital makes ROIIC not interpretable", () => {
  const input = fixtures.cases[3].input as ReturnNormalizationInput;

  assert.equal(
    deriveRoiicInterpretability(input.roiic),
    "NOT_INTERPRETABLE",
  );
});

test("one-year ROIIC is diagnostic only and cannot pass as primary normalization", () => {
  const base = structuredClone(
    fixtures.cases[0].input,
  ) as ReturnNormalizationInput;

  base.roiic = {
    ...base.roiic,
    primaryWindowYears: 1,
  };

  const output = evaluateReturnNormalization(base);

  assert.equal(output.finalizable, false);
  assert.equal(
    output.validationCodes.includes("ROIIC_1Y_DIAGNOSTIC_ONLY"),
    true,
  );
});

test("material acquisition capital requires All-In ROIC and acquisition cohort return", () => {
  const base = structuredClone(
    fixtures.cases[1].input,
  ) as ReturnNormalizationInput;

  base.headlineMeasures = ["STANDARD_ROIC"];
  base.specialMeasures = [];

  const output = evaluateReturnNormalization(base);

  assert.equal(output.finalizable, false);
  assert.equal(
    output.validationCodes.includes(
      "ALL_IN_ROIC_REQUIRED_FOR_MATERIAL_ACQUISITION_CAPITAL",
    ),
    true,
  );
  assert.equal(
    output.validationCodes.includes(
      "ACQUISITION_COHORT_RETURN_REQUIRED",
    ),
    true,
  );
});

test("ROIC ex-goodwill remains diagnostic rather than a headline return metric", () => {
  const base = structuredClone(
    fixtures.cases[1].input,
  ) as ReturnNormalizationInput;

  base.headlineMeasures = [
    "STANDARD_ROIC",
    "ALL_IN_ROIC",
    "ROIC_EX_GOODWILL",
  ];
  base.diagnosticMeasures = [];

  const output = evaluateReturnNormalization(base);

  assert.equal(
    output.validationCodes.includes(
      "ROIC_EX_GOODWILL_DIAGNOSTIC_ONLY",
    ),
    true,
  );
});

test("near-zero invested capital requires alternative economics and rejects denominator-driven headline ROIC", () => {
  const base = structuredClone(
    fixtures.cases[2].input,
  ) as ReturnNormalizationInput;

  base.headlineMeasures = ["STANDARD_ROIC"];
  base.diagnosticMeasures = [];
  base.alternativeEconomicsEvidenceIds = [];

  const output = evaluateReturnNormalization(base);

  assert.equal(output.finalizable, false);
  assert.equal(
    output.validationCodes.includes(
      "EXPLOSIVE_DENOMINATOR_ROIC_NOT_HEADLINE_QUALITY_EVIDENCE",
    ),
    true,
  );
  assert.equal(
    output.validationCodes.includes(
      "ALTERNATIVE_ECONOMICS_REQUIRED_FOR_UNINTERPRETABLE_ROIC",
    ),
    true,
  );
});

test("R&D current-period add-back requires denominator asset and amortization symmetry", () => {
  const output = evaluateReturnNormalization(
    fixtures.cases[4].input as ReturnNormalizationInput,
  );

  assert.equal(output.finalizable, false);
  assert.equal(
    output.validationCodes.includes(
      "RND_ADD_BACK_REQUIRES_ASSET_AND_AMORTIZATION_SYMMETRY",
    ),
    true,
  );
});

test("industrial-not-applicable framework requires a sector-valid headline framework", () => {
  const base = structuredClone(
    fixtures.cases[5].input,
  ) as ReturnNormalizationInput;

  base.headlineMeasures = [];

  const output = evaluateReturnNormalization(base);

  assert.equal(output.finalizable, false);
  assert.equal(
    output.validationCodes.includes(
      "SECTOR_VALID_RETURN_FRAMEWORK_REQUIRED",
    ),
    true,
  );
});

test("unknown metadata remains explicit as limitations rather than being coerced", () => {
  const base = structuredClone(
    fixtures.cases[0].input,
  ) as ReturnNormalizationInput;

  base.dataQuality = "UNKNOWN";
  base.attributability = "UNKNOWN";
  base.statedInterpretability = "UNKNOWN";

  const output = evaluateReturnNormalization(base);

  assert.deepEqual(output.limitations, [
    "ATTRIBUTABILITY_UNKNOWN",
    "DATA_QUALITY_UNKNOWN",
    "RETURN_INTERPRETABILITY_UNKNOWN",
  ]);
});

test("return normalization does not produce a return-quality score or elite verdict", () => {
  const output = evaluateReturnNormalization(
    fixtures.cases[0].input as ReturnNormalizationInput,
  );

  assert.equal(
    Object.prototype.hasOwnProperty.call(output, "returnQualityScore"),
    false,
  );
  assert.equal(
    Object.prototype.hasOwnProperty.call(output, "returnQualityElite"),
    false,
  );
});

test("finalizable output passes assertion and invalid output fails closed", () => {
  const valid = evaluateReturnNormalization(
    fixtures.cases[0].input as ReturnNormalizationInput,
  );
  const invalid = evaluateReturnNormalization(
    fixtures.cases[3].input as ReturnNormalizationInput,
  );

  assert.doesNotThrow(() =>
    assertReturnNormalizationFinalizable(valid),
  );

  assert.throws(
    () => assertReturnNormalizationFinalizable(invalid),
    /VNEXT_RETURN_NORMALIZATION_NOT_FINALIZABLE/,
  );
});
