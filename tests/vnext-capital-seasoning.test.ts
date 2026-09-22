import assert from "node:assert/strict";
import test from "node:test";

import moduleContract from "../contracts/orotitan-equity/vnext/modules/CAPITAL_SEASONING.module-contract.v0.1.json";
import fixtures from "./fixtures/vnext/capital-seasoning.v0.1.json";

import {
  assertCapitalSeasoningFinalizable,
  deriveReturnEvidenceUse,
  deriveSupportedSeasoningState,
  evaluateCapitalSeasoning,
  type CapitalCohortInput,
  type CapitalSeasoningInput,
} from "../runtime/vnext/modules/capital-seasoning";
import { assertValidModuleContract } from "../runtime/vnext/module-contract";

test("Capital Seasoning module contract is valid under frozen Gate 7 schema", () => {
  assert.doesNotThrow(() => assertValidModuleContract(moduleContract));
});

for (const fixture of fixtures.cases) {
  test(`Capital Seasoning fixture: ${fixture.id}`, () => {
    const output = evaluateCapitalSeasoning(
      fixture.input as CapitalSeasoningInput,
    );

    assert.equal(output.finalizable, fixture.expected.finalizable);
    assert.equal(
      output.portfolioSeasoningState,
      fixture.expected.portfolioSeasoningState,
    );

    if ("state" in fixture.expected) {
      assert.equal(output.cohorts[0]?.seasoningState, fixture.expected.state);
    }

    if ("returnEvidenceUse" in fixture.expected) {
      assert.equal(
        output.cohorts[0]?.returnEvidenceUse,
        fixture.expected.returnEvidenceUse,
      );
    }

    if ("containsValidation" in fixture.expected) {
      assert.equal(
        output.validationCodes.includes(
          fixture.expected.containsValidation,
        ),
        true,
      );
    }
  });
}

test("state derivation uses operating evidence rather than an elapsed-time threshold", () => {
  const cohort = fixtures.cases[1].input.cohorts[0] as CapitalCohortInput;
  assert.equal(deriveSupportedSeasoningState(cohort), "RAMPING");
});

test("only seasoned capital permits mature return evidence use", () => {
  assert.equal(
    deriveReturnEvidenceUse("SEASONED"),
    "MATURE_RETURN_EVIDENCE_ALLOWED",
  );
  assert.equal(
    deriveReturnEvidenceUse("STABILIZING"),
    "UNSEASONED_DO_NOT_JUDGE",
  );
  assert.equal(deriveReturnEvidenceUse("UNKNOWN"), "UNKNOWN");
});

test("recent capital can have observed return evidence without being treated as mature", () => {
  const output = evaluateCapitalSeasoning(
    fixtures.cases[1].input as CapitalSeasoningInput,
  );

  assert.equal(output.finalizable, true);
  assert.equal(
    output.cohorts[0]?.returnEvidenceUse,
    "UNSEASONED_DO_NOT_JUDGE",
  );
});

test("sequence conflicts fail closed", () => {
  const base = structuredClone(
    fixtures.cases[1].input,
  ) as CapitalSeasoningInput;

  base.cohorts[0] = {
    ...base.cohorts[0],
    deployed: {
      state: "NO",
      rationale: "Capital is not deployed.",
      evidenceIds: ["E-NO-DEPLOY"],
      assumptionIds: [],
    },
    inService: {
      state: "YES",
      rationale: "Conflicting evidence claims asset is in service.",
      evidenceIds: ["E-IN-SERVICE"],
      assumptionIds: [],
    },
  };

  const output = evaluateCapitalSeasoning(base);

  assert.equal(output.finalizable, false);
  assert.equal(
    output.validationCodes.includes(
      "CAPITAL_SEASONING_SEQUENCE_CONFLICT",
    ),
    true,
  );
});

test("UNKNOWN materiality is preserved at portfolio level", () => {
  const output = evaluateCapitalSeasoning(
    fixtures.cases[5].input as CapitalSeasoningInput,
  );

  assert.equal(output.portfolioSeasoningState, "UNKNOWN");
  assert.deepEqual(output.materialUnknownCohortIds, ["C-6"]);
});

test("no fixed seasoning duration or return threshold appears in output", () => {
  const output = evaluateCapitalSeasoning(
    fixtures.cases[2].input as CapitalSeasoningInput,
  );

  assert.equal(
    Object.prototype.hasOwnProperty.call(output, "seasoningYears"),
    false,
  );
  assert.equal(
    Object.prototype.hasOwnProperty.call(output, "returnThreshold"),
    false,
  );
});

test("valid output passes finalization assertion and invalid output fails closed", () => {
  const valid = evaluateCapitalSeasoning(
    fixtures.cases[2].input as CapitalSeasoningInput,
  );
  const invalid = evaluateCapitalSeasoning(
    fixtures.cases[4].input as CapitalSeasoningInput,
  );

  assert.doesNotThrow(() => assertCapitalSeasoningFinalizable(valid));
  assert.throws(
    () => assertCapitalSeasoningFinalizable(invalid),
    /VNEXT_CAPITAL_SEASONING_NOT_FINALIZABLE/,
  );
});
