import assert from "node:assert/strict";
import test from "node:test";

import moduleContract from "../contracts/orotitan-equity/vnext/modules/DECISION_STATE_ARCHITECTURE.module-contract.v0.1.json";
import fixtures from "./fixtures/vnext/decision-state-architecture.v0.1.json";

import {
  assertDecisionStateFinalizable,
  deriveEligibleDecisionStates,
  evaluateDecisionStateArchitecture,
  type DecisionStateInput,
} from "../runtime/vnext/modules/decision-state-architecture";
import { assertValidModuleContract } from "../runtime/vnext/module-contract";

test("Decision State Architecture module contract is valid under frozen Gate 7 schema", () => {
  assert.doesNotThrow(() => assertValidModuleContract(moduleContract));
});

for (const fixture of fixtures.cases) {
  test(`Decision State fixture: ${fixture.id}`, () => {
    const result = evaluateDecisionStateArchitecture(
      fixture.input as DecisionStateInput,
    );

    assert.equal(result.finalizable, fixture.expected.finalizable);
    assert.deepEqual(
      result.eligibleCanonicalStates,
      fixture.expected.eligibleCanonicalStates,
    );
    assert.deepEqual(
      result.validationCodes,
      fixture.expected.validationCodes,
    );
  });
}

test("canonical decision-state vocabulary remains exact", () => {
  const states = new Set(
    fixtures.cases.map(
      (fixture) => fixture.input.proposedDecisionState,
    ),
  );

  for (const expected of [
    "UNKNOWN",
    "INVESTABLE_NOW",
    "WAIT_FOR_PRICE",
    "WAIT_FOR_EVIDENCE",
    "REFRESH_REQUIRED",
    "REJECT",
  ]) {
    assert.equal(states.has(expected), true);
  }
});

test("INVESTABLE_NOW cannot be generated from score or price alone", () => {
  const base = structuredClone(
    fixtures.cases[0].input,
  ) as DecisionStateInput;

  base.certificationRequirementsSatisfied = {
    state: "NO",
    rationale:
      "Certification requirements are not satisfied despite attractive economics.",
    evidenceIds: ["E-CERT-BLOCK"],
    assumptionIds: [],
  };

  const eligible = deriveEligibleDecisionStates(base);

  assert.equal(eligible.includes("INVESTABLE_NOW"), false);
});

test("WAIT_FOR_PRICE requires a prepared business and inadequate valuation", () => {
  const base = structuredClone(
    fixtures.cases[1].input,
  ) as DecisionStateInput;

  base.businessPrepared = {
    state: "NO",
    rationale: "The business dossier is not prepared.",
    evidenceIds: ["E-NOT-PREPARED"],
    assumptionIds: [],
  };

  const eligible = deriveEligibleDecisionStates(base);

  assert.equal(eligible.includes("WAIT_FOR_PRICE"), false);
});

test("WAIT_FOR_EVIDENCE is eligible when material economic uncertainty blocks a decision", () => {
  const input = fixtures.cases[2].input as DecisionStateInput;
  const eligible = deriveEligibleDecisionStates(input);

  assert.equal(eligible.includes("WAIT_FOR_EVIDENCE"), true);
});

test("REFRESH_REQUIRED is tied to stale current decision support", () => {
  const input = fixtures.cases[3].input as DecisionStateInput;
  const eligible = deriveEligibleDecisionStates(input);

  assert.equal(eligible.includes("REFRESH_REQUIRED"), true);
});

test("REJECT requires structural failure or thesis-breaking evidence", () => {
  const base = structuredClone(
    fixtures.cases[4].input,
  ) as DecisionStateInput;

  base.structuralEconomicsFail = {
    state: "NO",
    rationale: "No structural failure established.",
    evidenceIds: ["E-STR-NO"],
    assumptionIds: [],
  };
  base.thesisBreakingEvidence = {
    state: "NO",
    rationale: "No thesis-breaking evidence established.",
    evidenceIds: ["E-TH-NO"],
    assumptionIds: [],
  };

  const output = evaluateDecisionStateArchitecture(base);

  assert.equal(output.finalizable, false);
  assert.equal(
    output.validationCodes.includes(
      "PROPOSED_DECISION_INCOMPATIBLE_WITH_BASIS",
    ),
    true,
  );
});

test("deterministic architecture does not invent priority when multiple canonical states are plausible", () => {
  const output = evaluateDecisionStateArchitecture(
    fixtures.cases[5].input as DecisionStateInput,
  );

  assert.equal(output.selectionAmbiguous, true);
  assert.equal(output.finalizable, false);
  assert.deepEqual(output.eligibleCanonicalStates, [
    "REFRESH_REQUIRED",
    "REJECT",
  ]);
});

test("UNKNOWN remains explicit and cannot pass finalization", () => {
  const output = evaluateDecisionStateArchitecture(
    fixtures.cases[6].input as DecisionStateInput,
  );

  assert.equal(output.proposedDecisionState, "UNKNOWN");
  assert.equal(output.finalizable, false);

  assert.throws(
    () => assertDecisionStateFinalizable(output),
    /VNEXT_DECISION_STATE_NOT_FINALIZABLE/,
  );
});

test("finalizable canonical state passes the finalization assertion", () => {
  const output = evaluateDecisionStateArchitecture(
    fixtures.cases[0].input as DecisionStateInput,
  );

  assert.doesNotThrow(() =>
    assertDecisionStateFinalizable(output),
  );
});

test("YES basis claims require traceable evidence", () => {
  const base = structuredClone(
    fixtures.cases[0].input,
  ) as DecisionStateInput;

  base.investmentPolicySatisfied = {
    ...base.investmentPolicySatisfied,
    evidenceIds: [],
  };

  assert.throws(
    () => evaluateDecisionStateArchitecture(base),
    /VNEXT_DECISION_INVESTMENT_POLICY_YES_REQUIRES_EVIDENCE/,
  );
});

test("UNKNOWN basis claims require traceable evidence or assumptions", () => {
  const base = structuredClone(
    fixtures.cases[6].input,
  ) as DecisionStateInput;

  base.valuationAdequate = {
    ...base.valuationAdequate,
    evidenceIds: [],
    assumptionIds: [],
  };

  assert.throws(
    () => evaluateDecisionStateArchitecture(base),
    /VNEXT_DECISION_VALUATION_ADEQUATE_UNKNOWN_REQUIRES_TRACEABLE_BASIS/,
  );
});
