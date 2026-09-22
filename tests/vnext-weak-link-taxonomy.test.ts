import assert from "node:assert/strict";
import test from "node:test";

import moduleContract from "../contracts/orotitan-equity/vnext/modules/WEAK_LINK_TAXONOMY.module-contract.v0.1.json";
import fixtures from "./fixtures/vnext/weak-link-taxonomy.v0.1.json";

import {
  deriveMaterialWeakLink,
  evaluateWeakLinkTaxonomy,
  type WeakLinkTaxonomyInput,
} from "../runtime/vnext/modules/weak-link-taxonomy";
import { assertValidModuleContract } from "../runtime/vnext/module-contract";

test("Weak Link Taxonomy module contract is valid under frozen Gate 7 schema", () => {
  assert.doesNotThrow(() => assertValidModuleContract(moduleContract));
});

for (const fixture of fixtures.cases) {
  test(`Weak Link fixture: ${fixture.id}`, () => {
    const result = evaluateWeakLinkTaxonomy(
      fixture.input as WeakLinkTaxonomyInput,
    );

    assert.equal(
      result.materialWeakLink,
      fixture.expected.materialWeakLink,
    );
    assert.equal(
      result.terminalSignal,
      fixture.expected.terminalSignal,
    );
  });
}

test("frozen conjunction returns YES only when all three criteria hold", () => {
  assert.equal(
    deriveMaterialWeakLink("YES", "YES", "YES"),
    "YES",
  );

  assert.equal(
    deriveMaterialWeakLink("YES", "NO", "YES"),
    "NO",
  );

  assert.equal(
    deriveMaterialWeakLink("YES", "YES", "UNKNOWN"),
    "UNKNOWN",
  );
});

test("a NO criterion resolves the conjunction to NO even if another criterion is UNKNOWN", () => {
  assert.equal(
    deriveMaterialWeakLink("UNKNOWN", "NO", "YES"),
    "NO",
  );
});

test("YES criterion requires traceable supporting evidence", () => {
  const base = structuredClone(
    fixtures.cases[0].input,
  ) as WeakLinkTaxonomyInput;

  base.causality = {
    ...base.causality,
    evidenceIds: [],
  };

  assert.throws(
    () => evaluateWeakLinkTaxonomy(base),
    /VNEXT_WEAK_LINK_CAUSALITY_YES_REQUIRES_EVIDENCE/,
  );
});

test("UNKNOWN must remain traceable rather than becoming an empty placeholder", () => {
  const base = structuredClone(
    fixtures.cases[2].input,
  ) as WeakLinkTaxonomyInput;

  base.unresolvedness = {
    ...base.unresolvedness,
    evidenceIds: [],
    contradictingEvidenceIds: [],
  };

  assert.throws(
    () => evaluateWeakLinkTaxonomy(base),
    /VNEXT_WEAK_LINK_UNRESOLVEDNESS_UNKNOWN_REQUIRES_TRACEABLE_BASIS/,
  );
});

test("supporting and contradicting evidence cannot silently overlap", () => {
  const base = structuredClone(
    fixtures.cases[0].input,
  ) as WeakLinkTaxonomyInput;

  base.materiality = {
    ...base.materiality,
    contradictingEvidenceIds: ["E-101"],
  };

  assert.throws(
    () => evaluateWeakLinkTaxonomy(base),
    /VNEXT_WEAK_LINK_MATERIALITY_EVIDENCE_CONTRADICTION_OVERLAP/,
  );
});

test("weak-link module emits a signal but never sets OroTitan status directly", () => {
  const result = evaluateWeakLinkTaxonomy(
    fixtures.cases[0].input as WeakLinkTaxonomyInput,
  );

  assert.equal(
    result.terminalSignal,
    "OROTITAN_STATUS_NO_REQUIRED",
  );

  assert.equal(
    Object.prototype.hasOwnProperty.call(result, "orotitanStatus"),
    false,
  );
});

test("aggregated evidence and assumptions are deterministic, unique, and sorted", () => {
  const result = evaluateWeakLinkTaxonomy(
    fixtures.cases[0].input as WeakLinkTaxonomyInput,
  );

  assert.deepEqual(result.supportingEvidenceIds, [
    "E-101",
    "E-103",
    "E-104",
  ]);
  assert.deepEqual(result.contradictingEvidenceIds, [
    "E-102",
    "E-105",
  ]);
  assert.deepEqual(result.assumptionIds, ["A-007"]);
});
