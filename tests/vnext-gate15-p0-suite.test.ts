import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import weakLink from "../contracts/orotitan-equity/vnext/modules/WEAK_LINK_TAXONOMY.module-contract.v0.1.json";
import decisionState from "../contracts/orotitan-equity/vnext/modules/DECISION_STATE_ARCHITECTURE.module-contract.v0.1.json";
import returnNormalization from "../contracts/orotitan-equity/vnext/modules/RETURN_NORMALIZATION.module-contract.v0.1.json";
import capitalSeasoning from "../contracts/orotitan-equity/vnext/modules/CAPITAL_SEASONING.module-contract.v0.1.json";
import ownerCash from "../contracts/orotitan-equity/vnext/modules/OWNER_CASH.module-contract.v0.1.json";
import financingConsistency from "../contracts/orotitan-equity/vnext/modules/FINANCING_CONSISTENCY.module-contract.v0.1.json";
import valuationAssumption from "../contracts/orotitan-equity/vnext/modules/VALUATION_ASSUMPTION_INTEGRITY.module-contract.v0.1.json";
import valuationDiagnostic from "../contracts/orotitan-equity/vnext/modules/VALUATION_DIAGNOSTIC_INTEGRITY.module-contract.v0.1.json";

import { assertValidModuleContract } from "../runtime/vnext/module-contract";
import {
  VNEXT_P0_MODULE_MANIFEST,
  VNEXT_P0_MODULE_ORDER,
} from "../runtime/vnext/modules/p0-suite";

const contracts = [
  weakLink,
  decisionState,
  returnNormalization,
  capitalSeasoning,
  ownerCash,
  financingConsistency,
  valuationAssumption,
  valuationDiagnostic,
] as const;

test("Gate 15 P0 suite contains exactly the frozen eight modules in order", () => {
  assert.deepEqual(
    VNEXT_P0_MODULE_MANIFEST.map((entry) => entry.moduleId),
    VNEXT_P0_MODULE_ORDER,
  );
  assert.equal(VNEXT_P0_MODULE_MANIFEST.length, 8);
  assert.equal(
    new Set(VNEXT_P0_MODULE_ORDER).size,
    VNEXT_P0_MODULE_ORDER.length,
  );
});

test("all Gate 15 P0 contracts remain valid under frozen Gate 7 Module Contract schema", () => {
  for (const contract of contracts) {
    assert.doesNotThrow(() => assertValidModuleContract(contract));
  }
});

test("manifest identity matches exact module-contract identity and version", () => {
  for (let index = 0; index < contracts.length; index += 1) {
    const contract = contracts[index];
    const manifest = VNEXT_P0_MODULE_MANIFEST[index];

    assert.ok(contract);
    assert.ok(manifest);
    assert.equal(contract.module_id, manifest.moduleId);
    assert.equal(contract.module_version, manifest.version);
  }
});

test("Gate 15 P0 dependency graph is forward-only in the frozen suite order", () => {
  const order = new Map(
    VNEXT_P0_MODULE_ORDER.map((moduleId, index) => [
      moduleId,
      index,
    ]),
  );

  for (const contract of contracts) {
    const currentIndex = order.get(contract.module_id);
    assert.notEqual(currentIndex, undefined);

    for (const dependency of contract.dependencies.module_ids) {
      const dependencyIndex = order.get(
        dependency as (typeof VNEXT_P0_MODULE_ORDER)[number],
      );

      if (dependencyIndex === undefined) {
        continue;
      }

      assert.ok(
        dependencyIndex < (currentIndex as number),
        `${contract.module_id} cannot depend on later P0 module ${dependency}`,
      );
    }
  }
});

test("every Gate 15 P0 module is shadow-only and has no production write authority", () => {
  for (const contract of contracts) {
    assert.equal(
      contract.environment.allowed_environment,
      "VNEXT_SHADOW",
    );
    assert.equal(
      contract.environment.allowed_supabase_project_ref,
      "awgsurdyvsyolcgpnygh",
    );
    assert.equal(contract.environment.production_write_allowed, false);
  }

  for (const entry of VNEXT_P0_MODULE_MANIFEST) {
    assert.equal(entry.productionWriteAllowed, false);
    assert.equal(entry.shadowRunMutationAllowedDuringGate15, false);
  }
});

test("Gate 15 P0 runtimes have no provider, Azure, network, Supabase or environment dependency", () => {
  const forbidden = [
    /AnalyticalModelProvider/,
    /AzureProvider/,
    /openai\.azure\.com/i,
    /services\.ai\.azure\.com/i,
    /createClient\s*\(/,
    /supabase/i,
    /process\.env/,
    /fetch\s*\(/,
  ];

  for (const entry of VNEXT_P0_MODULE_MANIFEST) {
    const source = readFileSync(entry.runtimePath, "utf8");

    for (const pattern of forbidden) {
      assert.equal(
        pattern.test(source),
        false,
        `${entry.moduleId} unexpectedly matches forbidden runtime dependency ${pattern}`,
      );
    }
  }
});

test("Gate 15 P0 suite does not grant publication authority", () => {
  for (const contract of contracts) {
    assert.equal(
      contract.authority.forbidden_actions.includes(
        "WRITE_PRODUCTION",
      ),
      true,
    );
  }
});
