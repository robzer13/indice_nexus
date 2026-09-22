export const VNEXT_P0_MODULE_ORDER = [
  "WEAK_LINK_TAXONOMY",
  "DECISION_STATE_ARCHITECTURE",
  "RETURN_NORMALIZATION",
  "CAPITAL_SEASONING",
  "OWNER_CASH",
  "FINANCING_CONSISTENCY",
  "VALUATION_ASSUMPTION_INTEGRITY",
  "VALUATION_DIAGNOSTIC_INTEGRITY",
] as const;

export type VnextP0ModuleId =
  (typeof VNEXT_P0_MODULE_ORDER)[number];

export interface VnextP0ModuleManifestEntry {
  moduleId: VnextP0ModuleId;
  version: "0.1.0";
  contractPath: string;
  runtimePath: string;
  fixturePath: string;
  testPath: string;
  providerDependency: "NONE";
  productionWriteAllowed: false;
  shadowRunMutationAllowedDuringGate15: false;
}

export const VNEXT_P0_MODULE_MANIFEST: readonly VnextP0ModuleManifestEntry[] =
  [
    {
      moduleId: "WEAK_LINK_TAXONOMY",
      version: "0.1.0",
      contractPath:
        "contracts/orotitan-equity/vnext/modules/WEAK_LINK_TAXONOMY.module-contract.v0.1.json",
      runtimePath: "runtime/vnext/modules/weak-link-taxonomy.ts",
      fixturePath:
        "tests/fixtures/vnext/weak-link-taxonomy.v0.1.json",
      testPath: "tests/vnext-weak-link-taxonomy.test.ts",
      providerDependency: "NONE",
      productionWriteAllowed: false,
      shadowRunMutationAllowedDuringGate15: false,
    },
    {
      moduleId: "DECISION_STATE_ARCHITECTURE",
      version: "0.1.0",
      contractPath:
        "contracts/orotitan-equity/vnext/modules/DECISION_STATE_ARCHITECTURE.module-contract.v0.1.json",
      runtimePath:
        "runtime/vnext/modules/decision-state-architecture.ts",
      fixturePath:
        "tests/fixtures/vnext/decision-state-architecture.v0.1.json",
      testPath:
        "tests/vnext-decision-state-architecture.test.ts",
      providerDependency: "NONE",
      productionWriteAllowed: false,
      shadowRunMutationAllowedDuringGate15: false,
    },
    {
      moduleId: "RETURN_NORMALIZATION",
      version: "0.1.0",
      contractPath:
        "contracts/orotitan-equity/vnext/modules/RETURN_NORMALIZATION.module-contract.v0.1.json",
      runtimePath:
        "runtime/vnext/modules/return-normalization.ts",
      fixturePath:
        "tests/fixtures/vnext/return-normalization.v0.1.json",
      testPath: "tests/vnext-return-normalization.test.ts",
      providerDependency: "NONE",
      productionWriteAllowed: false,
      shadowRunMutationAllowedDuringGate15: false,
    },
    {
      moduleId: "CAPITAL_SEASONING",
      version: "0.1.0",
      contractPath:
        "contracts/orotitan-equity/vnext/modules/CAPITAL_SEASONING.module-contract.v0.1.json",
      runtimePath: "runtime/vnext/modules/capital-seasoning.ts",
      fixturePath:
        "tests/fixtures/vnext/capital-seasoning.v0.1.json",
      testPath: "tests/vnext-capital-seasoning.test.ts",
      providerDependency: "NONE",
      productionWriteAllowed: false,
      shadowRunMutationAllowedDuringGate15: false,
    },
    {
      moduleId: "OWNER_CASH",
      version: "0.1.0",
      contractPath:
        "contracts/orotitan-equity/vnext/modules/OWNER_CASH.module-contract.v0.1.json",
      runtimePath: "runtime/vnext/modules/owner-cash.ts",
      fixturePath: "tests/fixtures/vnext/owner-cash.v0.1.json",
      testPath: "tests/vnext-owner-cash.test.ts",
      providerDependency: "NONE",
      productionWriteAllowed: false,
      shadowRunMutationAllowedDuringGate15: false,
    },
    {
      moduleId: "FINANCING_CONSISTENCY",
      version: "0.1.0",
      contractPath:
        "contracts/orotitan-equity/vnext/modules/FINANCING_CONSISTENCY.module-contract.v0.1.json",
      runtimePath:
        "runtime/vnext/modules/financing-consistency.ts",
      fixturePath:
        "tests/fixtures/vnext/financing-consistency.v0.1.json",
      testPath: "tests/vnext-financing-consistency.test.ts",
      providerDependency: "NONE",
      productionWriteAllowed: false,
      shadowRunMutationAllowedDuringGate15: false,
    },
    {
      moduleId: "VALUATION_ASSUMPTION_INTEGRITY",
      version: "0.1.0",
      contractPath:
        "contracts/orotitan-equity/vnext/modules/VALUATION_ASSUMPTION_INTEGRITY.module-contract.v0.1.json",
      runtimePath:
        "runtime/vnext/modules/valuation-assumption-integrity.ts",
      fixturePath:
        "tests/fixtures/vnext/valuation-assumption-integrity.v0.1.json",
      testPath:
        "tests/vnext-valuation-assumption-integrity.test.ts",
      providerDependency: "NONE",
      productionWriteAllowed: false,
      shadowRunMutationAllowedDuringGate15: false,
    },
    {
      moduleId: "VALUATION_DIAGNOSTIC_INTEGRITY",
      version: "0.1.0",
      contractPath:
        "contracts/orotitan-equity/vnext/modules/VALUATION_DIAGNOSTIC_INTEGRITY.module-contract.v0.1.json",
      runtimePath:
        "runtime/vnext/modules/valuation-diagnostic-integrity.ts",
      fixturePath:
        "tests/fixtures/vnext/valuation-diagnostic-integrity.v0.1.json",
      testPath:
        "tests/vnext-valuation-diagnostic-integrity.test.ts",
      providerDependency: "NONE",
      productionWriteAllowed: false,
      shadowRunMutationAllowedDuringGate15: false,
    },
  ] as const;
