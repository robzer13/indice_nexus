import assert from "node:assert/strict";
import test from "node:test";

import {
  assertValidModuleContract,
  validateModuleContract,
  type VNextModuleContract,
} from "../runtime/vnext/module-contract";

function validAnalyticalContract(): VNextModuleContract {
  return {
    module_contract_schema_version: "0.1.0",
    module_id: "MOAT_PROOF",
    module_version: "0.1.0",
    module_kind: "ANALYTICAL",
    stage_code: "DEEP_DIVE",
    purpose: "Execute the frozen moat-proof analytical block.",
    authority: {
      reads: ["EVIDENCE_LEDGER"],
      writes: ["MOAT_PROOF_OUTPUT"],
      judgment_authority: true,
      deterministic_authority: false,
      forbidden_actions: [
        "PRODUCTION_PUBLICATION",
        "PRODUCTION_REGISTRY_MUTATION",
      ],
    },
    inputs: {
      required_artifact_types: ["EVIDENCE_LEDGER"],
      required_exact_references: true,
      required_stage_state: "IN_PROGRESS",
      required_contract_pins: true,
      data_cutoff_policy: "INHERIT_RUN_DATA_CUTOFF",
    },
    dependencies: {
      module_ids: ["EVIDENCE_LEDGER"],
      dependency_mode: "ALL_DECLARED",
    },
    execution: {
      status_authority: "DEEP_DIVE_BLOCK_LIFECYCLE",
      status_vocabulary: [
        "INSUFFICIENT",
        "IN_PROGRESS",
        "PROVISIONALLY_STABLE",
        "LOCKED",
      ],
      can_parallelize: false,
      checkpoint_policy: "MODULE_ARTIFACT_ONLY",
      retry_policy: "NO_RETRY",
      fail_closed: true,
    },
    outputs: {
      artifact_types: ["MOAT_PROOF_OUTPUT"],
      authority_class: "AUTHORITATIVE_STAGE_OUTPUT",
      persistence_required: true,
      hash_required: true,
    },
    uncertainty: {
      unknown_policy: "PRESERVE_EXPLICIT",
      not_applicable_policy: "REQUIRE_SEMANTIC_REASON",
      missing_policy: "PRESERVE_AND_BLOCK_WHEN_REQUIRED",
      assumption_policy: "EXPLICIT_TRACEABLE_NEVER_FABRICATE_TO_COMPLETE",
    },
    recovery: {
      recoverable_failures: ["TRANSIENT_SOURCE_FAILURE"],
      reopen_triggers: ["NEW_MATERIAL_COUNTEREVIDENCE"],
      recovery_action: "PINNED_CONTRACT",
    },
    environment: {
      allowed_environment: "VNEXT_SHADOW",
      allowed_supabase_project_ref: "awgsurdyvsyolcgpnygh",
      production_write_allowed: false,
    },
  };
}

test("Gate 7 accepts a conforming analytical Module Contract", () => {
  const contract = validAnalyticalContract();

  assert.deepEqual(validateModuleContract(contract), {
    valid: true,
    errors: [],
    schemaErrors: [],
  });

  assert.doesNotThrow(() => assertValidModuleContract(contract));
});

test("Module Contract schema fails closed on production environment drift", () => {
  const contract = validAnalyticalContract() as unknown as Record<
    string,
    unknown
  >;

  contract.environment = {
    allowed_environment: "PRODUCTION",
    allowed_supabase_project_ref: "cugpgtzygqqlxetyeven",
    production_write_allowed: true,
  };

  const result = validateModuleContract(contract);
  assert.equal(result.valid, false);
  assert.ok(result.errors.includes("MODULE_CONTRACT_SCHEMA_INVALID"));
  assert.ok(result.schemaErrors.length > 0);
});

test("Deep Dive block lifecycle requires the frozen exact vocabulary", () => {
  const contract = validAnalyticalContract();
  contract.execution.status_vocabulary = [
    "IN_PROGRESS",
    "LOCKED",
  ];

  const result = validateModuleContract(contract);
  assert.equal(result.valid, false);
  assert.ok(
    result.errors.includes(
      "MODULE_CONTRACT_DEEP_DIVE_LIFECYCLE_VOCABULARY_MISMATCH",
    ),
  );
});

test("Deep Dive block lifecycle cannot be attached to another stage", () => {
  const contract = validAnalyticalContract();
  contract.stage_code = "RESEARCH";

  const result = validateModuleContract(contract);
  assert.equal(result.valid, false);
  assert.ok(
    result.errors.includes(
      "MODULE_CONTRACT_DEEP_DIVE_LIFECYCLE_REQUIRES_DEEP_DIVE_STAGE",
    ),
  );
  assert.ok(
    result.errors.includes(
      "MODULE_CONTRACT_ANALYTICAL_OR_CERTIFICATION_REQUIRES_DEEP_DIVE_STAGE",
    ),
  );
});

test("authoritative stage output requires persisted hashed bytes", () => {
  const contract = validAnalyticalContract();
  contract.outputs.persistence_required = false;
  contract.outputs.hash_required = false;

  const result = validateModuleContract(contract);
  assert.equal(result.valid, false);
  assert.ok(
    result.errors.includes(
      "MODULE_CONTRACT_AUTHORITATIVE_OUTPUT_REQUIRES_PERSISTENCE_AND_HASH",
    ),
  );
});

test("self dependency is forbidden", () => {
  const contract = validAnalyticalContract();
  contract.dependencies.module_ids = ["MOAT_PROOF"];

  const result = validateModuleContract(contract);
  assert.equal(result.valid, false);
  assert.ok(
    result.errors.includes("MODULE_CONTRACT_SELF_DEPENDENCY_FORBIDDEN"),
  );
});

test("deterministic module cannot claim analyst judgment", () => {
  const contract = validAnalyticalContract();
  contract.module_id = "SCHEMA_VALIDATOR";
  contract.module_kind = "DETERMINISTIC";
  contract.authority.judgment_authority = true;
  contract.authority.deterministic_authority = true;
  contract.execution.status_authority = "TECHNICAL_DIAGNOSTIC";
  contract.execution.status_vocabulary = ["PASS", "FAIL"];

  const result = validateModuleContract(contract);
  assert.equal(result.valid, false);
  assert.ok(
    result.errors.includes(
      "MODULE_CONTRACT_DETERMINISTIC_MODULE_CANNOT_OWN_JUDGMENT",
    ),
  );
});

test("deterministic module must declare deterministic authority", () => {
  const contract = validAnalyticalContract();
  contract.module_id = "SCHEMA_VALIDATOR";
  contract.module_kind = "DETERMINISTIC";
  contract.authority.judgment_authority = false;
  contract.authority.deterministic_authority = false;
  contract.execution.status_authority = "TECHNICAL_DIAGNOSTIC";
  contract.execution.status_vocabulary = ["PASS", "FAIL"];

  const result = validateModuleContract(contract);
  assert.equal(result.valid, false);
  assert.ok(
    result.errors.includes(
      "MODULE_CONTRACT_DETERMINISTIC_MODULE_REQUIRES_DETERMINISTIC_AUTHORITY",
    ),
  );
});

test("module kinds remain bound to their architectural stages", () => {
  const research = validAnalyticalContract();
  research.module_id = "SOURCE_GATHERING";
  research.module_kind = "RESEARCH";
  research.stage_code = "DEEP_DIVE";
  research.execution.status_authority = "PINNED_CONTRACT";
  research.execution.status_vocabulary = ["IN_PROGRESS"];

  assert.ok(
    validateModuleContract(research).errors.includes(
      "MODULE_CONTRACT_RESEARCH_KIND_REQUIRES_RESEARCH_STAGE",
    ),
  );

  const integration = validAnalyticalContract();
  integration.module_id = "SNAPSHOT_BUILDER";
  integration.module_kind = "INTEGRATION";
  integration.stage_code = "DEEP_DIVE";
  integration.execution.status_authority = "PINNED_CONTRACT";
  integration.execution.status_vocabulary = ["IN_PROGRESS"];

  assert.ok(
    validateModuleContract(integration).errors.includes(
      "MODULE_CONTRACT_INTEGRATION_KIND_REQUIRES_INTEGRATION_STAGE",
    ),
  );
});
