import Ajv2020, { type ErrorObject } from "ajv/dist/2020";

import schema from "../../schemas/vnext/orotitan-vnext-module-contract.schema.v0.1.json";

export const MODULE_KINDS = [
  "RESEARCH",
  "ANALYTICAL",
  "DETERMINISTIC",
  "CERTIFICATION",
  "INTEGRATION",
  "ASSURANCE",
] as const;

export type ModuleKind = (typeof MODULE_KINDS)[number];

export const MODULE_STAGE_CODES = [
  "RESEARCH",
  "DEEP_DIVE",
  "INTEGRATION",
] as const;

export type ModuleStageCode = (typeof MODULE_STAGE_CODES)[number];

export interface VNextModuleContract {
  module_contract_schema_version: "0.1.0";
  module_id: string;
  module_version: string;
  module_kind: ModuleKind;
  stage_code: ModuleStageCode;
  purpose: string;
  authority: {
    reads: string[];
    writes: string[];
    judgment_authority: boolean;
    deterministic_authority: boolean;
    forbidden_actions: string[];
  };
  inputs: {
    required_artifact_types: string[];
    required_exact_references: true;
    required_stage_state: string | null;
    required_contract_pins: true;
    data_cutoff_policy: "INHERIT_RUN_DATA_CUTOFF";
  };
  dependencies: {
    module_ids: string[];
    dependency_mode:
      | "ALL_DECLARED"
      | "PINNED_CONTRACT"
      | "ANALYST_IMPACT_JUDGMENT";
  };
  execution: {
    status_authority:
      | "STAGE_STATE_MODEL"
      | "DEEP_DIVE_BLOCK_LIFECYCLE"
      | "PINNED_CONTRACT"
      | "TECHNICAL_DIAGNOSTIC";
    status_vocabulary: string[];
    can_parallelize: boolean;
    checkpoint_policy:
      | "STAGE_CHECKPOINT"
      | "MODULE_ARTIFACT_ONLY"
      | "NOT_APPLICABLE";
    retry_policy:
      | "IDEMPOTENT_SAME_REQUEST"
      | "NO_RETRY"
      | "PINNED_CONTRACT";
    fail_closed: true;
  };
  outputs: {
    artifact_types: string[];
    authority_class:
      | "AUTHORITATIVE_STAGE_OUTPUT"
      | "SUPPORTING_EXECUTION_ARTIFACT"
      | "TECHNICAL_ASSURANCE";
    persistence_required: boolean;
    hash_required: boolean;
  };
  uncertainty: {
    unknown_policy: "PRESERVE_EXPLICIT";
    not_applicable_policy: "REQUIRE_SEMANTIC_REASON";
    missing_policy: "PRESERVE_AND_BLOCK_WHEN_REQUIRED";
    assumption_policy: "EXPLICIT_TRACEABLE_NEVER_FABRICATE_TO_COMPLETE";
  };
  recovery: {
    recoverable_failures: string[];
    reopen_triggers: string[];
    recovery_action:
      | "RETRY"
      | "RESUME_STAGE"
      | "REOPEN_STAGE"
      | "SUCCESSOR_RUN"
      | "PINNED_CONTRACT";
  };
  environment: {
    allowed_environment: "VNEXT_SHADOW";
    allowed_supabase_project_ref: "awgsurdyvsyolcgpnygh";
    production_write_allowed: false;
  };
}

export interface ModuleContractValidationResult {
  valid: boolean;
  errors: readonly string[];
  schemaErrors: readonly ErrorObject[];
}

const ajv = new Ajv2020({
  allErrors: true,
  strict: true,
});

const validateSchema = ajv.compile(schema);

const DEEP_DIVE_BLOCK_LIFECYCLE = [
  "INSUFFICIENT",
  "IN_PROGRESS",
  "PROVISIONALLY_STABLE",
  "LOCKED",
] as const;

function equalVocabulary(actual: readonly string[], expected: readonly string[]) {
  return (
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  );
}

export function validateModuleContract(
  value: unknown,
): ModuleContractValidationResult {
  const schemaValid = validateSchema(value);

  if (!schemaValid) {
    return {
      valid: false,
      errors: ["MODULE_CONTRACT_SCHEMA_INVALID"],
      schemaErrors: [...(validateSchema.errors ?? [])],
    };
  }

  const contract = value as unknown as VNextModuleContract;
  const errors: string[] = [];

  if (contract.dependencies.module_ids.includes(contract.module_id)) {
    errors.push("MODULE_CONTRACT_SELF_DEPENDENCY_FORBIDDEN");
  }

  if (
    contract.execution.status_authority === "DEEP_DIVE_BLOCK_LIFECYCLE"
  ) {
    if (contract.stage_code !== "DEEP_DIVE") {
      errors.push(
        "MODULE_CONTRACT_DEEP_DIVE_LIFECYCLE_REQUIRES_DEEP_DIVE_STAGE",
      );
    }

    if (
      !equalVocabulary(
        contract.execution.status_vocabulary,
        DEEP_DIVE_BLOCK_LIFECYCLE,
      )
    ) {
      errors.push(
        "MODULE_CONTRACT_DEEP_DIVE_LIFECYCLE_VOCABULARY_MISMATCH",
      );
    }
  }

  if (
    contract.outputs.authority_class === "AUTHORITATIVE_STAGE_OUTPUT" &&
    (!contract.outputs.persistence_required || !contract.outputs.hash_required)
  ) {
    errors.push(
      "MODULE_CONTRACT_AUTHORITATIVE_OUTPUT_REQUIRES_PERSISTENCE_AND_HASH",
    );
  }

  if (
    contract.module_kind === "DETERMINISTIC" &&
    contract.authority.judgment_authority
  ) {
    errors.push(
      "MODULE_CONTRACT_DETERMINISTIC_MODULE_CANNOT_OWN_JUDGMENT",
    );
  }

  if (
    contract.module_kind === "DETERMINISTIC" &&
    !contract.authority.deterministic_authority
  ) {
    errors.push(
      "MODULE_CONTRACT_DETERMINISTIC_MODULE_REQUIRES_DETERMINISTIC_AUTHORITY",
    );
  }

  if (
    contract.module_kind === "RESEARCH" &&
    contract.stage_code !== "RESEARCH"
  ) {
    errors.push("MODULE_CONTRACT_RESEARCH_KIND_REQUIRES_RESEARCH_STAGE");
  }

  if (
    (contract.module_kind === "ANALYTICAL" ||
      contract.module_kind === "CERTIFICATION") &&
    contract.stage_code !== "DEEP_DIVE"
  ) {
    errors.push(
      "MODULE_CONTRACT_ANALYTICAL_OR_CERTIFICATION_REQUIRES_DEEP_DIVE_STAGE",
    );
  }

  if (
    contract.module_kind === "INTEGRATION" &&
    contract.stage_code !== "INTEGRATION"
  ) {
    errors.push(
      "MODULE_CONTRACT_INTEGRATION_KIND_REQUIRES_INTEGRATION_STAGE",
    );
  }

  return {
    valid: errors.length === 0,
    errors,
    schemaErrors: [],
  };
}

export function assertValidModuleContract(
  value: unknown,
): asserts value is VNextModuleContract {
  const result = validateModuleContract(value);

  if (!result.valid) {
    const schemaDetails = result.schemaErrors
      .map(
        (error) =>
          `${error.instancePath || "/"} ${error.message ?? error.keyword}`,
      )
      .join("; ");

    const details = [...result.errors, schemaDetails]
      .filter(Boolean)
      .join("; ");

    throw new Error(`VNEXT_MODULE_CONTRACT_INVALID: ${details}`);
  }
}
