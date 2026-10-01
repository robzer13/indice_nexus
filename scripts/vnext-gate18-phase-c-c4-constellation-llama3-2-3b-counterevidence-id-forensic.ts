import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { execFileSync } from "node:child_process";

import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import type {
  Gate18ArtifactPin,
  Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";
import {
  buildVerifiedGate18V10MoatPacket,
  gate18PhaseBV10OutputSchema,
} from "../runtime/vnext/model-calibration-pilot-v10";
import { evaluateGate18V11Validation } from "../runtime/vnext/model-calibration-validation-v11";

const AUTHORIZATION_ID =
  "G18-PHASEC-C4-CONSTELLATION-LLAMA3_2-3B-COUNTEREVIDENCE-ID-FORENSIC-AUTH-001";
const AUTHORIZATION_PATH =
  "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_LLAMA3_2_3B_COUNTEREVIDENCE_ID_FORENSIC_AUTH_001.json";
const EXPECTED_MATRIX_CELL_ID =
  "C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_LLAMA3_2_3B_V1_1_CONTEXT16384_001";
const EXPECTED_ATTEMPT_ID =
  "C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_LLAMA3_2_3B_V1_1_CONTEXT16384_OUTPUT1024_TIMEOUT600_LOOPBACK_GUARDED_001";
const EXPECTED_COMPANY = "Constellation Software";
const EXPECTED_MODEL = "llama3.2:3b-instruct-q4_K_M";
const EXPECTED_PROMPT_SHA256 =
  "0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8";
const EXPECTED_PACKET_SHA256 =
  "9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8";
const EXPECTED_RAW_SHA256 =
  "f0cfb503a74b895ddfed73ca8030ce378a4b216c5c982b2c01def50f14acbefd";
const EXPECTED_SCHEMA_ERROR =
  "VNEXT_GATE18_V11_RAW_SCHEMA_INVALID";
const EVIDENCE_ID_PATTERN = /^E-\d{3}$/;
const EVIDENCE_ID_GLOBAL_PATTERN = /E-\d{3}/g;
const SEPARATOR_ONLY_PATTERN =
  /^[\s,;|/&+()[\]{}"'\`:-]*$/;

interface CliOptions {
  privateRepoRoot: string;
  privateOutputPath: string;
  authorizationId: string | null;
}

interface AuthorizationArtifact {
  authorization_id?: string;
  status?: string;
  scope?: {
    read_existing_private_artifact?: boolean;
    read_pinned_private_packet?: boolean;
    inspect_only_malformed_counterevidence_id_path?: string;
    extract_canonical_evidence_id_tokens?: boolean;
    compare_extracted_ids_to_packet?: boolean;
    diagnostic_in_memory_replacement_if_unambiguous?: boolean;
    rerun_frozen_v11_validator_on_in_memory_copy?: boolean;
    print_raw_malformed_value?: boolean;
    print_raw_generated_narrative?: boolean;
    mutate_source_artifact?: boolean;
    execute_model_inference?: boolean;
    call_ollama?: boolean;
    external_network_access?: boolean;
    retry_inference?: boolean;
  };
  constraints?: {
    authorized_run_count?: number;
  };
}

interface PrivateRunArtifact {
  status?: string;
  invocation?: {
    matrixCellId?: string;
    attemptId?: string;
    company?: string;
    model?: string;
    promptSha256?: string;
    fullPacketSha256?: string;
  };
  execution?: {
    doneReason?: string | null;
    promptEvalCount?: number | null;
    evalCount?: number | null;
    runtimeError?: string | null;
    schemaValid?: boolean;
    schemaError?: string | null;
    semanticValid?: boolean;
    semanticError?: string | null;
  };
  response?: {
    rawText?: string | null;
    parsedJson?: unknown;
  };
  validationV11?: {
    rawSchemaPass?: boolean;
    rawSchemaError?: string | null;
    substantiveValidation?: {
      status?: string;
      pass?: boolean | null;
      error?: string | null;
    };
  } | null;
}

function parseArgs(argv: readonly string[]): CliOptions {
  let privateRepoRoot = "";
  let privateOutputPath = "";
  let authorizationId: string | null = null;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === "--private-repo-root") {
      privateRepoRoot = argv[++index] ?? "";
      continue;
    }

    if (arg === "--private-output-path") {
      privateOutputPath = argv[++index] ?? "";
      continue;
    }

    if (arg === "--authorization-id") {
      authorizationId = argv[++index] ?? null;
      continue;
    }

    throw new Error(
      `VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_UNKNOWN_ARG:${arg}`,
    );
  }

  if (!privateRepoRoot.trim()) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }

  if (!privateOutputPath.trim()) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_PRIVATE_OUTPUT_REQUIRED",
    );
  }

  return {
    privateRepoRoot,
    privateOutputPath,
    authorizationId,
  };
}

function findConstellation(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) => company.display_name === EXPECTED_COMPANY,
  );

  if (matches.length !== 1) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_COMPANY_NOT_UNIQUE",
    );
  }

  return matches[0] as Gate18PilotCompany;
}

function artifactReader(
  privateRepoRoot: string,
): (pin: Gate18ArtifactPin) => Uint8Array {
  return (pin) => {
    const normalizedRoot = resolve(privateRepoRoot).replaceAll("\\", "/");
    const absolute = resolve(privateRepoRoot, pin.path);
    const normalizedAbsolute = absolute.replaceAll("\\", "/");
    const prefix = normalizedRoot.endsWith("/")
      ? normalizedRoot
      : `${normalizedRoot}/`;

    if (!normalizedAbsolute.startsWith(prefix)) {
      throw new Error(
        "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_PRIVATE_PATH_ESCAPE",
      );
    }

    execFileSync(
      "git",
      [
        "-C",
        privateRepoRoot,
        "cat-file",
        "-e",
        `${pin.commit_sha}^{commit}`,
      ],
      {
        stdio: ["ignore", "ignore", "pipe"],
      },
    );

    return execFileSync(
      "git",
      [
        "-C",
        privateRepoRoot,
        "show",
        `${pin.commit_sha}:${pin.path.replaceAll("\\", "/")}`,
      ],
      {
        encoding: "buffer",
        maxBuffer: 20 * 1024 * 1024,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
  };
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function main(): void {
  const options = parseArgs(process.argv.slice(2));

  if (options.authorizationId !== AUTHORIZATION_ID) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_AUTHORIZATION_ID_MISMATCH",
    );
  }

  const authAbsolute = resolve(process.cwd(), AUTHORIZATION_PATH);
  if (!existsSync(authAbsolute)) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_AUTHORIZATION_MISSING",
    );
  }

  const auth = JSON.parse(
    readFileSync(authAbsolute, "utf8"),
  ) as AuthorizationArtifact;

  if (
    auth.authorization_id !== AUTHORIZATION_ID ||
    auth.status !== "AUTHORIZED_READ_ONLY_NO_INFERENCE" ||
    auth.constraints?.authorized_run_count !== 1 ||
    auth.scope?.read_existing_private_artifact !== true ||
    auth.scope.read_pinned_private_packet !== true ||
    auth.scope.inspect_only_malformed_counterevidence_id_path !==
      "priority_findings.1.counterevidence_ids.0" ||
    auth.scope.extract_canonical_evidence_id_tokens !== true ||
    auth.scope.compare_extracted_ids_to_packet !== true ||
    auth.scope.diagnostic_in_memory_replacement_if_unambiguous !== true ||
    auth.scope.rerun_frozen_v11_validator_on_in_memory_copy !== true ||
    auth.scope.print_raw_malformed_value !== false ||
    auth.scope.print_raw_generated_narrative !== false ||
    auth.scope.mutate_source_artifact !== false ||
    auth.scope.execute_model_inference !== false ||
    auth.scope.call_ollama !== false ||
    auth.scope.external_network_access !== false ||
    auth.scope.retry_inference !== false
  ) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_AUTHORIZATION_INVALID",
    );
  }

  const privateAbsolute = resolve(options.privateOutputPath);
  if (!existsSync(privateAbsolute)) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_PRIVATE_OUTPUT_NOT_FOUND",
    );
  }

  const run = JSON.parse(
    readFileSync(privateAbsolute, "utf8"),
  ) as PrivateRunArtifact;

  if (
    run.status !== "FAIL" ||
    run.invocation?.matrixCellId !== EXPECTED_MATRIX_CELL_ID ||
    run.invocation?.attemptId !== EXPECTED_ATTEMPT_ID ||
    run.invocation?.company !== EXPECTED_COMPANY ||
    run.invocation?.model !== EXPECTED_MODEL ||
    run.invocation?.promptSha256 !== EXPECTED_PROMPT_SHA256 ||
    run.invocation?.fullPacketSha256 !== EXPECTED_PACKET_SHA256 ||
    run.execution?.doneReason !== "stop" ||
    run.execution?.promptEvalCount !== 3157 ||
    run.execution?.evalCount !== 895 ||
    run.execution?.runtimeError !== null ||
    run.execution?.schemaValid !== false ||
    run.execution?.schemaError !== EXPECTED_SCHEMA_ERROR ||
    run.execution?.semanticError !== "NOT_EVALUATED_SCHEMA_FAILURE" ||
    run.validationV11?.rawSchemaPass !== false ||
    run.validationV11?.rawSchemaError !== EXPECTED_SCHEMA_ERROR
  ) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_RUN_IDENTITY_MISMATCH",
    );
  }

  const rawText = run.response?.rawText;
  const parsedJson = run.response?.parsedJson;

  if (
    typeof rawText !== "string" ||
    rawText.length === 0 ||
    sha256(rawText) !== EXPECTED_RAW_SHA256
  ) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_RAW_TEXT_IDENTITY_MISMATCH",
    );
  }

  const rawSchema = gate18PhaseBV10OutputSchema.safeParse(parsedJson);
  if (rawSchema.success) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_EXPECTED_RAW_SCHEMA_FAILURE",
    );
  }

  const rawIssues = rawSchema.error.issues;
  if (
    rawIssues.length !== 1 ||
    rawIssues[0]?.path.join(".") !==
      "priority_findings.1.counterevidence_ids.0" ||
    rawIssues[0]?.code !== "invalid_format"
  ) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_UNEXPECTED_SCHEMA_ISSUES",
    );
  }

  if (
    parsedJson === null ||
    typeof parsedJson !== "object" ||
    Array.isArray(parsedJson)
  ) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_PARSED_OBJECT_REQUIRED",
    );
  }

  const root = parsedJson as Record<string, unknown>;
  const findings = root.priority_findings;
  if (!Array.isArray(findings) || findings.length !== 3) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_FINDINGS_SHAPE_MISMATCH",
    );
  }

  const finding = findings[1];
  if (
    finding === null ||
    typeof finding !== "object" ||
    Array.isArray(finding)
  ) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_TARGET_FINDING_SHAPE_MISMATCH",
    );
  }

  const counterevidenceIds = (
    finding as Record<string, unknown>
  ).counterevidence_ids;

  if (
    !Array.isArray(counterevidenceIds) ||
    counterevidenceIds.length < 1 ||
    counterevidenceIds.length > 2 ||
    typeof counterevidenceIds[0] !== "string"
  ) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_TARGET_IDS_SHAPE_MISMATCH",
    );
  }

  const malformed = counterevidenceIds[0] as string;
  const extracted = malformed.match(EVIDENCE_ID_GLOBAL_PATTERN) ?? [];
  const uniqueExtracted = [...new Set(extracted)];
  const residual = malformed.replace(EVIDENCE_ID_GLOBAL_PATTERN, "");
  const separatorOnlyResidual = SEPARATOR_ONLY_PATTERN.test(residual);

  const verified = buildVerifiedGate18V10MoatPacket(
    findConstellation(),
    artifactReader(options.privateRepoRoot),
  );

  if (verified.packetSha256 !== EXPECTED_PACKET_SHA256) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_COUNTEREVIDENCE_ID_FORENSIC_PACKET_IDENTITY_MISMATCH",
    );
  }

  const knownEvidenceIds = new Set(
    verified.packet.evidence_items.map((item) => item.evidence_id),
  );

  const extractedKnownFlags = extracted.map((id) =>
    knownEvidenceIds.has(id),
  );
  const allExtractedKnown =
    extracted.length > 0 && extractedKnownFlags.every(Boolean);
  const extractedUnique =
    extracted.length === uniqueExtracted.length;

  const trailingIds = counterevidenceIds.slice(1);
  const trailingIdsValid = trailingIds.every(
    (value) =>
      typeof value === "string" &&
      EVIDENCE_ID_PATTERN.test(value) &&
      knownEvidenceIds.has(value),
  );

  const candidateIds = [
    ...extracted,
    ...(trailingIds as string[]),
  ];
  const candidateIdsUnique =
    candidateIds.length === new Set(candidateIds).size;

  const unambiguous =
    extracted.length >= 1 &&
    extracted.length <= 2 &&
    extractedUnique &&
    allExtractedKnown &&
    separatorOnlyResidual &&
    trailingIdsValid &&
    candidateIds.length <= 2 &&
    candidateIdsUnique;

  let diagnosticReplacementApplied = false;
  let repairedSchemaPass: boolean | null = null;
  let repairedSchemaError: string | null = null;
  let v11RawSchemaPass: boolean | null = null;
  let v11RawPresentationCompliant: boolean | null = null;
  let v11NormalizedPathCount: number | null = null;
  let v11SubstantiveStatus: string | null = null;
  let v11SubstantivePass: boolean | null = null;
  let v11SubstantiveError: string | null = null;

  if (unambiguous) {
    const diagnostic = structuredClone(
      parsedJson,
    ) as Record<string, unknown>;
    const diagnosticFindings =
      diagnostic.priority_findings as Array<Record<string, unknown>>;
    diagnosticFindings[1] = {
      ...diagnosticFindings[1],
      counterevidence_ids: candidateIds,
    };
    diagnostic.priority_findings = diagnosticFindings;

    diagnosticReplacementApplied = true;

    const repairedSchema =
      gate18PhaseBV10OutputSchema.safeParse(diagnostic);
    repairedSchemaPass = repairedSchema.success;

    if (!repairedSchema.success) {
      repairedSchemaError =
        repairedSchema.error.issues
          .map((issue) => `${issue.path.join(".")}:${issue.code}`)
          .join("|");
    } else {
      const v11 = evaluateGate18V11Validation(
        verified.packet,
        diagnostic,
        "FULL",
      );
      v11RawSchemaPass = v11.rawSchemaPass;
      v11RawPresentationCompliant =
        v11.rawPresentationCompliance?.compliant ?? null;
      v11NormalizedPathCount =
        v11.normalization.normalizedPathCount;
      v11SubstantiveStatus =
        v11.substantiveValidation.status;
      v11SubstantivePass =
        v11.substantiveValidation.pass;
      v11SubstantiveError =
        v11.substantiveValidation.error;
    }
  }

  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C4_LLAMA3_2_3B_COUNTEREVIDENCE_ID_FORENSIC_V0.1",
        status: "PASS_READ_ONLY_FORENSIC_COMPLETE",
        mode:
          "LOCAL_READ_ONLY_IN_MEMORY_DIAGNOSTIC_NO_INFERENCE",
        sourceRun: {
          matrixCellId: EXPECTED_MATRIX_CELL_ID,
          attemptId: EXPECTED_ATTEMPT_ID,
          company: EXPECTED_COMPANY,
          model: EXPECTED_MODEL,
          doneReason: run.execution?.doneReason ?? null,
          promptEvalCount:
            run.execution?.promptEvalCount ?? null,
          evalCount: run.execution?.evalCount ?? null,
          schemaError:
            run.execution?.schemaError ?? null,
          rawTextSha256: EXPECTED_RAW_SHA256,
        },
        malformedReference: {
          path:
            "priority_findings.1.counterevidence_ids.0",
          rawValuePrinted: false,
          rawValueLength: malformed.length,
          extractedCanonicalTokenCount: extracted.length,
          extractedCanonicalIds: extracted,
          allExtractedIdsExistInPacket: allExtractedKnown,
          extractedTokensUnique: extractedUnique,
          residualSeparatorOnly: separatorOnlyResidual,
          trailingCounterevidenceIdCount:
            trailingIds.length,
          trailingCounterevidenceIdsValid:
            trailingIdsValid,
          candidateCounterevidenceIds: candidateIds,
          candidateIdsUnique,
          unambiguousDeterministicReplacement:
            unambiguous,
        },
        diagnosticNormalization: {
          rule:
            "Replace only the single malformed counterevidence_ids[0] string when it is fully decomposable into one or two unique canonical packet E-NNN IDs plus separator characters, preserving any already-valid trailing counterevidence ID.",
          applied: diagnosticReplacementApplied,
          sourceArtifactMutated: false,
          rawOutputMutated: false,
          generatedNarrativePrinted: false,
          autoRepairExecuted: false,
          retroactivePassAllowed: false,
        },
        downstreamValidation: {
          repairedSchemaPass,
          repairedSchemaError,
          v11RawSchemaPass,
          v11RawPresentationCompliant,
          v11NormalizedPathCount,
          v11SubstantiveStatus,
          v11SubstantivePass,
          v11SubstantiveError,
        },
        interpretationBoundary: {
          originalRunStatusChanged: false,
          inferenceExecuted: false,
          retryAuthorized: false,
          modelCapabilityFailureConcluded: false,
          nextAction:
            "Use deterministic replacement admissibility and downstream V1.1 status to decide whether a single reliability retry is informative or Llama 3.2 should stop.",
        },
        safety: {
          externalNetworkAccessRequested: false,
          ollamaApiCalled: false,
          modelInferenceExecuted: false,
          sourceArtifactMutated: false,
          rawNarrativeContentPublished: false,
          productionMutation: false,
          publicationAuthority: false,
        },
      },
      null,
      2,
    ),
  );
}

main();
