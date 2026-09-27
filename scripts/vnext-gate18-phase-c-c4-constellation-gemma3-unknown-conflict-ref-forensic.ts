import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { execFileSync } from "node:child_process";

import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import type { Gate18ArtifactPin, Gate18PilotCompany } from "../runtime/vnext/model-calibration-pilot";
import {
  assertGate18PhaseBV10Semantics,
  buildVerifiedGate18V10MoatPacket,
  gate18PhaseBV10OutputSchema,
} from "../runtime/vnext/model-calibration-pilot-v10";
import {
  normalizeGate18V11Presentation,
  inspectGate18V11Presentation,
} from "../runtime/vnext/model-calibration-validation-v11";

const EXPECTED_MATRIX_CELL_ID =
  "C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_GEMMA3_4B_V1_1_CONTEXT16384_001";
const EXPECTED_ATTEMPT_ID =
  "C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_GEMMA3_4B_V1_1_CONTEXT16384_OUTPUT1024_TIMEOUT600_LOOPBACK_GUARDED_001";
const EXPECTED_COMPANY = "Constellation Software";
const EXPECTED_PROMPT_SHA256 =
  "0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8";
const EXPECTED_PACKET_SHA256 =
  "9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8";
const EXPECTED_FIRST_ERROR =
  "VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS";
const EXPECTED_SECOND_ERROR =
  "VNEXT_GATE18_V10_UNKNOWN_CONFLICT_REF";

function parseArgs(argv: readonly string[]) {
  let privateRepoRoot = "";
  let privateArtifact = "";

  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--private-repo-root") {
      privateRepoRoot = argv[++i] ?? "";
      continue;
    }
    if (argv[i] === "--private-artifact") {
      privateArtifact = argv[++i] ?? "";
      continue;
    }
    throw new Error(`VNEXT_GATE18_GEMMA3_CONSTELLATION_CONFLICT_FORENSIC_UNKNOWN_ARG:${argv[i]}`);
  }

  if (!privateRepoRoot.trim()) {
    throw new Error("VNEXT_GATE18_GEMMA3_CONSTELLATION_CONFLICT_FORENSIC_PRIVATE_REPO_ROOT_REQUIRED");
  }
  if (!privateArtifact.trim()) {
    throw new Error("VNEXT_GATE18_GEMMA3_CONSTELLATION_CONFLICT_FORENSIC_PRIVATE_ARTIFACT_REQUIRED");
  }

  return { privateRepoRoot, privateArtifact };
}

function findConstellation(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) => company.display_name === EXPECTED_COMPANY,
  );
  if (matches.length !== 1) {
    throw new Error("VNEXT_GATE18_GEMMA3_CONSTELLATION_CONFLICT_FORENSIC_COMPANY_NOT_UNIQUE");
  }
  return matches[0] as Gate18PilotCompany;
}

function artifactReader(privateRepoRoot: string): (pin: Gate18ArtifactPin) => Uint8Array {
  return (pin) => {
    const normalizedRoot = resolve(privateRepoRoot).replaceAll("\\", "/");
    const absolute = resolve(privateRepoRoot, pin.path);
    const normalizedAbsolute = absolute.replaceAll("\\", "/");
    const prefix = normalizedRoot.endsWith("/") ? normalizedRoot : `${normalizedRoot}/`;

    if (!normalizedAbsolute.startsWith(prefix)) {
      throw new Error("VNEXT_GATE18_GEMMA3_CONSTELLATION_CONFLICT_FORENSIC_PRIVATE_PATH_ESCAPE");
    }

    execFileSync(
      "git",
      ["-C", privateRepoRoot, "cat-file", "-e", `${pin.commit_sha}^{commit}`],
      { stdio: ["ignore", "ignore", "pipe"] },
    );

    return execFileSync(
      "git",
      ["-C", privateRepoRoot, "show", `${pin.commit_sha}:${pin.path.replaceAll("\\", "/")}`],
      {
        encoding: "buffer",
        maxBuffer: 20 * 1024 * 1024,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
  };
}

function main(): void {
  const { privateRepoRoot, privateArtifact } = parseArgs(process.argv.slice(2));

  const raw = JSON.parse(readFileSync(resolve(privateArtifact), "utf8")) as {
    status?: string;
    invocation?: {
      matrixCellId?: string;
      attemptId?: string;
      company?: string;
      promptSha256?: string;
      fullPacketSha256?: string;
    };
    execution?: {
      doneReason?: string | null;
      evalCount?: number | null;
      schemaValid?: boolean;
      schemaError?: string | null;
      semanticValid?: boolean;
      semanticError?: string | null;
    };
    response?: { parsedJson?: unknown };
    validationV11?: {
      normalization?: { normalizedPathCount?: number };
      substantiveValidation?: { status?: string; error?: string | null };
    };
  };

  if (
    raw.status !== "FAIL" ||
    raw.invocation?.matrixCellId !== EXPECTED_MATRIX_CELL_ID ||
    raw.invocation?.attemptId !== EXPECTED_ATTEMPT_ID ||
    raw.invocation?.company !== EXPECTED_COMPANY ||
    raw.invocation?.promptSha256 !== EXPECTED_PROMPT_SHA256 ||
    raw.invocation?.fullPacketSha256 !== EXPECTED_PACKET_SHA256 ||
    raw.execution?.doneReason !== "stop" ||
    raw.execution?.schemaValid !== true ||
    raw.execution?.schemaError !== null ||
    raw.execution?.semanticValid !== false ||
    raw.execution?.semanticError !== EXPECTED_FIRST_ERROR ||
    raw.validationV11?.normalization?.normalizedPathCount !== 13 ||
    raw.validationV11?.substantiveValidation?.status !== "FAIL" ||
    raw.validationV11?.substantiveValidation?.error !== EXPECTED_FIRST_ERROR
  ) {
    throw new Error("VNEXT_GATE18_GEMMA3_CONSTELLATION_CONFLICT_FORENSIC_RUN_IDENTITY_MISMATCH");
  }

  const parsed = gate18PhaseBV10OutputSchema.safeParse(raw.response?.parsedJson);
  if (!parsed.success) {
    throw new Error("VNEXT_GATE18_GEMMA3_CONSTELLATION_CONFLICT_FORENSIC_SCHEMA_REPARSE_FAILED");
  }

  const verified = buildVerifiedGate18V10MoatPacket(
    findConstellation(),
    artifactReader(privateRepoRoot),
  );
  if (verified.packetSha256 !== EXPECTED_PACKET_SHA256) {
    throw new Error("VNEXT_GATE18_GEMMA3_CONSTELLATION_CONFLICT_FORENSIC_PACKET_IDENTITY_MISMATCH");
  }

  const knownConflictIds = new Set(
    verified.packet.conflicts.map((conflict) => conflict.conflict_id),
  );

  const v11 = normalizeGate18V11Presentation(parsed.data);
  const presentation = inspectGate18V11Presentation(v11.normalized);
  if (!presentation.compliant || v11.normalizedPaths.length !== 13) {
    throw new Error("VNEXT_GATE18_GEMMA3_CONSTELLATION_CONFLICT_FORENSIC_V11_NORMALIZATION_MISMATCH");
  }

  const diagnostic = structuredClone(v11.normalized);

  const firstRuleFindingIndexes: number[] = [];
  diagnostic.priority_findings.forEach((finding, index) => {
    if (finding.counterevidence_ids.length === 0 && finding.counterevidence_link !== null) {
      finding.counterevidence_link = null;
      firstRuleFindingIndexes.push(index + 1);
    }
  });

  let firstRuleValidationError: string | null = null;
  try {
    assertGate18PhaseBV10Semantics(verified.packet, diagnostic);
  } catch (error) {
    firstRuleValidationError = error instanceof Error ? error.message : String(error);
  }

  if (firstRuleValidationError !== EXPECTED_SECOND_ERROR) {
    throw new Error(
      `VNEXT_GATE18_GEMMA3_CONSTELLATION_CONFLICT_FORENSIC_EXPECTED_SECOND_ERROR_MISMATCH:${firstRuleValidationError}`,
    );
  }

  const conflictReferences = {
    knownConflictIds: [...knownConflictIds],
    priorityFindings: diagnostic.priority_findings.map((finding, index) => ({
      findingIndex: index + 1,
      conflictIds: [...finding.conflict_ids],
      unknownConflictIds: finding.conflict_ids.filter((id) => !knownConflictIds.has(id)),
    })),
    materialConflicts: diagnostic.material_conflicts.map((item, index) => ({
      itemIndex: index + 1,
      conflictId: item.conflict_id,
      unknown: !knownConflictIds.has(item.conflict_id),
    })),
    weakLinkCandidates: diagnostic.weak_link_candidates.map((item, index) => ({
      itemIndex: index + 1,
      conflictIds: [...item.conflict_ids],
      unknownConflictIds: item.conflict_ids.filter((id) => !knownConflictIds.has(id)),
    })),
    unresolvedPoints: diagnostic.unresolved_points.map((item, index) => ({
      itemIndex: index + 1,
      conflictIds: [...item.conflict_ids],
      unknownConflictIds: item.conflict_ids.filter((id) => !knownConflictIds.has(id)),
    })),
  };

  const removed = {
    priorityFindingConflictRefs: [] as Array<{ findingIndex: number; conflictId: string }>,
    materialConflictEntries: [] as Array<{ itemIndex: number; conflictId: string }>,
    weakLinkConflictRefs: [] as Array<{ itemIndex: number; conflictId: string }>,
    unresolvedPointConflictRefs: [] as Array<{ itemIndex: number; conflictId: string }>,
  };

  diagnostic.priority_findings.forEach((finding, index) => {
    finding.conflict_ids = finding.conflict_ids.filter((id) => {
      if (knownConflictIds.has(id)) return true;
      removed.priorityFindingConflictRefs.push({ findingIndex: index + 1, conflictId: id });
      return false;
    });
  });

  diagnostic.material_conflicts = diagnostic.material_conflicts.filter((item, index) => {
    if (knownConflictIds.has(item.conflict_id)) return true;
    removed.materialConflictEntries.push({ itemIndex: index + 1, conflictId: item.conflict_id });
    return false;
  });

  diagnostic.weak_link_candidates.forEach((item, index) => {
    item.conflict_ids = item.conflict_ids.filter((id) => {
      if (knownConflictIds.has(id)) return true;
      removed.weakLinkConflictRefs.push({ itemIndex: index + 1, conflictId: id });
      return false;
    });
  });

  diagnostic.unresolved_points.forEach((item, index) => {
    item.conflict_ids = item.conflict_ids.filter((id) => {
      if (knownConflictIds.has(id)) return true;
      removed.unresolvedPointConflictRefs.push({ itemIndex: index + 1, conflictId: id });
      return false;
    });
  });

  let downstreamValidationPass = false;
  let downstreamValidationError: string | null = null;
  try {
    assertGate18PhaseBV10Semantics(verified.packet, diagnostic);
    downstreamValidationPass = true;
  } catch (error) {
    downstreamValidationError = error instanceof Error ? error.message : String(error);
  }

  const removedUnknownConflictRefCount =
    removed.priorityFindingConflictRefs.length +
    removed.materialConflictEntries.length +
    removed.weakLinkConflictRefs.length +
    removed.unresolvedPointConflictRefs.length;

  console.log(JSON.stringify({
    format:"OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GEMMA3_UNKNOWN_CONFLICT_REF_FORENSIC_V0.1",
    gate:18,
    phase:"C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage:"C4_DETERMINISTIC_SEMANTIC_FORENSICS",
    status:"FORENSIC_COMPLETE",
    mode:"CUMULATIVE_IN_MEMORY_DIAGNOSTIC_ONLY_NO_ARTIFACT_MUTATION_NO_INFERENCE",
    sourceRun:{
      matrixCellId:EXPECTED_MATRIX_CELL_ID,
      attemptId:EXPECTED_ATTEMPT_ID,
      doneReason:raw.execution?.doneReason ?? null,
      evalCount:raw.execution?.evalCount ?? null,
      schemaValid:raw.execution?.schemaValid ?? null,
      semanticValid:raw.execution?.semanticValid ?? null,
      semanticError:raw.execution?.semanticError ?? null,
      v11NormalizedPathCount:v11.normalizedPaths.length
    },
    firstDiagnosticRule:{
      counterevidenceLinkNormalizedFindingIndexes:firstRuleFindingIndexes,
      downstreamErrorAfterFirstRule:firstRuleValidationError
    },
    conflictReferenceAudit:{
      ...conflictReferences,
      rawNarrativeTextIncludedInConsole:false
    },
    diagnosticNormalization:{
      rule:"Remove only conflict references absent from the canonical packet; remove a material_conflicts entry only when its conflict_id is unknown.",
      removed,
      removedUnknownConflictRefCount,
      sourceArtifactMutated:false,
      rawOutputMutated:false,
      generatedContentPublished:false,
      autoRepairExecuted:false
    },
    downstreamValidation:{
      fullFrozenV10SemanticValidatorExecutedOnInMemoryCopy:true,
      pass:downstreamValidationPass,
      error:downstreamValidationError
    },
    interpretationBoundary:{
      retroactivePassAllowed:false,
      originalRunStatusChanged:false,
      inferenceExecuted:false,
      retryAuthorized:false,
      modelCapabilityFailureConcluded:false,
      allKnownDeterministicDefectsExhausted:downstreamValidationPass,
      purpose:"Determine whether another deterministic semantic defect remains after cumulative diagnostic normalization of counterevidence-link structure and unknown conflict references."
    },
    safety:{
      externalNetworkAccessRequested:false,
      ollamaApiCalled:false,
      modelInferenceExecuted:false,
      productionMutation:false,
      publicationAuthority:false
    }
  },null,2));
}

main();
