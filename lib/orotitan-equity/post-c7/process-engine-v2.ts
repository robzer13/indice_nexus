import { createHash } from "node:crypto";

export type BlockCode =
  | "BUSINESS_MODEL"
  | "ECONOMIC_QUALITY"
  | "INDUSTRY_STRUCTURE"
  | "TECHNOLOGY"
  | "CYCLICALITY"
  | "MOAT"
  | "RUNWAY"
  | "RETURN_QUALITY"
  | "FCF_FORENSIC"
  | "CAPITAL_ALLOCATION"
  | "MANAGEMENT_GOVERNANCE"
  | "OUTSIDE_VIEW"
  | "RISK_RESILIENCE"
  | "RED_TEAM"
  | "VALUATION"
  | "CROSS_BLOCK_RECONCILIATION";

export type BlockStatus =
  | "READY"
  | "IN_PROGRESS"
  | "CHECKPOINTED"
  | "COMPLETE"
  | "BLOCKED"
  | "NOT_ASSESSABLE"
  | "REOPENED"
  | "STALE";

export type RegistryLifecycle =
  | "NOT_STARTED"
  | "IN_PROGRESS"
  | "PAUSED"
  | "BLOCKED"
  | "COMPLETE";

export type ExecutionPhase =
  | "RESEARCH"
  | "FUNDAMENTALS"
  | "VALUATION"
  | "CERTIFICATION"
  | "INTEGRATION";

export type RefreshClass =
  | "PRICE_ONLY_DELTA"
  | "ROUTINE_FUNDAMENTAL_DELTA"
  | "FULL_REFRESH_REQUIRED";

export type AnalyticalBlockState = {
  block: BlockCode;
  status: BlockStatus;
  upstreamBlockRefs: BlockCode[];
  criticalUnresolvedGap: boolean;
  requiredOverlayMissing: boolean;
  materialRevalidationStatus: "NOT_REQUIRED" | "PENDING" | "PASS" | "FAIL";
};

export type DependencyGraphResult =
  | {
      ok: true;
      reverse: Map<BlockCode, Set<BlockCode>>;
      topologicalOrder: BlockCode[];
    }
  | {
      ok: false;
      errors: string[];
    };

export type RefreshPlan =
  | {
      ok: true;
      refreshClass: RefreshClass;
      researchMode: "MINIMAL_REVALIDATION" | "TARGETED_DELTA" | "FULL";
      reopenBlocks: BlockCode[];
      preservedBlocks: BlockCode[];
      allowFundamentalScoreChange: boolean;
      requiresValuation: boolean;
      requiresCertification: boolean;
      requiresIntegration: boolean;
    }
  | {
      ok: false;
      errors: string[];
    };

export type SaveEligibilityInput = {
  phase: ExecutionPhase;
  registryLifecycle: RegistryLifecycle;
  selfAuditPassed: boolean;
  requiredArtifactsPresent: boolean;
  persistenceVerified: boolean;
  registryReconciled: boolean;
  criticalBlockers: string[];
  phaseGate:
    | "NOT_EVALUATED"
    | "NO"
    | "YES";
};

export type SaveDisposition =
  | {
      action: "CHECKPOINT";
      reason: string;
      publishAuthorized: false;
    }
  | {
      action: "FINALIZE";
      reason: string;
      publishAuthorized: false;
    }
  | {
      action: "BLOCK";
      reason: string;
      publishAuthorized: false;
    }
  | {
      action: "NOOP";
      reason: string;
      publishAuthorized: false;
    };

export type FingerprintInput = {
  runId: string;
  block: BlockCode;
  inputVersion: string;
  evidenceSetHash: string;
  methodVersion: string;
  outputSchemaVersion: string;
};

export type RetryDecision =
  | { allowed: true; reason: string }
  | { allowed: false; reason: string };

const FUNDAMENTAL_BLOCKS: BlockCode[] = [
  "BUSINESS_MODEL",
  "ECONOMIC_QUALITY",
  "INDUSTRY_STRUCTURE",
  "TECHNOLOGY",
  "CYCLICALITY",
  "MOAT",
  "RUNWAY",
  "RETURN_QUALITY",
  "FCF_FORENSIC",
  "CAPITAL_ALLOCATION",
  "MANAGEMENT_GOVERNANCE",
  "OUTSIDE_VIEW",
  "RISK_RESILIENCE",
  "RED_TEAM",
];

function uniqueSorted<T extends string>(values: Iterable<T>): T[] {
  return [...new Set(values)].sort() as T[];
}

export function buildDependencyGraph(
  blocks: AnalyticalBlockState[],
): DependencyGraphResult {
  const errors: string[] = [];
  const byCode = new Map<BlockCode, AnalyticalBlockState>();
  for (const block of blocks) {
    if (byCode.has(block.block)) errors.push(`duplicate block ${block.block}`);
    byCode.set(block.block, block);
  }

  const reverse = new Map<BlockCode, Set<BlockCode>>();
  for (const code of byCode.keys()) reverse.set(code, new Set());

  for (const block of blocks) {
    for (const upstream of block.upstreamBlockRefs) {
      if (upstream === block.block) {
        errors.push(`block ${block.block} cannot depend on itself`);
        continue;
      }
      if (!byCode.has(upstream)) {
        errors.push(`block ${block.block} references unknown upstream block ${upstream}`);
        continue;
      }
      reverse.get(upstream)?.add(block.block);
    }
  }

  if (errors.length > 0) return { ok: false, errors };

  const indegree = new Map<BlockCode, number>();
  for (const block of blocks) indegree.set(block.block, block.upstreamBlockRefs.length);

  const queue = uniqueSorted(
    [...indegree.entries()]
      .filter(([, degree]) => degree === 0)
      .map(([code]) => code),
  );

  const order: BlockCode[] = [];
  while (queue.length > 0) {
    const current = queue.shift() as BlockCode;
    order.push(current);
    for (const downstream of reverse.get(current) ?? []) {
      const next = (indegree.get(downstream) ?? 0) - 1;
      indegree.set(downstream, next);
      if (next === 0) {
        queue.push(downstream);
        queue.sort();
      }
    }
  }

  if (order.length !== blocks.length) {
    return {
      ok: false,
      errors: ["dependency graph contains a cycle or unresolved dependency"],
    };
  }

  return { ok: true, reverse, topologicalOrder: order };
}

export function computeDownstreamClosure(
  changedBlocks: BlockCode[],
  blocks: AnalyticalBlockState[],
): { ok: true; blocks: BlockCode[] } | { ok: false; errors: string[] } {
  const graph = buildDependencyGraph(blocks);
  if (!graph.ok) return graph;

  const known = new Set(blocks.map((block) => block.block));
  const errors = changedBlocks
    .filter((block) => !known.has(block))
    .map((block) => `changed block ${block} is absent from current analytical package`);
  if (errors.length > 0) return { ok: false, errors };

  const visited = new Set<BlockCode>();
  const queue = [...changedBlocks];

  while (queue.length > 0) {
    const current = queue.shift() as BlockCode;
    if (visited.has(current)) continue;
    visited.add(current);
    for (const downstream of graph.reverse.get(current) ?? []) {
      if (!visited.has(downstream)) queue.push(downstream);
    }
  }

  return { ok: true, blocks: uniqueSorted(visited) };
}

export function blockCompletionEligibility(
  block: AnalyticalBlockState,
  allBlocks: AnalyticalBlockState[],
): { eligible: true } | { eligible: false; reasons: string[] } {
  const reasons: string[] = [];

  if (block.criticalUnresolvedGap) reasons.push("critical unresolved gap");
  if (block.requiredOverlayMissing) reasons.push("required sector overlay missing");
  if (block.materialRevalidationStatus === "PENDING") reasons.push("material change revalidation pending");
  if (block.materialRevalidationStatus === "FAIL") reasons.push("material change revalidation failed");

  const byCode = new Map(allBlocks.map((item) => [item.block, item]));
  for (const upstream of block.upstreamBlockRefs) {
    const parent = byCode.get(upstream);
    if (!parent) {
      reasons.push(`upstream block ${upstream} missing`);
      continue;
    }
    if (parent.status !== "COMPLETE") {
      reasons.push(`upstream block ${upstream} is not COMPLETE`);
    }
  }

  return reasons.length === 0 ? { eligible: true } : { eligible: false, reasons };
}

export function planRefresh(
  refreshClass: RefreshClass,
  changedBlocks: BlockCode[],
  blocks: AnalyticalBlockState[],
): RefreshPlan {
  const currentCodes = new Set(blocks.map((block) => block.block));

  if (refreshClass === "PRICE_ONLY_DELTA") {
    const reopen = ["VALUATION", "CROSS_BLOCK_RECONCILIATION"].filter(
      (block): block is BlockCode => currentCodes.has(block as BlockCode),
    );
    return {
      ok: true,
      refreshClass,
      researchMode: "MINIMAL_REVALIDATION",
      reopenBlocks: reopen,
      preservedBlocks: uniqueSorted(
        blocks
          .map((block) => block.block)
          .filter((block) => !reopen.includes(block)),
      ),
      allowFundamentalScoreChange: false,
      requiresValuation: currentCodes.has("VALUATION"),
      requiresCertification: true,
      requiresIntegration: true,
    };
  }

  if (refreshClass === "FULL_REFRESH_REQUIRED") {
    return {
      ok: true,
      refreshClass,
      researchMode: "FULL",
      reopenBlocks: uniqueSorted(blocks.map((block) => block.block)),
      preservedBlocks: [],
      allowFundamentalScoreChange: true,
      requiresValuation: currentCodes.has("VALUATION"),
      requiresCertification: true,
      requiresIntegration: true,
    };
  }

  if (changedBlocks.length === 0) {
    return {
      ok: false,
      errors: ["ROUTINE_FUNDAMENTAL_DELTA requires at least one materially changed block"],
    };
  }

  const nonFundamental = changedBlocks.filter(
    (block) => !FUNDAMENTAL_BLOCKS.includes(block),
  );
  if (nonFundamental.length > 0) {
    return {
      ok: false,
      errors: nonFundamental.map(
        (block) => `routine fundamental delta cannot originate from non-fundamental block ${block}`,
      ),
    };
  }

  const closure = computeDownstreamClosure(changedBlocks, blocks);
  if (!closure.ok) return closure;

  const reopen = new Set<BlockCode>(closure.blocks);
  if (currentCodes.has("VALUATION")) reopen.add("VALUATION");
  if (currentCodes.has("CROSS_BLOCK_RECONCILIATION")) reopen.add("CROSS_BLOCK_RECONCILIATION");

  const reopenBlocks = uniqueSorted(reopen);
  return {
    ok: true,
    refreshClass,
    researchMode: "TARGETED_DELTA",
    reopenBlocks,
    preservedBlocks: uniqueSorted(
      blocks
        .map((block) => block.block)
        .filter((block) => !reopen.has(block)),
    ),
    allowFundamentalScoreChange: true,
    requiresValuation: currentCodes.has("VALUATION"),
    requiresCertification: true,
    requiresIntegration: true,
  };
}

export function deriveSaveDisposition(
  input: SaveEligibilityInput,
): SaveDisposition {
  if (input.registryLifecycle === "COMPLETE") {
    return {
      action: "NOOP",
      reason: "stage already COMPLETE; explicit reopen required before new analytical save",
      publishAuthorized: false,
    };
  }

  if (input.criticalBlockers.length > 0 || input.phaseGate === "NO") {
    return {
      action: "BLOCK",
      reason:
        input.criticalBlockers.length > 0
          ? `critical blockers: ${input.criticalBlockers.join("; ")}`
          : "phase gate is NO",
      publishAuthorized: false,
    };
  }

  const durableReady =
    input.selfAuditPassed &&
    input.requiredArtifactsPresent &&
    input.persistenceVerified &&
    input.registryReconciled;

  if (!durableReady) {
    return {
      action: "CHECKPOINT",
      reason: "durable completion prerequisites are not all satisfied",
      publishAuthorized: false,
    };
  }

  if (input.phase === "RESEARCH" && input.phaseGate === "YES") {
    return {
      action: "FINALIZE",
      reason: "Research completion gate satisfied",
      publishAuthorized: false,
    };
  }

  if (input.phase === "CERTIFICATION" && input.phaseGate === "YES") {
    return {
      action: "FINALIZE",
      reason: "Deep Dive Certification completion gate satisfied",
      publishAuthorized: false,
    };
  }

  if (input.phase === "INTEGRATION" && input.phaseGate === "YES") {
    return {
      action: "FINALIZE",
      reason: "Integration READY_TO_PUBLISH gate satisfied; publication remains separate",
      publishAuthorized: false,
    };
  }

  return {
    action: "CHECKPOINT",
    reason: `${input.phase} remains an intermediate phase`,
    publishAuthorized: false,
  };
}

export function computeExecutionFingerprint(input: FingerprintInput): string {
  const canonical = [
    input.runId,
    input.block,
    input.inputVersion,
    input.evidenceSetHash,
    input.methodVersion,
    input.outputSchemaVersion,
  ].join("|");
  return createHash("sha256").update(canonical, "utf8").digest("hex");
}

export function decideRetry(input: {
  priorFingerprint: string | null;
  nextFingerprint: string;
  newEvidence: boolean;
  resolvedConflict: boolean;
  methodChanged: boolean;
  deterministicBugFixed: boolean;
  criticalUserInputAdded: boolean;
  forensicJustification: boolean;
}): RetryDecision {
  if (input.priorFingerprint === null) {
    return { allowed: true, reason: "first execution" };
  }

  if (input.priorFingerprint !== input.nextFingerprint) {
    return { allowed: true, reason: "execution fingerprint changed" };
  }

  const justified = [
    ["new evidence", input.newEvidence],
    ["resolved conflict", input.resolvedConflict],
    ["method change", input.methodChanged],
    ["deterministic bug fix", input.deterministicBugFixed],
    ["critical user input", input.criticalUserInputAdded],
    ["forensic justification", input.forensicJustification],
  ] as const;

  const reason = justified.find(([, active]) => active)?.[0];
  if (reason) return { allowed: true, reason };

  return {
    allowed: false,
    reason: "same execution fingerprint with no new information or justified change",
  };
}

function unresolvedBlockPriority(status: BlockStatus): number {
  switch (status) {
    case "BLOCKED": return 0;
    case "REOPENED": return 1;
    case "STALE": return 2;
    case "IN_PROGRESS": return 3;
    case "READY": return 4;
    case "CHECKPOINTED": return 5;
    case "NOT_ASSESSABLE": return 6;
    case "COMPLETE": return 7;
  }
}

export function deriveNextBlock(
  blocks: AnalyticalBlockState[],
): { block: BlockCode | null; reason: string } {
  const graph = buildDependencyGraph(blocks);
  if (!graph.ok) return { block: null, reason: graph.errors.join("; ") };

  const byCode = new Map(blocks.map((block) => [block.block, block]));
  const candidates = blocks
    .filter((block) => !["COMPLETE", "NOT_ASSESSABLE"].includes(block.status))
    .filter((block) =>
      block.upstreamBlockRefs.every((upstream) => byCode.get(upstream)?.status === "COMPLETE"),
    )
    .sort((left, right) => {
      const statusDelta = unresolvedBlockPriority(left.status) - unresolvedBlockPriority(right.status);
      if (statusDelta !== 0) return statusDelta;
      return graph.topologicalOrder.indexOf(left.block) - graph.topologicalOrder.indexOf(right.block);
    });

  if (candidates.length === 0) {
    const unfinished = blocks.filter(
      (block) => !["COMPLETE", "NOT_ASSESSABLE"].includes(block.status),
    );
    if (unfinished.length === 0) return { block: null, reason: "all blocks resolved" };
    return {
      block: null,
      reason: "no executable block: unresolved upstream dependency or blocker requires intervention",
    };
  }

  return {
    block: candidates[0].block,
    reason: `highest-priority executable block with satisfied upstream dependencies (${candidates[0].status})`,
  };
}
