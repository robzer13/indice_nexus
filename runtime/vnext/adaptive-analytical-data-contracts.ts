import type { ValidationIssue } from "./analytical-data-contracts";

type Context = {
  run_id: string;
  stage_code: string;
  stage_revision: number;
  data_cutoff: string;
  contract_set_sha256: string;
};

type RunContext = {
  run_id: string;
  data_cutoff: string;
  contract_set_sha256: string;
  stage_code: string;
  stage_revision: number;
};

type TraceRefs = {
  evidence_ids?: string[];
  contradicting_evidence_ids?: string[];
  calculation_ids?: string[];
};

export type AdaptiveAnalyticalSupport = {
  run: RunContext;
  evidence_ids: string[];
  calculation_ids: string[];
  company_dna?: {
    context: Context;
    primary_sector: string;
    evidence_ids: string[];
    contradicting_evidence_ids: string[];
  };
  overlay_selection?: {
    context: Context;
    primary_sector: string;
    selected_overlays: Array<{
      overlay_id: string;
      evidence_ids: string[];
    }>;
  };
  cycles?: Array<
    TraceRefs & {
      context: Context;
      cycle_analysis_id: string;
    }
  >;
  technology_map?: TraceRefs & {
    context: Context;
    technology_map_id: string;
  };
  industry_structure?: TraceRefs & {
    context: Context;
    industry_structure_id: string;
  };
  causal_graph?: {
    context: Context;
    graph_id: string;
    nodes: Array<{
      node_id: string;
      evidence_ids: string[];
      contradicting_evidence_ids: string[];
    }>;
    edges: Array<{
      from_node: string;
      to_node: string;
      supporting_evidence_ids: string[];
      counterevidence_ids: string[];
    }>;
  };
  serial_acquirer?: {
    context: Context;
    evidence_ids: string[];
    calculation_ids: string[];
    capital_sources: Array<{ evidence_ids: string[] }>;
    capital_uses: Array<{ evidence_ids: string[]; calculation_ids: string[] }>;
    acquisition_cohorts: Array<{
      cohort_id: string;
      evidence_ids: string[];
      calculation_ids: string[];
    }>;
  };
};

function push(
  issues: ValidationIssue[],
  code: string,
  path: string,
  message: string,
) {
  issues.push({ code, path, message });
}

function checkContext(
  ctx: Context,
  expected: RunContext,
  path: string,
  issues: ValidationIssue[],
) {
  if (ctx.run_id !== expected.run_id) {
    push(issues, "RUN_ID_MISMATCH", `${path}.run_id`, "run_id mismatch");
  }
  if (ctx.stage_code !== expected.stage_code) {
    push(issues, "STAGE_CODE_MISMATCH", `${path}.stage_code`, "stage_code mismatch");
  }
  if (ctx.stage_revision !== expected.stage_revision) {
    push(
      issues,
      "STAGE_REVISION_MISMATCH",
      `${path}.stage_revision`,
      "stage_revision mismatch",
    );
  }
  if (ctx.data_cutoff !== expected.data_cutoff) {
    push(
      issues,
      "DATA_CUTOFF_MISMATCH",
      `${path}.data_cutoff`,
      "data_cutoff mismatch",
    );
  }
  if (ctx.contract_set_sha256 !== expected.contract_set_sha256) {
    push(
      issues,
      "CONTRACT_SET_MISMATCH",
      `${path}.contract_set_sha256`,
      "contract set mismatch",
    );
  }
}

function requireRefs(
  refs: string[] | undefined,
  known: Set<string>,
  path: string,
  kind: "EVIDENCE" | "CALCULATION",
  issues: ValidationIssue[],
) {
  (refs ?? []).forEach((id, index) => {
    if (!known.has(id)) {
      push(
        issues,
        `UNKNOWN_${kind}_ID`,
        `${path}[${index}]`,
        `Unknown ${kind.toLowerCase()} ID: ${id}`,
      );
    }
  });
}

function checkTraceRefs(
  item: TraceRefs,
  evidence: Set<string>,
  calculations: Set<string>,
  path: string,
  issues: ValidationIssue[],
) {
  requireRefs(item.evidence_ids, evidence, `${path}.evidence_ids`, "EVIDENCE", issues);
  requireRefs(
    item.contradicting_evidence_ids,
    evidence,
    `${path}.contradicting_evidence_ids`,
    "EVIDENCE",
    issues,
  );
  requireRefs(
    item.calculation_ids,
    calculations,
    `${path}.calculation_ids`,
    "CALCULATION",
    issues,
  );
}

export function validateAdaptiveAnalyticalSupport(
  value: AdaptiveAnalyticalSupport,
): { ok: boolean; issues: ValidationIssue[] } {
  const issues: ValidationIssue[] = [];
  const evidence = new Set(value.evidence_ids);
  const calculations = new Set(value.calculation_ids);

  if (value.company_dna) {
    checkContext(value.company_dna.context, value.run, "company_dna.context", issues);
    checkTraceRefs(value.company_dna, evidence, calculations, "company_dna", issues);
  }

  if (value.overlay_selection) {
    checkContext(
      value.overlay_selection.context,
      value.run,
      "overlay_selection.context",
      issues,
    );
    const overlayIds = new Set<string>();
    value.overlay_selection.selected_overlays.forEach((overlay, index) => {
      if (overlayIds.has(overlay.overlay_id)) {
        push(
          issues,
          "DUPLICATE_OVERLAY_ID",
          `overlay_selection.selected_overlays[${index}].overlay_id`,
          `Duplicate overlay ID: ${overlay.overlay_id}`,
        );
      }
      overlayIds.add(overlay.overlay_id);
      requireRefs(
        overlay.evidence_ids,
        evidence,
        `overlay_selection.selected_overlays[${index}].evidence_ids`,
        "EVIDENCE",
        issues,
      );
    });

    if (
      value.company_dna &&
      !["UNKNOWN", "NOT_APPLICABLE", "NOT_ASSESSABLE", "MISSING", "NOT_AVAILABLE"].includes(
        value.company_dna.primary_sector,
      ) &&
      value.company_dna.primary_sector !== value.overlay_selection.primary_sector
    ) {
      push(
        issues,
        "OVERLAY_PRIMARY_SECTOR_MISMATCH",
        "overlay_selection.primary_sector",
        "Overlay selection silently contradicts Company Economic DNA primary sector.",
      );
    }
  }

  const cycleIds = new Set<string>();
  value.cycles?.forEach((cycle, index) => {
    checkContext(cycle.context, value.run, `cycles[${index}].context`, issues);
    checkTraceRefs(cycle, evidence, calculations, `cycles[${index}]`, issues);
    if (cycleIds.has(cycle.cycle_analysis_id)) {
      push(
        issues,
        "DUPLICATE_CYCLE_ANALYSIS_ID",
        `cycles[${index}].cycle_analysis_id`,
        `Duplicate cycle analysis ID: ${cycle.cycle_analysis_id}`,
      );
    }
    cycleIds.add(cycle.cycle_analysis_id);
  });

  if (value.technology_map) {
    checkContext(
      value.technology_map.context,
      value.run,
      "technology_map.context",
      issues,
    );
    checkTraceRefs(value.technology_map, evidence, calculations, "technology_map", issues);
  }

  if (value.industry_structure) {
    checkContext(
      value.industry_structure.context,
      value.run,
      "industry_structure.context",
      issues,
    );
    checkTraceRefs(
      value.industry_structure,
      evidence,
      calculations,
      "industry_structure",
      issues,
    );
  }

  if (value.causal_graph) {
    checkContext(value.causal_graph.context, value.run, "causal_graph.context", issues);
    const nodes = new Set<string>();
    value.causal_graph.nodes.forEach((node, index) => {
      if (nodes.has(node.node_id)) {
        push(
          issues,
          "DUPLICATE_CAUSAL_NODE_ID",
          `causal_graph.nodes[${index}].node_id`,
          `Duplicate causal node ID: ${node.node_id}`,
        );
      }
      nodes.add(node.node_id);
      requireRefs(
        [...node.evidence_ids, ...node.contradicting_evidence_ids],
        evidence,
        `causal_graph.nodes[${index}].evidence_ids`,
        "EVIDENCE",
        issues,
      );
    });

    value.causal_graph.edges.forEach((edge, index) => {
      if (!nodes.has(edge.from_node)) {
        push(
          issues,
          "UNKNOWN_CAUSAL_FROM_NODE",
          `causal_graph.edges[${index}].from_node`,
          `Unknown causal node: ${edge.from_node}`,
        );
      }
      if (!nodes.has(edge.to_node)) {
        push(
          issues,
          "UNKNOWN_CAUSAL_TO_NODE",
          `causal_graph.edges[${index}].to_node`,
          `Unknown causal node: ${edge.to_node}`,
        );
      }
      requireRefs(
        [...edge.supporting_evidence_ids, ...edge.counterevidence_ids],
        evidence,
        `causal_graph.edges[${index}].evidence_ids`,
        "EVIDENCE",
        issues,
      );
    });
  }

  if (value.serial_acquirer) {
    checkContext(
      value.serial_acquirer.context,
      value.run,
      "serial_acquirer.context",
      issues,
    );
    checkTraceRefs(
      value.serial_acquirer,
      evidence,
      calculations,
      "serial_acquirer",
      issues,
    );

    value.serial_acquirer.capital_sources.forEach((row, index) => {
      requireRefs(
        row.evidence_ids,
        evidence,
        `serial_acquirer.capital_sources[${index}].evidence_ids`,
        "EVIDENCE",
        issues,
      );
    });

    value.serial_acquirer.capital_uses.forEach((row, index) => {
      requireRefs(
        row.evidence_ids,
        evidence,
        `serial_acquirer.capital_uses[${index}].evidence_ids`,
        "EVIDENCE",
        issues,
      );
      requireRefs(
        row.calculation_ids,
        calculations,
        `serial_acquirer.capital_uses[${index}].calculation_ids`,
        "CALCULATION",
        issues,
      );
    });

    const cohorts = new Set<string>();
    value.serial_acquirer.acquisition_cohorts.forEach((row, index) => {
      if (cohorts.has(row.cohort_id)) {
        push(
          issues,
          "DUPLICATE_ACQUISITION_COHORT_ID",
          `serial_acquirer.acquisition_cohorts[${index}].cohort_id`,
          `Duplicate cohort ID: ${row.cohort_id}`,
        );
      }
      cohorts.add(row.cohort_id);
      requireRefs(
        row.evidence_ids,
        evidence,
        `serial_acquirer.acquisition_cohorts[${index}].evidence_ids`,
        "EVIDENCE",
        issues,
      );
      requireRefs(
        row.calculation_ids,
        calculations,
        `serial_acquirer.acquisition_cohorts[${index}].calculation_ids`,
        "CALCULATION",
        issues,
      );
    });
  }

  return { ok: issues.length === 0, issues };
}
