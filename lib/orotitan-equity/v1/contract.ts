import { z } from "zod";
import {
  businessResearchStatusSchema, investmentConclusionStatusSchema, mosStatusSchema,
  scorePermissionSchema, valuationReliabilitySchema,
} from "./certification";
import { scoreExpectedReturnDelta } from "./expected-return";
import { computeOqs, evidenceStateSchema, qualityDimensionsSchema } from "./quality";
import { computeInvestmentScore, computeOvs, computeReturnComponent } from "./scoring";
import { canonicalScoreSchema } from "./semantic-states";
import { computeOroTitanStatus, eliteGatesSchema } from "./terminal-gate";

const deterministicSchema = z.object({
  oqsRaw: canonicalScoreSchema.optional(), weakLinkCap: canonicalScoreSchema.optional(), oqs: canonicalScoreSchema.optional(),
  fiveYearReturnScore: z.number().finite().min(0).max(100).optional(), tenYearReturnScore: z.number().finite().min(0).max(100).optional(),
  returnComponent: canonicalScoreSchema.optional(), ovs: canonicalScoreSchema.optional(),
  investmentRaw: canonicalScoreSchema.optional(), investmentScore: canonicalScoreSchema.optional(),
  orotitanStatus: z.enum(["YES", "NO"]).optional(),
}).strict();

const contractShape = z.object({
  dimensions: qualityDimensionsSchema,
  evidence: z.object({ moat: evidenceStateSchema, runway: evidenceStateSchema }).strict(),
  businessResearchStatus: businessResearchStatusSchema,
  investmentConclusionStatus: investmentConclusionStatusSchema,
  scorePermission: scorePermissionSchema,
  mosStatus: mosStatusSchema,
  valuationReliability: valuationReliabilitySchema,
  fiveYearDeltaPercentagePoints: z.number().finite(),
  tenYearDeltaPercentagePoints: z.number().finite(),
  eliteGates: eliteGatesSchema,
  deterministic: deterministicSchema.optional(),
}).strict();
export type CanonicalContractInput = z.infer<typeof contractShape>;

export interface CanonicalComputation {
  oqsRaw: number | "NOT_AVAILABLE";
  weakLinkCap: number | "NOT_AVAILABLE";
  oqs: number | "NOT_AVAILABLE";
  fiveYearReturnScore: number;
  tenYearReturnScore: number;
  returnComponent: number;
  ovs: ReturnType<typeof computeOvs>;
  investmentRaw: ReturnType<typeof computeInvestmentScore>["investmentRaw"];
  investmentScore: ReturnType<typeof computeInvestmentScore>["investmentScore"];
  orotitanStatus: ReturnType<typeof computeOroTitanStatus>;
}

export function computeCanonicalSnapshot(input: CanonicalContractInput): CanonicalComputation {
  let oqsResult: Pick<CanonicalComputation, "oqsRaw" | "weakLinkCap" | "oqs">;
  if (input.businessResearchStatus === "NOT_CERTIFIED" || input.scorePermission === "SUSPENDED") {
    oqsResult = { oqsRaw: "NOT_AVAILABLE", weakLinkCap: "NOT_AVAILABLE", oqs: "NOT_AVAILABLE" };
  } else {
    oqsResult = computeOqs(input.dimensions);
  }
  const fiveYearReturnScore = scoreExpectedReturnDelta(input.fiveYearDeltaPercentagePoints);
  const tenYearReturnScore = scoreExpectedReturnDelta(input.tenYearDeltaPercentagePoints);
  const returnComponent = computeReturnComponent(fiveYearReturnScore, tenYearReturnScore);
  const ovs = computeOvs({
    returnComponent, mosStatus: input.mosStatus, valuationReliability: input.valuationReliability,
    scorePermission: input.scorePermission, investmentConclusionStatus: input.investmentConclusionStatus,
  });
  const investment = computeInvestmentScore(oqsResult.oqs, ovs, input.scorePermission);
  return { ...oqsResult, fiveYearReturnScore, tenYearReturnScore, returnComponent, ovs,
    ...investment, orotitanStatus: computeOroTitanStatus(input.eliteGates, {
      businessResearchStatus: input.businessResearchStatus,
      investmentConclusionStatus: input.investmentConclusionStatus,
      scorePermission: input.scorePermission,
      valuationReliability: input.valuationReliability,
    }) };
}

function equivalent(left: unknown, right: unknown): boolean {
  return typeof left === "number" && typeof right === "number" ? Math.abs(left - right) <= 1e-10 : left === right;
}

export const canonicalContractSchema = contractShape.superRefine((input, context) => {
  const ceilingChecks = [
    ["MOAT", input.dimensions.MOAT, input.evidence.moat],
    ["RUNWAY", input.dimensions.RUNWAY, input.evidence.runway],
  ] as const;
  for (const [dimension, score, evidence] of ceilingChecks) {
    const ceiling = evidence === "PLAUSIBLE" ? 75 : evidence === "SUPPORTED" ? 90 : undefined;
    if (typeof score === "number" && ceiling !== undefined && score > ceiling) {
      context.addIssue({ code: "custom", path: ["dimensions", dimension], message: `${evidence} evidence caps ${dimension} at ${ceiling}` });
    }
  }
  if (input.evidence.moat === "FALSIFIED" && input.eliteGates.moatElite === "PASS") {
    context.addIssue({ code: "custom", path: ["eliteGates", "moatElite"], message: "FALSIFIED moat evidence is incompatible with an elite gate" });
  }
  if (input.evidence.runway === "FALSIFIED" && input.eliteGates.runwayElite === "PASS") {
    context.addIssue({ code: "custom", path: ["eliteGates", "runwayElite"], message: "FALSIFIED runway evidence is incompatible with an elite gate" });
  }
  if (input.eliteGates.moatElite === "PASS" && input.evidence.moat !== "STRONGLY_SUPPORTED") {
    context.addIssue({ code: "custom", path: ["eliteGates", "moatElite"], message: "PASS moat gate requires STRONGLY_SUPPORTED evidence" });
  }
  if (input.eliteGates.runwayElite === "PASS" && input.evidence.runway !== "STRONGLY_SUPPORTED") {
    context.addIssue({ code: "custom", path: ["eliteGates", "runwayElite"], message: "PASS runway gate requires STRONGLY_SUPPORTED evidence" });
  }
  let computed: CanonicalComputation;
  try {
    computed = computeCanonicalSnapshot(input);
  } catch (error) {
    context.addIssue({ code: "custom", path: ["dimensions"], message: error instanceof Error ? error.message : "Canonical computation failed" });
    return;
  }
  if (input.deterministic) {
    for (const [key, supplied] of Object.entries(input.deterministic)) {
      if (!equivalent(supplied, computed[key as keyof CanonicalComputation])) {
        context.addIssue({ code: "custom", path: ["deterministic", key], message: `Supplied value does not match canonical recomputation (${String(computed[key as keyof CanonicalComputation])})` });
      }
    }
  }
});

export function validateCanonicalContract(input: unknown) {
  return canonicalContractSchema.safeParse(input);
}
