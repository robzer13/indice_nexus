import { z } from "zod";
import {
  businessResearchStatusSchema, investmentConclusionStatusSchema, mosStatusSchema,
  scorePermissionSchema, valuationReliabilitySchema,
} from "./certification";
import { scoreExpectedReturnRange } from "./expected-return";
import { computeOqs, moatEvidenceStateSchema, qualityDimensionsSchema, runwayEvidenceStateSchema } from "./quality";
import { computeInvestmentScore, computeOvs, computeReturnComponent } from "./scoring";
import { canonicalScoreSchema, scoreRangeSchema, type CanonicalScore } from "./semantic-states";
import { computeOroTitanStatus, eliteGatesSchema } from "./terminal-gate";

const deterministicSchema = z.object({
  oqsRaw: canonicalScoreSchema.optional(), weakLinkCap: canonicalScoreSchema.optional(), oqs: canonicalScoreSchema.optional(),
  primaryExpectedReturnScore: canonicalScoreSchema.optional(), normalizedExpectedReturnScore: canonicalScoreSchema.optional(),
  returnComponent: canonicalScoreSchema.optional(), ovs: canonicalScoreSchema.optional(),
  investmentRaw: canonicalScoreSchema.optional(), investmentScore: canonicalScoreSchema.optional(),
  orotitanStatus: z.enum(["YES", "NO"]).optional(),
}).strict();

const contractShape = z.object({
  dimensions: qualityDimensionsSchema,
  evidence: z.object({ moat: moatEvidenceStateSchema, runway: runwayEvidenceStateSchema }).strict(),
  businessResearchStatus: businessResearchStatusSchema,
  investmentConclusionStatus: investmentConclusionStatusSchema,
  scorePermission: scorePermissionSchema,
  mosStatus: mosStatusSchema,
  valuationReliability: valuationReliabilitySchema,
  primaryExpectedReturnDeltaPercentagePoints: z.union([z.number().finite(), scoreRangeSchema]),
  normalizedExpectedReturnDeltaPercentagePoints: z.union([z.number().finite(), scoreRangeSchema]),
  eliteGates: eliteGatesSchema,
  deterministic: deterministicSchema.optional(),
}).strict();
export type CanonicalContractInput = z.infer<typeof contractShape>;

export interface CanonicalComputation {
  oqsRaw: CanonicalScore;
  weakLinkCap: CanonicalScore;
  oqs: CanonicalScore;
  primaryExpectedReturnScore: CanonicalScore;
  normalizedExpectedReturnScore: CanonicalScore;
  returnComponent: CanonicalScore;
  ovs: ReturnType<typeof computeOvs>;
  investmentRaw: ReturnType<typeof computeInvestmentScore>["investmentRaw"];
  investmentScore: ReturnType<typeof computeInvestmentScore>["investmentScore"];
  orotitanStatus: ReturnType<typeof computeOroTitanStatus>;
}

export function computeCanonicalSnapshot(input: CanonicalContractInput): CanonicalComputation {
  let oqsResult: Pick<CanonicalComputation, "oqsRaw" | "weakLinkCap" | "oqs">;
  if (input.scorePermission === "SUSPENDED") {
    oqsResult = { oqsRaw: "NOT_AVAILABLE", weakLinkCap: "NOT_AVAILABLE", oqs: "NOT_AVAILABLE" };
  } else {
    const computedOqs = computeOqs(input.dimensions);
    oqsResult = input.businessResearchStatus === "NOT_CERTIFIED"
      ? { ...computedOqs, oqs: "NOT_AVAILABLE" }
      : computedOqs;
  }
  const primaryExpectedReturnScore = scoreExpectedReturnRange(input.primaryExpectedReturnDeltaPercentagePoints);
  const normalizedExpectedReturnScore = scoreExpectedReturnRange(input.normalizedExpectedReturnDeltaPercentagePoints);
  const returnComponent = computeReturnComponent(primaryExpectedReturnScore, normalizedExpectedReturnScore);
  const ovs = computeOvs({
    returnComponent, mosStatus: input.mosStatus, valuationReliability: input.valuationReliability,
    scorePermission: input.scorePermission, investmentConclusionStatus: input.investmentConclusionStatus,
  });
  const investment = computeInvestmentScore(oqsResult.oqs, ovs, input.scorePermission);
  return { ...oqsResult, primaryExpectedReturnScore, normalizedExpectedReturnScore, returnComponent, ovs,
    ...investment, orotitanStatus: computeOroTitanStatus(input.eliteGates, {
      businessResearchStatus: input.businessResearchStatus,
      investmentConclusionStatus: input.investmentConclusionStatus,
      scorePermission: input.scorePermission,
      valuationReliability: input.valuationReliability,
    }) };
}

function equivalent(left: unknown, right: unknown): boolean {
  if (typeof left === "number" && typeof right === "number") return Math.abs(left - right) <= 1e-10;
  if (left && right && typeof left === "object" && typeof right === "object" && "min" in left && "min" in right && "max" in left && "max" in right) {
    return equivalent(left.min, right.min) && equivalent(left.max, right.max);
  }
  return left === right;
}

export const canonicalContractSchema = contractShape.superRefine((input, context) => {
  const ceilingChecks = [
    ["MOAT", input.dimensions.MOAT, input.evidence.moat],
    ["RUNWAY", input.dimensions.RUNWAY, input.evidence.runway],
  ] as const;
  for (const [dimension, score, evidence] of ceilingChecks) {
    const ceiling = evidence === "PLAUSIBLE" ? 75 : evidence === "SUPPORTED" ? 90 : undefined;
    const scoreMax = typeof score === "number" ? score : score && typeof score === "object" && "max" in score ? score.max : undefined;
    if (scoreMax !== undefined && ceiling !== undefined && scoreMax > ceiling) {
      context.addIssue({ code: "custom", path: ["dimensions", dimension], message: `${evidence} evidence caps ${dimension} at ${ceiling}` });
    }
  }
  if (input.evidence.moat === "FALSIFIED" && input.eliteGates.moatElite === "PASS") {
    context.addIssue({ code: "custom", path: ["eliteGates", "moatElite"], message: "FALSIFIED moat evidence is incompatible with an elite gate" });
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
