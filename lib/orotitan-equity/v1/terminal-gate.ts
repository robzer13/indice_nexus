import { z } from "zod";
import type { BusinessResearchStatus, InvestmentConclusionStatus, ScorePermission, ValuationReliability } from "./certification";

export const gateStateSchema = z.enum(["PASS", "FAIL", "NOT_ASSESSABLE"]);
export const eliteGatesSchema = z.object({
  researchFullyCertified: gateStateSchema, moatElite: gateStateSchema, runwayElite: gateStateSchema,
  returnQualityElite: gateStateSchema, cashEconomicsElite: gateStateSchema, capitalAllocationElite: gateStateSchema,
  managementGovernanceElite: gateStateSchema, resilienceElite: gateStateSchema, valuationElite: gateStateSchema,
  materialWeakLink: gateStateSchema,
}).strict();
export type EliteGates = z.infer<typeof eliteGatesSchema>;
export type GateState = z.infer<typeof gateStateSchema>;
export type OroTitanStatus = "YES" | "NO";

export interface TerminalGateContext {
  businessResearchStatus: BusinessResearchStatus;
  investmentConclusionStatus: InvestmentConclusionStatus;
  scorePermission: ScorePermission;
  valuationReliability: ValuationReliability;
}

export function computeOroTitanStatus(gates: EliteGates, context: TerminalGateContext): OroTitanStatus {
  const certificationPass = context.businessResearchStatus === "CERTIFIED"
    && context.investmentConclusionStatus === "CERTIFIED"
    && context.scorePermission === "ALLOWED"
    && context.valuationReliability === "HIGH";
  const allGatesPass = Object.values(gates).every((state) => state === "PASS");
  return certificationPass && allGatesPass ? "YES" : "NO";
}
