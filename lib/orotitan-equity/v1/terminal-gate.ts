import { z } from "zod";

export const eliteGatesSchema = z.object({
  researchFullyCertified: z.boolean(), moatElite: z.boolean(), runwayElite: z.boolean(),
  returnQualityElite: z.boolean(), cashEconomicsElite: z.boolean(), capitalAllocationElite: z.boolean(),
  managementGovernanceElite: z.boolean(), resilienceElite: z.boolean(), valuationElite: z.boolean(),
  materialWeakLink: z.enum(["YES", "NO"]),
}).strict();
export type EliteGates = z.infer<typeof eliteGatesSchema>;
export type OroTitanStatus = "YES" | "NO";

export function computeOroTitanStatus(gates: EliteGates): OroTitanStatus {
  return gates.researchFullyCertified && gates.moatElite && gates.runwayElite && gates.returnQualityElite
    && gates.cashEconomicsElite && gates.capitalAllocationElite && gates.managementGovernanceElite
    && gates.resilienceElite && gates.valuationElite && gates.materialWeakLink === "NO" ? "YES" : "NO";
}
