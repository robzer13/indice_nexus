import { z } from "zod";

export const CERTIFICATION_STATES = [
  "CERTIFIED",
  "CERTIFIED_WITH_LIMITATIONS",
  "NOT_CERTIFIED",
  "INSUFFICIENT_DATA",
] as const;

export const businessResearchStatusSchema = z.enum(CERTIFICATION_STATES);
export const investmentConclusionStatusSchema = z.enum(CERTIFICATION_STATES);
export const scorePermissionSchema = z.enum(["ALLOWED", "CONDITIONAL", "SUSPENDED"]);
export const valuationReliabilitySchema = z.enum(["HIGH", "MEDIUM", "LOW", "NOT_ASSESSABLE"]);
export const mosStatusSchema = z.enum(["ROBUST", "ADEQUATE", "THIN", "NONE", "NOT_ASSESSABLE"]);

export type BusinessResearchStatus = z.infer<typeof businessResearchStatusSchema>;
export type InvestmentConclusionStatus = z.infer<typeof investmentConclusionStatusSchema>;
export type ScorePermission = z.infer<typeof scorePermissionSchema>;
export type ValuationReliability = z.infer<typeof valuationReliabilitySchema>;
export type MosStatus = z.infer<typeof mosStatusSchema>;
