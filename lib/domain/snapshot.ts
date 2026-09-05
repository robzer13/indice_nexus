import { z } from 'zod';

export const snapshotStatuses = [
  'OROTITAN',
  'FINALIST',
  'PRICE_WAIT',
  'TIER_1',
  'WATCHLIST',
  'REJECTED',
] as const;

const nullableScore100 = z.number().min(0).max(100).nullable();
const nullableConfidence = z.number().min(0).max(10).nullable();
const nullablePositivePrice = z.number().positive().nullable();

export const snapshotInputSchema = z.object({
  company_id: z.string().uuid(),
  analysis_date: z.iso.date(),
  model_version: z.string().trim().min(1).max(120),
  status: z.enum(snapshotStatuses),
  quality_orotitan: z.boolean().nullable(),
  business_quality_score: nullableScore100,
  investment_score: nullableScore100,
  valuation_score: nullableScore100,
  orotitan_score: nullableScore100,
  confidence_score: nullableConfidence,
  fair_value_low: nullablePositivePrice,
  fair_value_base: nullablePositivePrice,
  fair_value_high: nullablePositivePrice,
  price_o85: nullablePositivePrice,
  price_o90: nullablePositivePrice,
  price_o92: nullablePositivePrice,
  price_o95: nullablePositivePrice,
  thesis: z.string().trim().min(1).nullable(),
  main_risk: z.string().trim().min(1).nullable(),
  invalidation: z.string().trim().min(1).nullable(),
  source_title: z.string().trim().min(1, 'source_title est obligatoire'),
  notes: z.string().trim().min(1).nullable(),
  score_components: z.record(z.string(), z.unknown()),
});

export type SnapshotInput = z.infer<typeof snapshotInputSchema>;

export class DuplicateSnapshotError extends Error {
  constructor() {
    super(
      'Un snapshot existe déjà pour cette société, cette date et cette model_version. Utilisez une nouvelle model_version ou une nouvelle date.',
    );
    this.name = 'DuplicateSnapshotError';
  }
}

export function assertSnapshotDoesNotExist(exists: boolean): void {
  if (exists) {
    throw new DuplicateSnapshotError();
  }
}
