import { z } from 'zod';

export const quoteUnits = ['MAJOR', 'MINOR'] as const;

const optionalText = z.string().trim().min(1).nullable();

export const companyInputSchema = z.object({
  slug: z.string().trim().min(2).max(80).regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, 'slug invalide'),
  ticker: z.string().trim().min(1).max(20),
  name: z.string().trim().min(2).max(160),
  exchange: z.string().trim().min(2).max(120),
  currency: z.string().trim().length(3).transform((value) => value.toUpperCase()),
  quote_unit: z.enum(quoteUnits),
  price_decimals: z.number().int().min(0).max(6),
  market_data_symbol: optionalText,
  market_data_multiplier: z.number().positive(),
  country: optionalText,
  sector: optionalText,
  active: z.boolean(),
});

export type CompanyInput = z.infer<typeof companyInputSchema>;
