import type {
  InvestmentConclusionStatus,
  MosStatus,
  ScorePermission,
  ValuationReliability,
} from './v1/certification';
import { scoreExpectedReturnDelta } from './v1/expected-return';
import { computeInvestmentScore, computeOvs, computeReturnComponent } from './v1/scoring';

export interface MarketAdaptiveValuationInput {
  currentPrice: number;
  referencePrice: number;
  horizonYears: number;
  primaryExpectedReturnPct: number;
  normalizationExpectedReturnPct: number;
  requiredReturnPct: number;
  mosStatus: MosStatus;
  valuationReliability: ValuationReliability;
  scorePermission: ScorePermission;
  investmentConclusionStatus: InvestmentConclusionStatus;
  oqs: number;
}

export interface MarketAdaptiveValuation {
  currentPrice: number;
  referencePrice: number;
  priceChangeVsReferencePct: number;
  primaryExpectedReturnPct: number;
  normalizationExpectedReturnPct: number;
  primaryExpectedReturnScore: number;
  normalizedExpectedReturnScore: number;
  returnComponent: number;
  ovs: number;
  investmentRaw: number;
  investmentScore: number;
}

function rebaseAnnualizedReturn(
  referencePrice: number,
  currentPrice: number,
  annualizedReturnPct: number,
  horizonYears: number,
): number | null {
  if (
    !Number.isFinite(referencePrice) ||
    !Number.isFinite(currentPrice) ||
    !Number.isFinite(annualizedReturnPct) ||
    !Number.isFinite(horizonYears) ||
    referencePrice <= 0 ||
    currentPrice <= 0 ||
    horizonYears <= 0 ||
    annualizedReturnPct <= -100
  ) {
    return null;
  }

  const terminalPrice = referencePrice * ((1 + annualizedReturnPct / 100) ** horizonYears);
  if (!Number.isFinite(terminalPrice) || terminalPrice <= 0) return null;
  return (((terminalPrice / currentPrice) ** (1 / horizonYears)) - 1) * 100;
}

export function computeMarketAdaptiveValuation(
  input: MarketAdaptiveValuationInput,
): MarketAdaptiveValuation | null {
  const primaryExpectedReturnPct = rebaseAnnualizedReturn(
    input.referencePrice,
    input.currentPrice,
    input.primaryExpectedReturnPct,
    input.horizonYears,
  );
  const normalizationExpectedReturnPct = rebaseAnnualizedReturn(
    input.referencePrice,
    input.currentPrice,
    input.normalizationExpectedReturnPct,
    input.horizonYears,
  );

  if (primaryExpectedReturnPct === null || normalizationExpectedReturnPct === null) return null;

  const primaryExpectedReturnScore = scoreExpectedReturnDelta(
    primaryExpectedReturnPct - input.requiredReturnPct,
  );
  const normalizedExpectedReturnScore = scoreExpectedReturnDelta(
    normalizationExpectedReturnPct - input.requiredReturnPct,
  );
  const returnComponent = computeReturnComponent(
    primaryExpectedReturnScore,
    normalizedExpectedReturnScore,
  );
  if (typeof returnComponent !== 'number') return null;

  const ovs = computeOvs({
    returnComponent,
    mosStatus: input.mosStatus,
    valuationReliability: input.valuationReliability,
    scorePermission: input.scorePermission,
    investmentConclusionStatus: input.investmentConclusionStatus,
  });
  if (typeof ovs !== 'number') return null;

  const investment = computeInvestmentScore(input.oqs, ovs, input.scorePermission);
  if (typeof investment.investmentRaw !== 'number' || typeof investment.investmentScore !== 'number') {
    return null;
  }

  return {
    currentPrice: input.currentPrice,
    referencePrice: input.referencePrice,
    priceChangeVsReferencePct: ((input.currentPrice / input.referencePrice) - 1) * 100,
    primaryExpectedReturnPct,
    normalizationExpectedReturnPct,
    primaryExpectedReturnScore,
    normalizedExpectedReturnScore,
    returnComponent,
    ovs,
    investmentRaw: investment.investmentRaw,
    investmentScore: investment.investmentScore,
  };
}
