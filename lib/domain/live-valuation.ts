import { scoreExpectedReturnDelta } from '@/lib/orotitan-equity/v1/expected-return';
import { computeInvestmentScore, computeOvs, computeReturnComponent } from '@/lib/orotitan-equity/v1/scoring';
import type {
  InvestmentConclusionStatus,
  MosStatus,
  ScorePermission,
  ValuationReliability,
} from '@/lib/orotitan-equity/v1/certification';

export interface LiveValuationInput {
  currentPrice: number | null;
  referencePrice: number | null;
  horizonYears: number | null;
  primaryExpectedReturn: number | null;
  matureNormalizationReturn: number | null;
  noMultipleExpansionReturn: number | null;
  requiredReturnH: number | null;
  marginOfSafety: MosStatus | null;
  valuationReliability: ValuationReliability | null;
  scorePermission: ScorePermission | null;
  investmentConclusionStatus: InvestmentConclusionStatus | null;
  oqs: number | null;
  canonicalOvs: number | null;
  canonicalInvestmentScore: number | null;
}

export interface LiveValuationResult {
  mode: 'PRICE_ONLY';
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
  canonicalOvs: number | null;
  canonicalInvestmentScore: number | null;
  ovsDelta: number | null;
  investmentScoreDelta: number | null;
  requiredReturnH: number;
  marginOfSafety: MosStatus;
  valuationReliability: ValuationReliability;
}

function finitePositive(value: number | null): value is number {
  return value !== null && Number.isFinite(value) && value > 0;
}

function finite(value: number | null): value is number {
  return value !== null && Number.isFinite(value);
}

export function repriceExpectedReturn(
  referenceReturnPct: number,
  referencePrice: number,
  currentPrice: number,
  horizonYears: number,
): number {
  if (!finitePositive(referencePrice) || !finitePositive(currentPrice) || !finitePositive(horizonYears)) {
    throw new Error('Price-only repricing requires positive price and horizon inputs');
  }
  if (!Number.isFinite(referenceReturnPct) || 1 + referenceReturnPct / 100 <= 0) {
    throw new Error('Reference expected return is invalid');
  }

  const terminalValue = referencePrice * Math.pow(1 + referenceReturnPct / 100, horizonYears);
  return (Math.pow(terminalValue / currentPrice, 1 / horizonYears) - 1) * 100;
}

export function computeLiveValuation(input: LiveValuationInput): LiveValuationResult | null {
  const normalizationReturn = finite(input.matureNormalizationReturn)
    ? input.matureNormalizationReturn
    : finite(input.noMultipleExpansionReturn)
      ? input.noMultipleExpansionReturn
      : null;

  if (
    !finitePositive(input.currentPrice) ||
    !finitePositive(input.referencePrice) ||
    !finitePositive(input.horizonYears) ||
    !finite(input.primaryExpectedReturn) ||
    normalizationReturn === null ||
    !finite(input.requiredReturnH) ||
    !finite(input.oqs) ||
    input.marginOfSafety === null ||
    input.marginOfSafety === 'NOT_ASSESSABLE' ||
    input.valuationReliability === null ||
    input.valuationReliability === 'NOT_ASSESSABLE' ||
    input.scorePermission === null ||
    input.scorePermission === 'SUSPENDED' ||
    input.investmentConclusionStatus === null ||
    input.investmentConclusionStatus === 'NOT_CERTIFIED' ||
    input.investmentConclusionStatus === 'INSUFFICIENT_DATA'
  ) {
    return null;
  }

  const primaryExpectedReturnPct = repriceExpectedReturn(
    input.primaryExpectedReturn,
    input.referencePrice,
    input.currentPrice,
    input.horizonYears,
  );
  const normalizationExpectedReturnPct = repriceExpectedReturn(
    normalizationReturn,
    input.referencePrice,
    input.currentPrice,
    input.horizonYears,
  );

  const primaryExpectedReturnScore = scoreExpectedReturnDelta(primaryExpectedReturnPct - input.requiredReturnH);
  const normalizedExpectedReturnScore = scoreExpectedReturnDelta(normalizationExpectedReturnPct - input.requiredReturnH);
  const returnComponentValue = computeReturnComponent(primaryExpectedReturnScore, normalizedExpectedReturnScore);
  if (typeof returnComponentValue !== 'number') return null;

  const ovsValue = computeOvs({
    returnComponent: returnComponentValue,
    mosStatus: input.marginOfSafety,
    valuationReliability: input.valuationReliability,
    scorePermission: input.scorePermission,
    investmentConclusionStatus: input.investmentConclusionStatus,
  });
  if (typeof ovsValue !== 'number') return null;

  const investment = computeInvestmentScore(input.oqs, ovsValue, input.scorePermission);
  if (typeof investment.investmentRaw !== 'number' || typeof investment.investmentScore !== 'number') return null;

  return {
    mode: 'PRICE_ONLY',
    currentPrice: input.currentPrice,
    referencePrice: input.referencePrice,
    priceChangeVsReferencePct: (input.currentPrice / input.referencePrice - 1) * 100,
    primaryExpectedReturnPct,
    normalizationExpectedReturnPct,
    primaryExpectedReturnScore,
    normalizedExpectedReturnScore,
    returnComponent: returnComponentValue,
    ovs: ovsValue,
    investmentRaw: investment.investmentRaw,
    investmentScore: investment.investmentScore,
    canonicalOvs: input.canonicalOvs,
    canonicalInvestmentScore: input.canonicalInvestmentScore,
    ovsDelta: input.canonicalOvs === null ? null : ovsValue - input.canonicalOvs,
    investmentScoreDelta: input.canonicalInvestmentScore === null ? null : investment.investmentScore - input.canonicalInvestmentScore,
    requiredReturnH: input.requiredReturnH,
    marginOfSafety: input.marginOfSafety,
    valuationReliability: input.valuationReliability,
  };
}
