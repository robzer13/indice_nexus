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
  primaryExpectedReturnPct: number | null;
  normalizationExpectedReturnPct: number | null;
  requiredReturnPct: number | null;
  marginOfSafety: string | null;
  valuationReliability: string | null;
  scorePermission: string | null;
  investmentConclusionStatus: string | null;
  oqs: number | null;
}

export interface LiveValuationResult {
  available: boolean;
  primaryExpectedReturnPct: number | null;
  normalizationExpectedReturnPct: number | null;
  valuationScore: number | null;
  investmentRaw: number | null;
  investmentScore: number | null;
  priceChangeVsReferencePct: number | null;
  reason: string | null;
}

const MOS_VALUES = new Set<MosStatus>(['ROBUST', 'ADEQUATE', 'THIN', 'NONE', 'NOT_ASSESSABLE']);
const RELIABILITY_VALUES = new Set<ValuationReliability>(['HIGH', 'MEDIUM', 'LOW', 'NOT_ASSESSABLE']);
const PERMISSION_VALUES = new Set<ScorePermission>(['ALLOWED', 'CONDITIONAL', 'SUSPENDED']);
const CONCLUSION_VALUES = new Set<InvestmentConclusionStatus>(['CERTIFIED', 'CERTIFIED_WITH_LIMITATIONS', 'NOT_CERTIFIED', 'INSUFFICIENT_DATA']);

function finitePositive(value: number | null): value is number {
  return value !== null && Number.isFinite(value) && value > 0;
}

function scalar(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asMosStatus(value: string | null): MosStatus | null {
  return value && MOS_VALUES.has(value as MosStatus) ? value as MosStatus : null;
}

function asReliability(value: string | null): ValuationReliability | null {
  return value && RELIABILITY_VALUES.has(value as ValuationReliability) ? value as ValuationReliability : null;
}

function asPermission(value: string | null): ScorePermission | null {
  return value && PERMISSION_VALUES.has(value as ScorePermission) ? value as ScorePermission : null;
}

function asConclusion(value: string | null): InvestmentConclusionStatus | null {
  return value && CONCLUSION_VALUES.has(value as InvestmentConclusionStatus) ? value as InvestmentConclusionStatus : null;
}

/**
 * Reprices an annualized expected return while holding the certified terminal
 * economic outcome constant. Only the entry price changes.
 */
export function repriceAnnualizedReturnPct(input: {
  currentPrice: number;
  referencePrice: number;
  referenceAnnualizedReturnPct: number;
  horizonYears: number;
}): number | null {
  const { currentPrice, referencePrice, referenceAnnualizedReturnPct, horizonYears } = input;
  if (!Number.isFinite(currentPrice) || currentPrice <= 0) return null;
  if (!Number.isFinite(referencePrice) || referencePrice <= 0) return null;
  if (!Number.isFinite(horizonYears) || horizonYears <= 0) return null;
  if (!Number.isFinite(referenceAnnualizedReturnPct) || referenceAnnualizedReturnPct <= -100) return null;

  const terminalFactor = Math.pow(1 + referenceAnnualizedReturnPct / 100, horizonYears);
  const terminalValue = referencePrice * terminalFactor;
  if (!Number.isFinite(terminalValue) || terminalValue <= 0) return null;

  const repriced = (Math.pow(terminalValue / currentPrice, 1 / horizonYears) - 1) * 100;
  return Number.isFinite(repriced) ? repriced : null;
}

export function computeLiveValuation(input: LiveValuationInput): LiveValuationResult {
  const unavailable = (reason: string): LiveValuationResult => ({
    available: false,
    primaryExpectedReturnPct: null,
    normalizationExpectedReturnPct: null,
    valuationScore: null,
    investmentRaw: null,
    investmentScore: null,
    priceChangeVsReferencePct: finitePositive(input.currentPrice) && finitePositive(input.referencePrice)
      ? (input.currentPrice / input.referencePrice - 1) * 100
      : null,
    reason,
  });

  if (!finitePositive(input.currentPrice) || !finitePositive(input.referencePrice)) return unavailable('Cours courant ou cours de référence indisponible.');
  if (!finitePositive(input.horizonYears)) return unavailable('Horizon de rendement indisponible.');
  if (input.primaryExpectedReturnPct === null || input.normalizationExpectedReturnPct === null || input.requiredReturnPct === null || input.oqs === null) {
    return unavailable('Entrées de valorisation certifiées incomplètes.');
  }

  const marginOfSafety = asMosStatus(input.marginOfSafety);
  const valuationReliability = asReliability(input.valuationReliability);
  const scorePermission = asPermission(input.scorePermission);
  const investmentConclusionStatus = asConclusion(input.investmentConclusionStatus);
  if (!marginOfSafety || !valuationReliability || !scorePermission || !investmentConclusionStatus) {
    return unavailable('États de certification incompatibles avec un recalcul live.');
  }

  const primaryExpectedReturnPct = repriceAnnualizedReturnPct({
    currentPrice: input.currentPrice,
    referencePrice: input.referencePrice,
    referenceAnnualizedReturnPct: input.primaryExpectedReturnPct,
    horizonYears: input.horizonYears,
  });
  const normalizationExpectedReturnPct = repriceAnnualizedReturnPct({
    currentPrice: input.currentPrice,
    referencePrice: input.referencePrice,
    referenceAnnualizedReturnPct: input.normalizationExpectedReturnPct,
    horizonYears: input.horizonYears,
  });
  if (primaryExpectedReturnPct === null || normalizationExpectedReturnPct === null) {
    return unavailable('Rendement live non calculable.');
  }

  const primaryScore = scoreExpectedReturnDelta(primaryExpectedReturnPct - input.requiredReturnPct);
  const normalizationScore = scoreExpectedReturnDelta(normalizationExpectedReturnPct - input.requiredReturnPct);
  const returnComponent = computeReturnComponent(primaryScore, normalizationScore);
  const valuationScore = computeOvs({
    returnComponent,
    mosStatus: marginOfSafety,
    valuationReliability,
    scorePermission,
    investmentConclusionStatus,
  });
  const numericValuationScore = scalar(valuationScore);
  if (numericValuationScore === null) return unavailable('OVS live non numérique avec les états de certification courants.');

  const investment = computeInvestmentScore(input.oqs, numericValuationScore, scorePermission);
  const investmentRaw = scalar(investment.investmentRaw);
  const investmentScore = scalar(investment.investmentScore);
  if (investmentRaw === null || investmentScore === null) return unavailable('Score investissement live non numérique.');

  return {
    available: true,
    primaryExpectedReturnPct,
    normalizationExpectedReturnPct,
    valuationScore: numericValuationScore,
    investmentRaw,
    investmentScore,
    priceChangeVsReferencePct: (input.currentPrice / input.referencePrice - 1) * 100,
    reason: null,
  };
}
