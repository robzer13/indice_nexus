import {
  categoricalAgreementRate,
  computeStabilityMetrics,
  verdictFlipRate,
  type StabilityMetrics,
} from './metrics';

export const BENCHMARK_DIMENSIONS = [
  'MOAT',
  'RUNWAY',
  'RETURN_QUALITY',
  'CASH_ECONOMICS',
  'CAPITAL_ALLOCATION',
  'MANAGEMENT_GOVERNANCE',
  'RESILIENCE_RISK',
] as const;

export type BenchmarkDimension = typeof BENCHMARK_DIMENSIONS[number];

export interface ReproducibilityExecution {
  caseId: string;
  repetitionIndex: number;
  dimensionScores: Record<BenchmarkDimension, number | null>;
  oqs: number | null;
  certificationStatus: string | null;
  terminalVerdict: string | null;
  i2ReconciliationPass: boolean | null;
}

export interface CaseReproducibilityResult {
  caseId: string;
  executions: number;
  dimensions: Record<BenchmarkDimension, StabilityMetrics>;
  oqs: StabilityMetrics;
  certificationAgreementRate: number | null;
  certificationFlipRate: number | null;
  terminalVerdictAgreementRate: number | null;
  terminalVerdictFlipRate: number | null;
  i2PassRate: number | null;
}

function categoricalValues(values: Array<string | null>): string[] {
  return values.filter((value): value is string => value !== null);
}

function booleanPassRate(values: Array<boolean | null>): number | null {
  const clean = values.filter((value): value is boolean => value !== null);
  if (clean.length === 0) return null;
  return clean.filter(Boolean).length / clean.length;
}

export function evaluateCaseReproducibility(executions: ReproducibilityExecution[]): CaseReproducibilityResult {
  if (executions.length === 0) throw new Error('At least one execution is required.');
  const caseId = executions[0].caseId;
  if (executions.some((execution) => execution.caseId !== caseId)) {
    throw new Error('All executions must belong to the same benchmark case.');
  }

  const dimensions = Object.fromEntries(
    BENCHMARK_DIMENSIONS.map((dimension) => [
      dimension,
      computeStabilityMetrics(
        executions
          .map((execution) => execution.dimensionScores[dimension])
          .filter((value): value is number => value !== null),
      ),
    ]),
  ) as Record<BenchmarkDimension, StabilityMetrics>;

  const certifications = categoricalValues(executions.map((execution) => execution.certificationStatus));
  const verdicts = categoricalValues(executions.map((execution) => execution.terminalVerdict));

  return {
    caseId,
    executions: executions.length,
    dimensions,
    oqs: computeStabilityMetrics(
      executions.map((execution) => execution.oqs).filter((value): value is number => value !== null),
    ),
    certificationAgreementRate: categoricalAgreementRate(certifications),
    certificationFlipRate: verdictFlipRate(certifications),
    terminalVerdictAgreementRate: categoricalAgreementRate(verdicts),
    terminalVerdictFlipRate: verdictFlipRate(verdicts),
    i2PassRate: booleanPassRate(executions.map((execution) => execution.i2ReconciliationPass)),
  };
}

export interface CampaignReproducibilitySummary {
  cases: number;
  executions: number;
  weightedOqsMeanAbsoluteDeviation: number | null;
  exactOqsAgreementRate: number | null;
  terminalVerdictFlipRate: number | null;
  certificationFlipRate: number | null;
  i2PassRate: number | null;
}

function weightedAverage(pairs: Array<{ value: number | null; weight: number }>): number | null {
  const clean = pairs.filter((pair): pair is { value: number; weight: number } => pair.value !== null && pair.weight > 0);
  if (clean.length === 0) return null;
  const totalWeight = clean.reduce((sum, pair) => sum + pair.weight, 0);
  return clean.reduce((sum, pair) => sum + pair.value * pair.weight, 0) / totalWeight;
}

export function summarizeCampaign(results: CaseReproducibilityResult[]): CampaignReproducibilitySummary {
  return {
    cases: results.length,
    executions: results.reduce((sum, result) => sum + result.executions, 0),
    weightedOqsMeanAbsoluteDeviation: weightedAverage(
      results.map((result) => ({ value: result.oqs.meanAbsoluteDeviation, weight: result.oqs.n })),
    ),
    exactOqsAgreementRate: weightedAverage(
      results.map((result) => ({ value: result.oqs.exactAgreementRate, weight: result.oqs.n })),
    ),
    terminalVerdictFlipRate: weightedAverage(
      results.map((result) => ({ value: result.terminalVerdictFlipRate, weight: result.executions })),
    ),
    certificationFlipRate: weightedAverage(
      results.map((result) => ({ value: result.certificationFlipRate, weight: result.executions })),
    ),
    i2PassRate: weightedAverage(
      results.map((result) => ({ value: result.i2PassRate, weight: result.executions })),
    ),
  };
}
