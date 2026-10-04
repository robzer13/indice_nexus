export interface StabilityMetrics {
  n: number;
  mean: number | null;
  min: number | null;
  max: number | null;
  range: number | null;
  meanAbsoluteDeviation: number | null;
  exactAgreementRate: number | null;
}

export interface ClassificationMetrics {
  precision: number | null;
  recall: number | null;
  f1: number | null;
}

export interface ForecastMetrics {
  n: number;
  mae: number | null;
  bias: number | null;
  rmse: number | null;
  spearman: number | null;
}

function finite(values: number[]): number[] {
  return values.filter((value) => Number.isFinite(value));
}

function modeFrequency<T extends string | number | boolean>(values: T[]): number {
  const counts = new Map<T, number>();
  let max = 0;
  for (const value of values) {
    const next = (counts.get(value) ?? 0) + 1;
    counts.set(value, next);
    if (next > max) max = next;
  }
  return max;
}

export function computeStabilityMetrics(values: number[]): StabilityMetrics {
  const clean = finite(values);
  if (clean.length === 0) {
    return { n: 0, mean: null, min: null, max: null, range: null, meanAbsoluteDeviation: null, exactAgreementRate: null };
  }
  const mean = clean.reduce((sum, value) => sum + value, 0) / clean.length;
  const min = Math.min(...clean);
  const max = Math.max(...clean);
  const mad = clean.reduce((sum, value) => sum + Math.abs(value - mean), 0) / clean.length;
  return {
    n: clean.length,
    mean,
    min,
    max,
    range: max - min,
    meanAbsoluteDeviation: mad,
    exactAgreementRate: modeFrequency(clean) / clean.length,
  };
}

export function categoricalAgreementRate<T extends string | number | boolean>(values: T[]): number | null {
  if (values.length === 0) return null;
  return modeFrequency(values) / values.length;
}

export function verdictFlipRate<T extends string | number | boolean>(values: T[]): number | null {
  const agreement = categoricalAgreementRate(values);
  return agreement === null ? null : 1 - agreement;
}

export function computeClassificationMetrics(tp: number, fp: number, fn: number): ClassificationMetrics {
  const precisionDenominator = tp + fp;
  const recallDenominator = tp + fn;
  const precision = precisionDenominator > 0 ? tp / precisionDenominator : null;
  const recall = recallDenominator > 0 ? tp / recallDenominator : null;
  const f1 = precision !== null && recall !== null && precision + recall > 0
    ? 2 * precision * recall / (precision + recall)
    : null;
  return { precision, recall, f1 };
}

function ranks(values: number[]): number[] {
  const indexed = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
  const output = new Array<number>(values.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i + 1;
    while (j < indexed.length && indexed[j].value === indexed[i].value) j += 1;
    const averageRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k += 1) output[indexed[k].index] = averageRank;
    i = j;
  }
  return output;
}

function pearson(a: number[], b: number[]): number | null {
  if (a.length !== b.length || a.length < 2) return null;
  const meanA = a.reduce((sum, value) => sum + value, 0) / a.length;
  const meanB = b.reduce((sum, value) => sum + value, 0) / b.length;
  let numerator = 0;
  let denomA = 0;
  let denomB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    numerator += da * db;
    denomA += da * da;
    denomB += db * db;
  }
  if (denomA === 0 || denomB === 0) return null;
  return numerator / Math.sqrt(denomA * denomB);
}

export function spearmanRankCorrelation(a: number[], b: number[]): number | null {
  if (a.length !== b.length || a.length < 2) return null;
  if (a.some((value) => !Number.isFinite(value)) || b.some((value) => !Number.isFinite(value))) return null;
  return pearson(ranks(a), ranks(b));
}

export function computeForecastMetrics(forecasts: number[], realized: number[]): ForecastMetrics {
  if (forecasts.length !== realized.length) throw new Error('Forecast and realized arrays must have equal length.');
  const pairs = forecasts
    .map((forecast, index) => ({ forecast, realized: realized[index] }))
    .filter((pair) => Number.isFinite(pair.forecast) && Number.isFinite(pair.realized));

  if (pairs.length === 0) return { n: 0, mae: null, bias: null, rmse: null, spearman: null };

  const errors = pairs.map((pair) => pair.forecast - pair.realized);
  const mae = errors.reduce((sum, error) => sum + Math.abs(error), 0) / errors.length;
  const bias = errors.reduce((sum, error) => sum + error, 0) / errors.length;
  const rmse = Math.sqrt(errors.reduce((sum, error) => sum + error * error, 0) / errors.length);

  return {
    n: pairs.length,
    mae,
    bias,
    rmse,
    spearman: spearmanRankCorrelation(
      pairs.map((pair) => pair.forecast),
      pairs.map((pair) => pair.realized),
    ),
  };
}
