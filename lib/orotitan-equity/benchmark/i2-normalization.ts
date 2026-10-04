export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface NormalizedI2Report {
  status: 'PASS' | 'FAIL' | 'UNKNOWN';
  exactReconciliation: boolean | null;
  authorityName: string | null;
  authorityVersion: string | null;
  authoritySha256: string | null;
  recomputedOutputs: JsonValue | null;
  admittedOutputs: JsonValue | null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function firstString(record: Record<string, unknown>, keys: string[]): string | null {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'string' && value.length > 0) return value;
  }
  return null;
}

function normalizeNumericJson(value: unknown): JsonValue {
  if (value === null) return null;
  if (typeof value === 'boolean' || typeof value === 'number') return value;
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (trimmed !== '' && /^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(trimmed)) {
      const parsed = Number(trimmed);
      if (Number.isFinite(parsed)) return parsed;
    }
    return value;
  }
  if (Array.isArray(value)) return value.map(normalizeNumericJson);
  if (typeof value === 'object') {
    const output: Record<string, JsonValue> = {};
    for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
      output[key] = normalizeNumericJson(nested);
    }
    return output;
  }
  return String(value);
}

export function normalizeI2Report(raw: unknown): NormalizedI2Report {
  const report = asRecord(raw);
  if (!report) {
    return {
      status: 'UNKNOWN',
      exactReconciliation: null,
      authorityName: null,
      authorityVersion: null,
      authoritySha256: null,
      recomputedOutputs: null,
      admittedOutputs: null,
    };
  }

  const declared = firstString(report, ['status', 'reconciliation_status', 'i2_reconciliation']);
  const status = declared === 'PASS' ? 'PASS' : declared === 'FAIL' ? 'FAIL' : 'UNKNOWN';

  const comparisons = asRecord(report.comparisons);
  const exact =
    typeof report.exact_reconciliation === 'boolean'
      ? report.exact_reconciliation
      : typeof comparisons?.all_deterministic_outputs_equal === 'boolean'
        ? comparisons.all_deterministic_outputs_equal
        : null;

  const authority = asRecord(report.authority);
  const recomputed = report.recomputation ?? report.recomputed ?? null;
  const admitted =
    report.admitted_outputs ??
    report.authoritative_deep_dive_outputs ??
    report.authoritative_outputs ??
    null;

  return {
    status,
    exactReconciliation: exact,
    authorityName: authority ? firstString(authority, ['name']) : null,
    authorityVersion: authority ? firstString(authority, ['version']) : null,
    authoritySha256: authority ? firstString(authority, ['content_sha256', 'sha256']) : null,
    recomputedOutputs: recomputed === null ? null : normalizeNumericJson(recomputed),
    admittedOutputs: admitted === null ? null : normalizeNumericJson(admitted),
  };
}

function stable(value: JsonValue): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stable).join(',')}]`;
  const record = value as Record<string, JsonValue>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${stable(record[key])}`).join(',')}}`;
}

export function deterministicOutputsEqual(report: NormalizedI2Report): boolean | null {
  if (report.exactReconciliation !== null) return report.exactReconciliation;
  if (report.recomputedOutputs === null || report.admittedOutputs === null) return null;
  return stable(report.recomputedOutputs) === stable(report.admittedOutputs);
}
