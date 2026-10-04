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
  tolerance: number | null;
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

function firstNumber(record: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
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

function allRecordValues(record: Record<string, unknown> | null, predicate: (value: unknown) => boolean): boolean | null {
  if (!record) return null;
  const values = Object.values(record);
  if (values.length === 0) return null;
  return values.every(predicate);
}

function parseAuthority(report: Record<string, unknown>) {
  const authorityRecord = asRecord(report.authority) ?? asRecord(report.i2_authority);
  if (authorityRecord) {
    return {
      name: firstString(authorityRecord, ['name']),
      version: firstString(authorityRecord, ['version']),
      sha256: firstString(authorityRecord, ['content_sha256', 'sha256']),
    };
  }
  if (typeof report.authority === 'string') {
    const [name, version] = report.authority.split('@');
    return { name: name || null, version: version || null, sha256: null };
  }
  return { name: null, version: null, sha256: null };
}

export function normalizeI2Report(raw: unknown): NormalizedI2Report {
  const report = asRecord(raw);
  if (!report) {
    return {
      status: 'UNKNOWN',
      exactReconciliation: null,
      tolerance: null,
      authorityName: null,
      authorityVersion: null,
      authoritySha256: null,
      recomputedOutputs: null,
      admittedOutputs: null,
    };
  }

  const declared = firstString(report, [
    'status',
    'reconciliation_status',
    'i2_reconciliation',
    'i2_reconciliation_status',
    'result',
  ]);
  const status = declared === 'PASS' ? 'PASS' : declared === 'FAIL' ? 'FAIL' : 'UNKNOWN';

  const comparisons = asRecord(report.comparisons);
  const candidateMatch = asRecord(report.candidate_match);
  const absoluteDeltas = asRecord(report.absolute_deltas);

  let exact: boolean | null =
    typeof report.exact_reconciliation === 'boolean'
      ? report.exact_reconciliation
      : typeof report.exact_match === 'boolean'
        ? report.exact_match
        : typeof comparisons?.all_deterministic_outputs_equal === 'boolean'
          ? comparisons.all_deterministic_outputs_equal
          : null;

  if (exact === null) {
    const candidateExact = allRecordValues(candidateMatch, (value) => value === 'PASS');
    if (candidateExact !== null) exact = candidateExact;
  }
  if (exact === null) {
    const deltasExact = allRecordValues(absoluteDeltas, (value) => {
      const parsed = typeof value === 'number' ? value : Number(value);
      return Number.isFinite(parsed) && parsed === 0;
    });
    if (deltasExact !== null) exact = deltasExact;
  }

  const authority = parseAuthority(report);
  const recomputed = report.recomputation ?? report.recomputed ?? null;
  const admitted =
    report.admitted_outputs ??
    report.authoritative_deep_dive_outputs ??
    report.authoritative_outputs ??
    report.admitted ??
    report.persisted ??
    report.certified ??
    null;

  return {
    status,
    exactReconciliation: exact,
    tolerance: firstNumber(report, ['tolerance']),
    authorityName: authority.name,
    authorityVersion: authority.version,
    authoritySha256: authority.sha256,
    recomputedOutputs: recomputed === null ? null : normalizeNumericJson(recomputed),
    admittedOutputs: admitted === null ? null : normalizeNumericJson(admitted),
  };
}

function numericClose(a: number, b: number, tolerance: number): boolean {
  return Math.abs(a - b) <= tolerance;
}

function equivalent(a: JsonValue, b: JsonValue, tolerance: number): boolean {
  if (typeof a === 'number' && typeof b === 'number') return numericClose(a, b, tolerance);
  if (a === null || b === null) return a === b;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((value, index) => equivalent(value, b[index], tolerance));
  }
  if (typeof a === 'object' && typeof b === 'object') {
    const left = a as Record<string, JsonValue>;
    const right = b as Record<string, JsonValue>;
    const keys = Object.keys(left).filter((key) => key in right).sort();
    if (keys.length === 0) return false;
    return keys.every((key) => equivalent(left[key], right[key], tolerance));
  }
  return a === b;
}

export function deterministicOutputsEqual(report: NormalizedI2Report): boolean | null {
  if (report.exactReconciliation !== null) return report.exactReconciliation;
  if (report.recomputedOutputs === null || report.admittedOutputs === null) return null;
  return equivalent(report.recomputedOutputs, report.admittedOutputs, 0);
}

export function deterministicOutputsWithinTolerance(report: NormalizedI2Report): boolean | null {
  if (report.status === 'FAIL') return false;
  if (report.exactReconciliation === true) return true;
  if (report.recomputedOutputs !== null && report.admittedOutputs !== null) {
    return equivalent(report.recomputedOutputs, report.admittedOutputs, report.tolerance ?? 0);
  }
  return report.status === 'PASS' ? true : null;
}
