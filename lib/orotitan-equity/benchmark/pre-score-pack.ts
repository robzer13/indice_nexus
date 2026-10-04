import { createHash } from 'node:crypto';

export const PRE_SCORE_SANITIZER_VERSION = '1.0.0';

const FORBIDDEN_EXACT = new Set([
  'scores',
  'scoring_summary',
  'terminal',
  'terminal_summary',
  'valuation',
  'valuation_lock',
  'certification',
  'quality_class',
  'weak_link_cap',
  'investment_raw',
  'investment_score',
  'orotitan_status',
  'score_permission',
]);

function forbiddenKey(key: string): boolean {
  const normalized = key.toLowerCase();
  if (FORBIDDEN_EXACT.has(normalized)) return true;
  if (normalized === 'score' || normalized === 'score_input' || normalized === 'provisional_score') return true;
  if (normalized.endsWith('_score')) return true;
  if (normalized.startsWith('oqs')) return true;
  if (normalized === 'ovs' || normalized.startsWith('ovs_')) return true;
  if (normalized.includes('score_precision_rule')) return true;
  return false;
}

export function sanitizePreScoreInput(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sanitizePreScoreInput);
  if (value === null || typeof value !== 'object') return value;

  const output: Record<string, unknown> = {};
  for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
    if (forbiddenKey(key)) continue;
    output[key] = sanitizePreScoreInput(nested);
  }
  return output;
}

export function stableJson(value: unknown): string {
  if (Array.isArray(value)) return '[' + value.map(stableJson).join(',') + ']';
  if (value !== null && typeof value === 'object') {
    return '{' + Object.keys(value as Record<string, unknown>)
      .sort()
      .map((key) => JSON.stringify(key) + ':' + stableJson((value as Record<string, unknown>)[key]))
      .join(',') + '}';
  }
  return JSON.stringify(value);
}

export function sha256Json(value: unknown): string {
  return createHash('sha256').update(stableJson(value), 'utf8').digest('hex');
}

export function findForbiddenScoringKeys(value: unknown, path = '$'): string[] {
  if (Array.isArray(value)) {
    return value.flatMap((nested, index) => findForbiddenScoringKeys(nested, path + '[' + index + ']'));
  }
  if (value === null || typeof value !== 'object') return [];

  const issues: string[] = [];
  for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
    const nextPath = path + '.' + key;
    if (forbiddenKey(key)) issues.push(nextPath);
    issues.push(...findForbiddenScoringKeys(nested, nextPath));
  }
  return issues;
}
