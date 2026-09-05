import { getFreshness } from '@/lib/domain/freshness';
import type { CompanyState } from '@/lib/domain/types';

export const ANALYSIS_STALE_DAYS = 180;

export type DataHealthSeverity = 'error' | 'warning' | 'info';

export interface DataHealthIssue {
  code:
    | 'MISSING_PRICE'
    | 'STALE_PRICE'
    | 'MISSING_O90'
    | 'MISSING_MARKET_SYMBOL'
    | 'INVALID_MULTIPLIER'
    | 'MISSING_ANALYSIS'
    | 'STALE_ANALYSIS'
    | 'MISSING_COUNTRY'
    | 'MISSING_SECTOR';
  severity: DataHealthSeverity;
  label: string;
}

function analysisAgeDays(date: string | null, now: Date): number | null {
  if (!date) return null;
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return null;
  return Math.max(0, (now.getTime() - parsed.getTime()) / 86_400_000);
}

export function getCompanyDataHealth(company: CompanyState, now = new Date()): DataHealthIssue[] {
  const issues: DataHealthIssue[] = [];
  const freshness = getFreshness(company.price_as_of, now);
  const ageDays = analysisAgeDays(company.analysis_date, now);

  if (company.price === null) {
    issues.push({ code: 'MISSING_PRICE', severity: 'error', label: 'Cours manquant' });
  } else if (freshness.stale) {
    issues.push({ code: 'STALE_PRICE', severity: 'warning', label: 'Cours périmé' });
  }

  if (company.price_o90 === null) {
    issues.push({ code: 'MISSING_O90', severity: 'warning', label: 'O90 non calibré' });
  }

  if (!company.market_data_symbol) {
    issues.push({ code: 'MISSING_MARKET_SYMBOL', severity: 'warning', label: 'Symbole marché manquant' });
  }

  if (!Number.isFinite(company.market_data_multiplier) || company.market_data_multiplier <= 0) {
    issues.push({ code: 'INVALID_MULTIPLIER', severity: 'error', label: 'Multiplicateur marché invalide' });
  }

  if (!company.analysis_date) {
    issues.push({ code: 'MISSING_ANALYSIS', severity: 'error', label: 'Analyse manquante' });
  } else if (ageDays !== null && ageDays > ANALYSIS_STALE_DAYS) {
    issues.push({ code: 'STALE_ANALYSIS', severity: 'warning', label: 'Analyse à actualiser' });
  }

  if (!company.country) issues.push({ code: 'MISSING_COUNTRY', severity: 'info', label: 'Pays non renseigné' });
  if (!company.sector) issues.push({ code: 'MISSING_SECTOR', severity: 'info', label: 'Secteur non renseigné' });

  return issues;
}

export function summarizeDataHealth(companies: CompanyState[], now = new Date()) {
  const rows = companies.map((company) => ({ company, issues: getCompanyDataHealth(company, now) }));
  return {
    rows,
    clean: rows.filter((row) => row.issues.length === 0).length,
    withErrors: rows.filter((row) => row.issues.some((issue) => issue.severity === 'error')).length,
    withWarnings: rows.filter((row) => row.issues.some((issue) => issue.severity === 'warning')).length,
    totalIssues: rows.reduce((sum, row) => sum + row.issues.length, 0),
  };
}
