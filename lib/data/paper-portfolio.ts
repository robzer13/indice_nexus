import 'server-only';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import { getCompanyStates } from '@/lib/data/companies';
import { getFreshness } from '@/lib/domain/freshness';
import type { CompanyState } from '@/lib/domain/types';
import type { PaperInvestmentProfile, PaperSignalAction, PaperThesisState, PaperMilestoneState } from '@/lib/domain/paper-signals';

export type PaperPortfolioRow = {
  id: string; code: string; initial_cash_eur: number | string; cash_eur: number | string; created_at: string;
};
export type PaperPolicyRow = {
  id: string; portfolio_id: string; company_id: string; snapshot_id: string; version: number;
  profile: PaperInvestmentProfile; thesis_state: PaperThesisState; milestone_state: PaperMilestoneState;
  entry_price_eur: number | string | null; exit_target_eur: number | string | null;
  rationale: string; created_at: string;
};
export type PaperSignalRow = {
  id: string; company_id: string; policy_id: string; snapshot_id: string; quote_id: number;
  quote_price_eur: number | string; quote_as_of: string; ruleset: string;
  action: PaperSignalAction; reason: string; created_at: string;
};
export type PaperPositionRow = {
  company_id: string; quantity: number | string; avg_cost_eur: number | string;
  realized_pnl_eur: number | string;
};
export type PaperFillRow = {
  id: string; company_id: string; signal_id: string; quote_id: number; side: 'BUY' | 'SELL';
  quantity: number | string; price_eur: number | string; fee_eur: number | string;
  realized_pnl_eur: number | string; filled_at: string;
};
export type PaperQuote = { id: number; company_id: string; price: number | string; as_of: string; created_at: string; source: string; raw: unknown };
export type PaperDashboard = {
  migrationMissing: boolean;
  portfolio: PaperPortfolioRow | null;
  companies: CompanyState[];
  policies: PaperPolicyRow[];
  positions: PaperPositionRow[];
  signals: PaperSignalRow[];
  fills: PaperFillRow[];
  quotes: Map<string, PaperQuote>;
  equityEur: number | null;
  unrealizedPnlEur: number | null;
  markedPositions: number;
  unpricedPositions: number;
};

function isMissingRelation(code: string | undefined) {
  return code === '42P01' || code === 'PGRST205';
}
export function eurNumber(value: unknown): number {
  const n = Number(value);
  if (!Number.isFinite(n)) throw new Error('Montant virtuel invalide.');
  return n;
}
export async function latestStoredQuote(issuerId: string): Promise<PaperQuote | null> {
  // Canonical issuer IDs differ from historical companies IDs used by market_prices.
  // Resolve only through the audited identity crosswalk; no implicit UUID equality.
  const supabase = createServerSupabaseClient();
  const { data: link, error: linkError } = await supabase.from('legacy_company_identity_map')
    .select('legacy_company_id').eq('issuer_id', issuerId).maybeSingle();
  if (linkError) throw new Error('Résolution des identités de cours impossible : ' + linkError.message);
  if (!link) return null;
  const { data, error } = await supabase.from('market_prices')
    .select('id,company_id,price,as_of,created_at,source,raw').eq('company_id', link.legacy_company_id)
    .order('as_of', { ascending: false }).order('id', { ascending: false }).limit(1).maybeSingle();
  if (error) throw new Error('Lecture du cours impossible : ' + error.message);
  return data as PaperQuote | null;
}

export async function loadPaperDashboard(): Promise<PaperDashboard> {
  const supabase = createServerSupabaseClient();
  const base = {
    migrationMissing: false, portfolio: null, companies: [], policies: [], positions: [],
    signals: [], fills: [], quotes: new Map<string, PaperQuote>(),
    equityEur: null, unrealizedPnlEur: null, markedPositions: 0, unpricedPositions: 0,
  } satisfies PaperDashboard;
  const { data: portfolio, error } = await supabase.from('paper_portfolios')
    .select('*').eq('code', 'default').maybeSingle();
  if (error) {
    if (isMissingRelation(error.code)) return { ...base, migrationMissing: true };
    throw new Error('Lecture portefeuille impossible : ' + error.message);
  }
  const companies = await getCompanyStates();
  if (!portfolio) return { ...base, companies };
  const [pr, ps, ss, ff] = await Promise.all([
    supabase.from('paper_policy_versions').select('*').eq('portfolio_id', portfolio.id)
      .order('version', { ascending: false }),
    supabase.from('paper_positions').select('*').eq('portfolio_id', portfolio.id).gt('quantity', 0),
    supabase.from('paper_signals').select('*').eq('portfolio_id', portfolio.id)
      .order('created_at', { ascending: false }).limit(70),
    supabase.from('paper_fills').select('*').eq('portfolio_id', portfolio.id)
      .order('filled_at', { ascending: false }).limit(70),
  ]);
  for (const result of [pr, ps, ss, ff]) {
    if (result.error) throw new Error('Lecture portefeuille impossible : ' + result.error.message);
  }
  const allPolicies = (pr.data ?? []) as PaperPolicyRow[];
  const seen = new Set<string>();
  const policies = allPolicies.filter((policy) => {
    if (seen.has(policy.company_id)) return false;
    seen.add(policy.company_id);
    return true;
  });
  const positions = (ps.data ?? []) as PaperPositionRow[];
  const companyIds = [...new Set([...policies.map(p => p.company_id), ...positions.map(p => p.company_id)])];
  const pairs = await Promise.all(companyIds.map(async id => [id, await latestStoredQuote(id)] as const));
  const quotes = new Map<string, PaperQuote>();
  for (const [id, quote] of pairs) if (quote) quotes.set(id, quote);

  let equity = eurNumber(portfolio.cash_eur);
  let unrealized = 0;
  let markedPositions = 0;
  let unpricedPositions = 0;
  const companyById = new Map(companies.map(c => [c.id, c]));
  for (const pos of positions) {
    const quote = quotes.get(pos.company_id);
    const company = companyById.get(pos.company_id);
    if (!quote || !company || company.currency !== 'EUR' || company.quote_unit !== 'MAJOR'
      || getFreshness(quote.as_of).stale) {
      unpricedPositions++;
      continue;
    }
    equity += eurNumber(quote.price) * eurNumber(pos.quantity);
    unrealized += (eurNumber(quote.price) - eurNumber(pos.avg_cost_eur)) * eurNumber(pos.quantity);
    markedPositions++;
  }
  return {
    migrationMissing: false, portfolio: portfolio as PaperPortfolioRow, companies, policies, positions,
    signals: (ss.data ?? []) as PaperSignalRow[], fills: (ff.data ?? []) as PaperFillRow[],
    quotes, equityEur: unpricedPositions ? null : equity,
    unrealizedPnlEur: unpricedPositions ? null : unrealized,
    markedPositions, unpricedPositions,
  };
}
