import 'server-only';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import { getDistanceO90 } from '@/lib/domain/distance';
import type { ActiveCompanyOption, CompanyState, SnapshotHistoryRow } from '@/lib/domain/types';

const COMPANY_STATE_COLUMNS = '*';

function asCompanyStates(data: unknown): CompanyState[] {
  return (data ?? []) as CompanyState[];
}

export async function getCompanyStates(): Promise<CompanyState[]> {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase
    .from('latest_company_state')
    .select(COMPANY_STATE_COLUMNS)
    .order('name', { ascending: true });
  if (error) throw new Error(`Unable to load company states: ${error.message}`);
  return asCompanyStates(data);
}

export async function getCompanyStateBySlug(slug: string): Promise<CompanyState | null> {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase
    .from('latest_company_state')
    .select(COMPANY_STATE_COLUMNS)
    .eq('slug', slug)
    .maybeSingle();
  if (error) throw new Error(`Unable to load company: ${error.message}`);
  return data ? (data as CompanyState) : null;
}

export async function getActiveCompanies(): Promise<ActiveCompanyOption[]> {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase
    .from('companies')
    .select('id,slug,ticker,name,exchange')
    .eq('active', true)
    .order('name', { ascending: true });
  if (error) throw new Error(`Unable to load companies: ${error.message}`);
  return (data ?? []) as ActiveCompanyOption[];
}

export async function getSnapshotHistory(companyId: string): Promise<SnapshotHistoryRow[]> {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase
    .from('snapshots')
    .select('*')
    .eq('company_id', companyId)
    .order('analysis_date', { ascending: false })
    .order('created_at', { ascending: false });
  if (error) throw new Error(`Unable to load snapshot history: ${error.message}`);
  return (data ?? []) as SnapshotHistoryRow[];
}

export function withDistance(company: CompanyState) {
  return { ...company, distance_o90_pct: getDistanceO90(company.price, company.price_o90) };
}
