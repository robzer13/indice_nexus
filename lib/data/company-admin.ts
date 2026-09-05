import 'server-only';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import type { CompanyInput } from '@/lib/domain/company';
import type { CompanyRecord } from '@/lib/domain/types';

export async function getAllCompanies(): Promise<CompanyRecord[]> {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase
    .from('companies')
    .select('*')
    .order('active', { ascending: false })
    .order('name', { ascending: true });
  if (error) throw new Error(`Unable to load all companies: ${error.message}`);
  return (data ?? []) as CompanyRecord[];
}

export async function getCompanyRecordBySlug(slug: string): Promise<CompanyRecord | null> {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase
    .from('companies')
    .select('*')
    .eq('slug', slug)
    .maybeSingle();
  if (error) throw new Error(`Unable to load company metadata: ${error.message}`);
  return data ? (data as CompanyRecord) : null;
}

export async function createCompany(input: CompanyInput): Promise<void> {
  const supabase = createServerSupabaseClient();
  const { error } = await supabase.from('companies').insert(input);
  if (error?.code === '23505') throw new Error('Une société avec ce slug existe déjà.');
  if (error) throw new Error(`Unable to create company: ${error.message}`);
}

export async function updateCompanyMetadata(companyId: string, input: CompanyInput): Promise<void> {
  const supabase = createServerSupabaseClient();
  const { error } = await supabase
    .from('companies')
    .update(input)
    .eq('id', companyId);
  if (error?.code === '23505') throw new Error('Ce slug est déjà utilisé par une autre société.');
  if (error) throw new Error(`Unable to update company: ${error.message}`);
}
