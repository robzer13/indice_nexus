'use server';

import { redirect } from 'next/navigation';
import { revalidatePath } from 'next/cache';
import { requireAdminSession } from '@/lib/auth/admin-session';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import { getCompanyStates } from '@/lib/data/companies';
import { eurNumber, latestStoredQuote, loadPaperDashboard } from '@/lib/data/paper-portfolio';
import { evaluateLongTermPaperSignal } from '@/lib/domain/paper-signals';
import type { PaperInvestmentProfile, PaperThesisState, PaperMilestoneState } from '@/lib/domain/paper-signals';

const PATH = '/admin/paper';

function returnToPage(kind: 'success' | 'error', message: string): never {
  redirect(PATH + '?' + kind + '=' + encodeURIComponent(message));
}

async function adminOnly() {
  try { await requireAdminSession(); }
  catch { redirect('/admin?error=' + encodeURIComponent('Session admin requise.')); }
}
function field(form: FormData, name: string): string {
  const v = form.get(name);
  return typeof v === 'string' ? v.trim() : '';
}
function positiveOrNull(raw: string): number | null {
  if (!raw) return null;
  const n = Number(raw.replace(',', '.'));
  if (!Number.isFinite(n) || n <= 0 || n > 1000000) throw new Error('Prix positif requis, inférieur à 1 000 000 €.');
  return n;
}
function enums<T extends string>(candidate: string, values: readonly T[]): T {
  if (!values.includes(candidate as T)) throw new Error('Valeur de stratégie non autorisée.');
  return candidate as T;
}

export async function createPaperPortfolioAction(): Promise<void> {
  await adminOnly();
  try {
    const { error } = await createServerSupabaseClient().from('paper_portfolios').insert({ code: 'default' });
    if (error && error.code !== '23505') throw new Error(error.message);
  } catch (e) {
    returnToPage('error', e instanceof Error ? e.message : 'Initialisation impossible.');
  }
  revalidatePath(PATH);
  returnToPage('success', 'Portefeuille fictif initialisé avec 100 000 €. Aucune transaction réelle.');
}

export async function createPaperPolicyAction(form: FormData): Promise<void> {
  await adminOnly();
  try {
    const view = await loadPaperDashboard();
    if (!view.portfolio || view.migrationMissing) throw new Error('Portefeuille ou migration indisponible.');
    const companyId = field(form, 'company_id');
    const company = view.companies.find(c => c.id === companyId);
    if (!company) throw new Error('Société sans snapshot canonique courant.');
    if (company.currency !== 'EUR' || company.quote_unit !== 'MAJOR')
      throw new Error('V1 : seules les actions cotées directement en EUR majeur sont prises en charge.');
    const profile = enums<PaperInvestmentProfile>(field(form, 'profile'), ['CORE','VALUE','CYCLICAL_VALUE','TURNAROUND']);
    const thesis = enums<PaperThesisState>(field(form, 'thesis_state'), ['ACTIVE','INVALIDATED','UNKNOWN']);
    const milestone = enums<PaperMilestoneState>(field(form, 'milestone_state'), ['CONFIRMED','UNCONFIRMED','NOT_APPLICABLE']);
    const rationale = field(form, 'rationale');
    if (rationale.length < 12 || rationale.length > 2000) throw new Error('Justification requise (12 à 2000 caractères).');
    const entry = positiveOrNull(field(form, 'entry_price_eur'));
    const target = positiveOrNull(field(form, 'exit_target_eur'));
    if (profile !== 'CORE' && target === null) throw new Error('Les stratégies VALUE exigent une cible de vente justifiée.');
    const { data: dossier, error: dossierError } = await createServerSupabaseClient()
      .from('research_dossiers').select('current_snapshot_id').eq('issuer_id', companyId)
      .eq('active', true).not('current_snapshot_id','is',null).limit(1).maybeSingle();
    if (dossierError || !dossier?.current_snapshot_id)
      throw new Error('Snapshot canonique courant introuvable.');
    const { error } = await createServerSupabaseClient().rpc('paper_create_policy', {
      p_portfolio: view.portfolio.id,
      p_company: companyId,
      p_snapshot: dossier.current_snapshot_id,
      p_profile: profile,
      p_thesis: thesis,
      p_milestone: milestone,
      p_entry: entry,
      p_exit: target,
      p_rationale: rationale,
    });
    if (error) throw new Error(error.message);
  } catch (e) {
    returnToPage('error', e instanceof Error ? e.message : 'Politique non enregistrée.');
  }
  revalidatePath(PATH);
  returnToPage('success', 'Nouvelle version de stratégie enregistrée et historisée.');
}

export async function evaluatePaperSignalAction(form: FormData): Promise<void> {
  await adminOnly();
  try {
    const view = await loadPaperDashboard();
    if (!view.portfolio) throw new Error('Portefeuille non initialisé.');
    const policy = view.policies.find(p => p.id === field(form, 'policy_id'));
    if (!policy) throw new Error('Stratégie active introuvable.');
    const company = view.companies.find(c => c.id === policy.company_id);
    if (!company) throw new Error('Analyse courante indisponible.');
    if (company.currency !== 'EUR' || company.quote_unit !== 'MAJOR')
      throw new Error('Unité de cotation ou devise non prise en charge.');
    const { data: current, error: currentError } = await createServerSupabaseClient()
      .from('research_dossiers').select('current_snapshot_id')
      .eq('issuer_id',policy.company_id).eq('active',true).eq('current_snapshot_id',policy.snapshot_id)
      .limit(1).maybeSingle();
    if (currentError || !current) throw new Error('Une analyse plus récente impose une révision de stratégie.');
    const quote = await latestStoredQuote(policy.company_id);
    if (!quote) throw new Error('Aucun cours de marché effectivement enregistré.');
    const position = view.positions.find(p => p.company_id === policy.company_id);
    const signal = evaluateLongTermPaperSignal({
      strategy: {
        profile: policy.profile,
        thesisState: policy.thesis_state,
        milestoneState: policy.milestone_state,
        entryPrice: policy.entry_price_eur === null ? null : eurNumber(policy.entry_price_eur),
        exitTargetPrice: policy.exit_target_eur === null ? null : eurNumber(policy.exit_target_eur),
      },
      snapshot: {
        id: policy.snapshot_id,
        analysisDate: company.analysis_date,
        canonicalEntryPrice: company.price_o90,
        eligibleForInvestment: company.status === 'OROTITAN' || company.status === 'FINALIST' || company.status === 'PRICE_WAIT',
        coreQualityConfirmed: company.quality_orotitan === true,
      },
      quote: { price: eurNumber(quote.price), asOf: quote.as_of },
      positionQuantity: position ? eurNumber(position.quantity) : 0,
      now: new Date().toISOString(),
    });
    if (signal.action === 'DATA_BLOCKED') throw new Error('Signal bloqué : '+signal.reason);
    const { error } = await createServerSupabaseClient().rpc('paper_record_signal', {
      p_policy: policy.id, p_quote: quote.id, p_ruleset: signal.ruleset,
      p_action: signal.action, p_reason: signal.reason,
    });
    if (error) throw new Error(error.message);
  } catch (e) {
    returnToPage('error', e instanceof Error ? e.message : 'Signal non émis.');
  }
  revalidatePath(PATH);
  returnToPage('success', 'Signal simulé enregistré avec sa cotation et sa politique.');
}

export async function executeVirtualFillAction(form: FormData): Promise<void> {
  await adminOnly();
  try {
    const view = await loadPaperDashboard();
    const signal = view.signals.find(s => s.id === field(form, 'signal_id'));
    if (!signal || !['BUY_CANDIDATE','SELL_CANDIDATE'].includes(signal.action))
      throw new Error('Signal inexistant ou non exécutable.');
    const qty = positiveOrNull(field(form, 'quantity'));
    if (qty === null || qty > 100000) throw new Error('Quantité virtuelle invalide.');
    const feeRaw = field(form, 'fee_eur') || '0';
    const fee = Number(feeRaw.replace(',', '.'));
    if (!Number.isFinite(fee) || fee < 0 || fee > 1000000) throw new Error('Frais virtuels invalides.');
    const quote = await latestStoredQuote(signal.company_id);
    if (!quote) throw new Error('Pas de nouvelle cotation disponible.');
    const { error } = await createServerSupabaseClient().rpc('paper_execute_fill', {
      p_signal: signal.id, p_quote: quote.id, p_quantity: qty, p_fee_eur: fee,
    });
    if (error) throw new Error(error.message);
  } catch (e) {
    returnToPage('error', e instanceof Error ? e.message : 'Simulation non exécutée.');
  }
  revalidatePath(PATH);
  returnToPage('success', 'Transaction fictive enregistrée au cours observé après le signal.');
}
