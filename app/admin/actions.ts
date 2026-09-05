'use server';

import { redirect } from 'next/navigation';
import { revalidatePath } from 'next/cache';
import {
  createAdminSession,
  clearAdminSession,
  requireAdminSession,
  verifyAdminPassword,
} from '@/lib/auth/admin-session';
import { createCompany, updateCompanyMetadata } from '@/lib/data/company-admin';
import { createImmutableSnapshot } from '@/lib/data/snapshots';
import { companyInputSchema } from '@/lib/domain/company';
import { DuplicateSnapshotError, snapshotInputSchema } from '@/lib/domain/snapshot';
import { refreshMarketPrices } from '@/lib/market/refresh-prices';

function textOrNull(formData: FormData, key: string): string | null {
  const value = formData.get(key);
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function numberOrNull(formData: FormData, key: string): number | null {
  const value = textOrNull(formData, key);
  if (value === null) return null;
  const parsed = Number(value.replace(',', '.'));
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function booleanOrNull(formData: FormData, key: string): boolean | null {
  const value = textOrNull(formData, key);
  if (value === null) return null;
  if (value === 'true') return true;
  if (value === 'false') return false;
  return null;
}

function errorUrl(path: string, message: string): string {
  return `${path}?error=${encodeURIComponent(message)}`;
}

async function requireAdmin(path = '/admin'): Promise<void> {
  try {
    await requireAdminSession();
  } catch {
    redirect(errorUrl(path, 'Session admin invalide ou expirée.'));
  }
}

function snapshotCandidateFromForm(formData: FormData) {
  let scoreComponents: Record<string, unknown> = {};
  const rawComponents = textOrNull(formData, 'score_components');
  if (rawComponents) {
    const parsed: unknown = JSON.parse(rawComponents);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('score_components doit être un objet JSON.');
    }
    scoreComponents = parsed as Record<string, unknown>;
  }

  return {
    company_id: textOrNull(formData, 'company_id'),
    analysis_date: textOrNull(formData, 'analysis_date'),
    model_version: textOrNull(formData, 'model_version'),
    status: textOrNull(formData, 'status'),
    quality_orotitan: booleanOrNull(formData, 'quality_orotitan'),
    business_quality_score: numberOrNull(formData, 'business_quality_score'),
    investment_score: numberOrNull(formData, 'investment_score'),
    valuation_score: numberOrNull(formData, 'valuation_score'),
    orotitan_score: numberOrNull(formData, 'orotitan_score'),
    confidence_score: numberOrNull(formData, 'confidence_score'),
    fair_value_low: numberOrNull(formData, 'fair_value_low'),
    fair_value_base: numberOrNull(formData, 'fair_value_base'),
    fair_value_high: numberOrNull(formData, 'fair_value_high'),
    price_o85: numberOrNull(formData, 'price_o85'),
    price_o90: numberOrNull(formData, 'price_o90'),
    price_o92: numberOrNull(formData, 'price_o92'),
    price_o95: numberOrNull(formData, 'price_o95'),
    thesis: textOrNull(formData, 'thesis'),
    main_risk: textOrNull(formData, 'main_risk'),
    invalidation: textOrNull(formData, 'invalidation'),
    source_title: textOrNull(formData, 'source_title'),
    notes: textOrNull(formData, 'notes'),
    score_components: scoreComponents,
  };
}

function companyCandidateFromForm(formData: FormData) {
  return {
    slug: textOrNull(formData, 'slug'),
    ticker: textOrNull(formData, 'ticker'),
    name: textOrNull(formData, 'name'),
    exchange: textOrNull(formData, 'exchange'),
    currency: textOrNull(formData, 'currency'),
    quote_unit: textOrNull(formData, 'quote_unit'),
    price_decimals: numberOrNull(formData, 'price_decimals'),
    market_data_symbol: textOrNull(formData, 'market_data_symbol'),
    market_data_multiplier: numberOrNull(formData, 'market_data_multiplier'),
    country: textOrNull(formData, 'country'),
    sector: textOrNull(formData, 'sector'),
    active: textOrNull(formData, 'active') === 'true',
  };
}

function issuesMessage(error: { issues: Array<{ path: PropertyKey[]; message: string }> }): string {
  return error.issues.map((issue) => `${issue.path.join('.')}: ${issue.message}`).join(' · ');
}

async function insertValidatedSnapshot(candidate: unknown, errorPath: string): Promise<never> {
  const parsed = snapshotInputSchema.safeParse(candidate);
  if (!parsed.success) redirect(errorUrl(errorPath, issuesMessage(parsed.error)));

  try {
    await createImmutableSnapshot(parsed.data);
  } catch (error) {
    if (error instanceof DuplicateSnapshotError) redirect(errorUrl(errorPath, error.message));
    redirect(errorUrl(errorPath, error instanceof Error ? error.message : 'Impossible de créer le snapshot.'));
  }

  revalidatePath('/');
  revalidatePath('/screener');
  revalidatePath('/company', 'layout');
  redirect(`${errorPath}?success=${encodeURIComponent('Snapshot créé sans modifier l’historique.')}`);
}

export async function loginAction(formData: FormData): Promise<void> {
  const password = formData.get('password');
  if (typeof password !== 'string' || !verifyAdminPassword(password)) {
    redirect(errorUrl('/admin', 'Mot de passe incorrect.'));
  }
  await createAdminSession();
  redirect('/admin');
}

export async function logoutAction(): Promise<void> {
  await clearAdminSession();
  redirect('/admin');
}

export async function createCompanyAction(formData: FormData): Promise<void> {
  await requireAdmin('/admin/companies/new');
  const parsed = companyInputSchema.safeParse(companyCandidateFromForm(formData));
  if (!parsed.success) redirect(errorUrl('/admin/companies/new', issuesMessage(parsed.error)));

  try {
    await createCompany(parsed.data);
  } catch (error) {
    redirect(errorUrl('/admin/companies/new', error instanceof Error ? error.message : 'Création impossible.'));
  }

  revalidatePath('/');
  revalidatePath('/screener');
  revalidatePath('/admin/companies');
  redirect('/admin/companies?success=Société%20créée.');
}

export async function updateCompanyAction(formData: FormData): Promise<void> {
  const slug = textOrNull(formData, 'current_slug') ?? '';
  const path = `/admin/companies/${slug}`;
  await requireAdmin(path);
  const companyId = textOrNull(formData, 'company_id');
  if (!companyId) redirect(errorUrl(path, 'company_id manquant.'));

  const parsed = companyInputSchema.safeParse(companyCandidateFromForm(formData));
  if (!parsed.success) redirect(errorUrl(path, issuesMessage(parsed.error)));

  try {
    await updateCompanyMetadata(companyId, parsed.data);
  } catch (error) {
    redirect(errorUrl(path, error instanceof Error ? error.message : 'Mise à jour impossible.'));
  }

  revalidatePath('/');
  revalidatePath('/screener');
  revalidatePath('/company', 'layout');
  revalidatePath('/admin/companies');
  redirect(`/admin/companies/${parsed.data.slug}?success=${encodeURIComponent('Métadonnées mises à jour.')}`);
}

export async function createSnapshotAction(formData: FormData): Promise<void> {
  const path = '/admin/snapshots/new';
  await requireAdmin(path);
  let candidate: unknown;
  try {
    candidate = snapshotCandidateFromForm(formData);
  } catch (error) {
    redirect(errorUrl(path, error instanceof Error ? error.message : 'Données de snapshot invalides.'));
  }
  return insertValidatedSnapshot(candidate, path);
}

export async function createSnapshotJsonAction(formData: FormData): Promise<void> {
  const path = '/admin/snapshots/new';
  await requireAdmin(path);
  const companyId = textOrNull(formData, 'json_company_id');
  const raw = textOrNull(formData, 'snapshot_json');
  if (!companyId || !raw) redirect(errorUrl(path, 'Sélectionnez une société et collez un JSON.'));

  let payload: Record<string, unknown>;
  try {
    const decoded: unknown = JSON.parse(raw);
    if (!decoded || typeof decoded !== 'object' || Array.isArray(decoded)) throw new Error('Le JSON doit être un objet.');
    payload = decoded as Record<string, unknown>;
  } catch (error) {
    redirect(errorUrl(path, error instanceof Error ? error.message : 'JSON invalide.'));
  }

  const candidate = {
    company_id: companyId,
    analysis_date: payload.analysis_date ?? null,
    model_version: payload.model_version ?? null,
    status: payload.status ?? null,
    quality_orotitan: payload.quality_orotitan ?? null,
    business_quality_score: payload.business_quality_score ?? null,
    investment_score: payload.investment_score ?? null,
    valuation_score: payload.valuation_score ?? null,
    orotitan_score: payload.orotitan_score ?? null,
    confidence_score: payload.confidence_score ?? null,
    fair_value_low: payload.fair_value_low ?? null,
    fair_value_base: payload.fair_value_base ?? null,
    fair_value_high: payload.fair_value_high ?? null,
    price_o85: payload.price_o85 ?? null,
    price_o90: payload.price_o90 ?? null,
    price_o92: payload.price_o92 ?? null,
    price_o95: payload.price_o95 ?? null,
    thesis: payload.thesis ?? null,
    main_risk: payload.main_risk ?? null,
    invalidation: payload.invalidation ?? null,
    source_title: payload.source_title ?? null,
    notes: payload.notes ?? null,
    score_components: payload.score_components ?? {},
  };

  return insertValidatedSnapshot(candidate, path);
}

export async function refreshPricesAction(): Promise<void> {
  const path = '/admin/prices';
  await requireAdmin(path);

  let result;
  try {
    result = await refreshMarketPrices('ADMIN');
  } catch (error) {
    redirect(errorUrl(path, error instanceof Error ? error.message : 'Rafraîchissement impossible.'));
  }

  revalidatePath('/');
  revalidatePath('/screener');
  revalidatePath('/company', 'layout');
  redirect(`${path}?success=${encodeURIComponent(`${result.inserted} cours inséré(s), ${result.failed} échec(s).`)}`);
}
