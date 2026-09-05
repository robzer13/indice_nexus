'use server';

import { redirect } from 'next/navigation';
import { revalidatePath } from 'next/cache';
import { createAdminSession, clearAdminSession, requireAdminSession, verifyAdminPassword } from '@/lib/auth/admin-session';
import { createImmutableSnapshot } from '@/lib/data/snapshots';
import { DuplicateSnapshotError, snapshotInputSchema } from '@/lib/domain/snapshot';

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

function errorRedirect(message: string): never {
  redirect(`/admin?error=${encodeURIComponent(message)}`);
}

export async function loginAction(formData: FormData): Promise<void> {
  const password = formData.get('password');
  if (typeof password !== 'string' || !verifyAdminPassword(password)) {
    errorRedirect('Mot de passe incorrect.');
  }
  await createAdminSession();
  redirect('/admin');
}

export async function logoutAction(): Promise<void> {
  await clearAdminSession();
  redirect('/admin');
}

export async function createSnapshotAction(formData: FormData): Promise<void> {
  try {
    await requireAdminSession();
  } catch {
    redirect('/admin?error=Session%20admin%20invalide%20ou%20expir%C3%A9e.');
  }

  let scoreComponents: Record<string, unknown> = {};
  const rawComponents = textOrNull(formData, 'score_components');
  if (rawComponents) {
    try {
      const parsed: unknown = JSON.parse(rawComponents);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        errorRedirect('score_components doit être un objet JSON.');
      }
      scoreComponents = parsed as Record<string, unknown>;
    } catch {
      errorRedirect('score_components contient un JSON invalide.');
    }
  }

  const parsed = snapshotInputSchema.safeParse({
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
  });

  if (!parsed.success) {
    const message = parsed.error.issues.map((issue) => `${issue.path.join('.')}: ${issue.message}`).join(' · ');
    errorRedirect(message);
  }

  try {
    await createImmutableSnapshot(parsed.data);
  } catch (error) {
    if (error instanceof DuplicateSnapshotError) errorRedirect(error.message);
    errorRedirect('Impossible de créer le snapshot. Vérifiez les données et la connexion Supabase.');
  }

  revalidatePath('/');
  revalidatePath('/screener');
  revalidatePath('/company', 'layout');
  redirect('/admin?success=Snapshot%20créé%20sans%20modifier%20l’historique.');
}
