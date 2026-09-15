import 'server-only';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import { getDistanceO90 } from '@/lib/domain/distance';
import type { ActiveCompanyOption, CompanyState, CompanyStatus, Json, QuoteUnit, SnapshotHistoryRow } from '@/lib/domain/types';

type UnknownRecord = Record<string, unknown>;

type CanonicalSnapshotRow = {
  snapshot_id: string;
  dossier_id: string;
  issuer_id: string;
  security_id: string;
  report_id: string;
  calculation_date: string;
  report_version: string;
  method_version: string;
  canonical_payload: unknown;
  created_at: string;
};

type CanonicalContext = {
  issuer: UnknownRecord;
  security: UnknownRecord;
  company: UnknownRecord | null;
  marketPrice: UnknownRecord | null;
};

function asRecord(value: unknown): UnknownRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as UnknownRecord : {};
}

function asString(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function asNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function slugify(value: string): string {
  return value
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function mapStatus(orotitanStatus: string | null, nextAction: string | null, readiness: string | null): CompanyStatus {
  if (orotitanStatus === 'YES') return 'OROTITAN';
  if (nextAction === 'WAIT_FOR_PRICE') return 'PRICE_WAIT';
  if (nextAction === 'REJECT' || readiness === 'REJECTED') return 'REJECTED';
  if (readiness === 'READY') return 'FINALIST';
  return 'WATCHLIST';
}

function canonicalFields(row: CanonicalSnapshotRow) {
  const payload = asRecord(row.canonical_payload);
  const dataLock = asRecord(payload.data_lock);
  const l2 = asRecord(payload.l2_research_fundamentals);
  const businessQuality = asRecord(l2.business_quality);
  const l3 = asRecord(payload.l3_investment_valuation);
  const valuation = asRecord(l3.valuation);
  const investment = asRecord(l3.investment);
  const priceLadder = asRecord(valuation.price_ladder);
  const l4 = asRecord(payload.l4_operational_state);
  const orotitan = asRecord(l4.orotitan);

  const orotitanStatus = asString(orotitan.orotitan_status);
  const nextAction = asString(l4.next_action);
  const readiness = asString(l4.dossier_readiness);
  const invalidationTriggers = asStringArray(l4.thesis_invalidation_triggers);

  const scoreComponents: Json = {
    oqs: asNumber(businessQuality.oqs),
    ovs: asNumber(valuation.ovs),
    investmentScore: asNumber(investment.investment_score),
    moat: asNumber(businessQuality.moat_score),
    runway: asNumber(businessQuality.runway_score),
    returnQuality: asNumber(businessQuality.return_quality_score),
    cashEconomics: asNumber(businessQuality.cash_economics_score),
    capitalAllocation: asNumber(businessQuality.capital_allocation_score),
    managementGovernance: asNumber(businessQuality.management_governance_score),
    resilienceRisk: asNumber(businessQuality.resilience_risk_score),
  };

  return {
    payload,
    dataLock,
    orotitanStatus,
    nextAction,
    readiness,
    status: mapStatus(orotitanStatus, nextAction, readiness),
    qualityOroTitan: orotitanStatus === null ? null : orotitanStatus === 'YES',
    oqs: asNumber(businessQuality.oqs),
    ovs: asNumber(valuation.ovs),
    investmentScore: asNumber(investment.investment_score),
    requiredReturnPrice: asNumber(priceLadder.price_for_required_return_h) ?? asNumber(asRecord(priceLadder.investable_price_zone).max_price),
    strongReturnPrice: asNumber(priceLadder.price_for_strong_return) ?? asNumber(asRecord(priceLadder.strong_opportunity_zone).max_price),
    exceptionalReturnPrice: asNumber(priceLadder.price_for_exceptional_return),
    potentialOroTitanPrice: asNumber(priceLadder.potential_orotitan_max_price),
    referencePrice: asNumber(dataLock.reference_price),
    referencePriceDate: asString(dataLock.reference_price_date),
    invalidation: invalidationTriggers.length > 0 ? invalidationTriggers.join('\n') : null,
    scoreComponents,
  };
}

function chooseDisplayedPrice(snapshot: ReturnType<typeof canonicalFields>, marketPrice: UnknownRecord | null) {
  const marketValue = marketPrice ? asNumber(marketPrice.price) : null;
  const marketAsOf = marketPrice ? asString(marketPrice.as_of) : null;
  const marketSource = marketPrice ? asString(marketPrice.source) : null;
  const referenceDate = snapshot.referencePriceDate;

  if (marketValue !== null && marketAsOf !== null) {
    const marketTs = Date.parse(marketAsOf);
    const referenceTs = referenceDate ? Date.parse(`${referenceDate}T00:00:00Z`) : Number.NaN;
    if (!Number.isFinite(referenceTs) || marketTs >= referenceTs) {
      return { price: marketValue, priceAsOf: marketAsOf, priceSource: marketSource ?? 'MARKET_PRICE' };
    }
  }

  if (snapshot.referencePrice !== null && referenceDate !== null) {
    return {
      price: snapshot.referencePrice,
      priceAsOf: `${referenceDate}T00:00:00Z`,
      priceSource: 'CANONICAL_REFERENCE_PRICE',
    };
  }

  return { price: marketValue, priceAsOf: marketAsOf, priceSource: marketSource };
}

function mapSnapshotHistory(row: CanonicalSnapshotRow): SnapshotHistoryRow {
  const canonical = canonicalFields(row);
  return {
    id: row.snapshot_id,
    company_id: row.issuer_id,
    analysis_date: row.calculation_date,
    model_version: `OroTitan ${row.method_version} / report ${row.report_version}`,
    status: canonical.status,
    quality_orotitan: canonical.qualityOroTitan,
    business_quality_score: canonical.oqs,
    investment_score: canonical.investmentScore,
    valuation_score: canonical.ovs,
    orotitan_score: canonical.investmentScore,
    confidence_score: null,
    fair_value_low: null,
    fair_value_base: null,
    fair_value_high: null,
    price_o85: canonical.potentialOroTitanPrice,
    price_o90: canonical.requiredReturnPrice,
    price_o92: canonical.strongReturnPrice,
    price_o95: canonical.exceptionalReturnPrice,
    thesis: null,
    main_risk: null,
    invalidation: canonical.invalidation,
    source_title: `Canonical snapshot ${row.snapshot_id}`,
    notes: [canonical.readiness, canonical.nextAction].filter(Boolean).join(' · ') || null,
    score_components: canonical.scoreComponents,
    created_at: row.created_at,
  };
}

function mapCompanyState(row: CanonicalSnapshotRow, context: CanonicalContext): CompanyState {
  const canonical = canonicalFields(row);
  const issuerName = asString(context.issuer.display_name) ?? asString(context.issuer.legal_name) ?? row.issuer_id;
  const price = chooseDisplayedPrice(canonical, context.marketPrice);
  const companySlug = asString(context.company?.slug) ?? slugify(issuerName);
  const quoteUnit = (asString(context.security.quote_unit) === 'MINOR' ? 'MINOR' : 'MAJOR') as QuoteUnit;

  return {
    id: row.issuer_id,
    slug: companySlug,
    ticker: asString(context.security.ticker) ?? '—',
    name: issuerName,
    exchange: asString(context.security.exchange) ?? '—',
    currency: asString(context.security.trading_currency) ?? asString(context.issuer.reporting_currency) ?? 'USD',
    quote_unit: quoteUnit,
    price_decimals: asNumber(context.security.price_decimals) ?? 2,
    country: asString(context.security.country) ?? asString(context.issuer.country),
    sector: asString(context.company?.sector),
    market_data_symbol: asString(context.security.market_data_symbol),
    market_data_multiplier: asNumber(context.security.market_data_multiplier) ?? 1,
    price: price.price,
    price_as_of: price.priceAsOf,
    price_source: price.priceSource,
    analysis_date: row.calculation_date,
    model_version: `OroTitan ${row.method_version} / report ${row.report_version}`,
    status: canonical.status,
    quality_orotitan: canonical.qualityOroTitan,
    business_quality_score: canonical.oqs,
    investment_score: canonical.investmentScore,
    valuation_score: canonical.ovs,
    orotitan_score: canonical.investmentScore,
    confidence_score: null,
    fair_value_low: null,
    fair_value_base: null,
    fair_value_high: null,
    price_o85: canonical.potentialOroTitanPrice,
    price_o90: canonical.requiredReturnPrice,
    price_o92: canonical.strongReturnPrice,
    price_o95: canonical.exceptionalReturnPrice,
    thesis: null,
    main_risk: null,
    invalidation: canonical.invalidation,
    source_title: `Canonical snapshot ${row.snapshot_id}`,
    notes: [canonical.readiness, canonical.nextAction].filter(Boolean).join(' · ') || null,
    score_components: canonical.scoreComponents,
  };
}

async function loadPublishedCanonicalStates(): Promise<CompanyState[]> {
  const supabase = createServerSupabaseClient();
  const { data: dossiers, error: dossierError } = await supabase
    .from('research_dossiers')
    .select('dossier_id,issuer_id,current_snapshot_id')
    .eq('active', true)
    .not('current_snapshot_id', 'is', null);
  if (dossierError) throw new Error(`Unable to load canonical dossiers: ${dossierError.message}`);
  if (!dossiers || dossiers.length === 0) return [];

  const snapshotIds = dossiers.map((row) => row.current_snapshot_id).filter((value): value is string => typeof value === 'string');
  const issuerIds = dossiers.map((row) => row.issuer_id).filter((value): value is string => typeof value === 'string');

  const [snapshotsResult, issuersResult, companiesResult, pricesResult] = await Promise.all([
    supabase
      .from('research_snapshots')
      .select('snapshot_id,dossier_id,issuer_id,security_id,report_id,calculation_date,report_version,method_version,canonical_payload,created_at')
      .in('snapshot_id', snapshotIds),
    supabase.from('issuers').select('*').in('issuer_id', issuerIds),
    supabase.from('companies').select('id,slug,sector').in('id', issuerIds),
    supabase.from('market_prices').select('company_id,price,as_of,source,created_at').in('company_id', issuerIds).order('as_of', { ascending: false }),
  ]);

  if (snapshotsResult.error) throw new Error(`Unable to load canonical snapshots: ${snapshotsResult.error.message}`);
  if (issuersResult.error) throw new Error(`Unable to load canonical issuers: ${issuersResult.error.message}`);
  if (companiesResult.error) throw new Error(`Unable to load company display metadata: ${companiesResult.error.message}`);
  if (pricesResult.error) throw new Error(`Unable to load market prices: ${pricesResult.error.message}`);

  const snapshots = (snapshotsResult.data ?? []) as CanonicalSnapshotRow[];
  const securityIds = snapshots.map((row) => row.security_id);
  const { data: securities, error: securitiesError } = await supabase.from('securities').select('*').in('security_id', securityIds);
  if (securitiesError) throw new Error(`Unable to load canonical securities: ${securitiesError.message}`);

  const issuerById = new Map((issuersResult.data ?? []).map((row) => [row.issuer_id as string, row as UnknownRecord]));
  const securityById = new Map((securities ?? []).map((row) => [row.security_id as string, row as UnknownRecord]));
  const companyById = new Map((companiesResult.data ?? []).map((row) => [row.id as string, row as UnknownRecord]));
  const latestPriceByCompany = new Map<string, UnknownRecord>();
  for (const row of pricesResult.data ?? []) {
    const companyId = row.company_id as string;
    if (!latestPriceByCompany.has(companyId)) latestPriceByCompany.set(companyId, row as UnknownRecord);
  }

  return snapshots
    .map((snapshot) => {
      const issuer = issuerById.get(snapshot.issuer_id);
      const security = securityById.get(snapshot.security_id);
      if (!issuer || !security) return null;
      return mapCompanyState(snapshot, {
        issuer,
        security,
        company: companyById.get(snapshot.issuer_id) ?? null,
        marketPrice: latestPriceByCompany.get(snapshot.issuer_id) ?? null,
      });
    })
    .filter((row): row is CompanyState => row !== null)
    .sort((a, b) => a.name.localeCompare(b.name));
}

export async function getCompanyStates(): Promise<CompanyState[]> {
  return loadPublishedCanonicalStates();
}

export async function getCompanyStateBySlug(slug: string): Promise<CompanyState | null> {
  const states = await loadPublishedCanonicalStates();
  return states.find((row) => row.slug === slug) ?? null;
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
  const { data: dossiers, error: dossierError } = await supabase
    .from('research_dossiers')
    .select('dossier_id')
    .eq('issuer_id', companyId)
    .eq('active', true)
    .order('created_at', { ascending: false })
    .limit(1);
  if (dossierError) throw new Error(`Unable to load canonical dossier history: ${dossierError.message}`);
  const dossierId = dossiers?.[0]?.dossier_id;
  if (!dossierId) return [];

  const { data, error } = await supabase
    .from('research_snapshots')
    .select('snapshot_id,dossier_id,issuer_id,security_id,report_id,calculation_date,report_version,method_version,canonical_payload,created_at')
    .eq('dossier_id', dossierId)
    .order('calculation_date', { ascending: false })
    .order('created_at', { ascending: false });
  if (error) throw new Error(`Unable to load canonical snapshot history: ${error.message}`);
  return ((data ?? []) as CanonicalSnapshotRow[]).map(mapSnapshotHistory);
}

export function withDistance(company: CompanyState) {
  return { ...company, distance_o90_pct: getDistanceO90(company.price, company.price_o90) };
}
