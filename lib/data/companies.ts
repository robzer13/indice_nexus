import 'server-only';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import { getDistanceO90 } from '@/lib/domain/distance';
import type { ActiveCompanyOption, CompanyState, CompanyStatus, Json, QuoteUnit, SnapshotHistoryRow } from '@/lib/domain/types';
import { computeMarketAdaptiveValuation } from '@/lib/orotitan-equity/market-adaptive';
import type { InvestmentConclusionStatus, MosStatus, ScorePermission, ValuationReliability } from '@/lib/orotitan-equity/v1/certification';

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

function asEnum<T extends string>(value: unknown, allowed: readonly T[]): T | null {
  return typeof value === 'string' && allowed.includes(value as T) ? value as T : null;
}

function isMissingRelation(error: { code?: string; message?: string } | null | undefined, relation: string): boolean {
  if (!error) return false;
  return error.code === '42P01' || error.code === 'PGRST205' || Boolean(error.message?.includes(relation));
}

function newestMarketRow(...rows: Array<UnknownRecord | null | undefined>): UnknownRecord | null {
  const present = rows.filter((row): row is UnknownRecord => Boolean(row));
  if (present.length === 0) return null;
  return present.sort((left, right) => {
    const leftTime = Date.parse(asString(left.as_of) ?? '');
    const rightTime = Date.parse(asString(right.as_of) ?? '');
    return (Number.isFinite(rightTime) ? rightTime : 0) - (Number.isFinite(leftTime) ? leftTime : 0);
  })[0] ?? null;
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
  const certification = asRecord(l2.certification);
  const l3 = asRecord(payload.l3_investment_valuation);
  const valuation = asRecord(l3.valuation);
  const investment = asRecord(l3.investment);
  const priceLadder = asRecord(valuation.price_ladder);
  const l4 = asRecord(payload.l4_operational_state);
  const orotitan = asRecord(l4.orotitan);
  const v2Product = asRecord(payload.v2_product);
  const classification = asRecord(v2Product.classification);
  const businessSummary = asRecord(v2Product.business_summary);
  const investmentThesis = asRecord(v2Product.investment_thesis);
  const portfolioFilters = asRecord(v2Product.portfolio_filters);

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

  const pea = asString(portfolioFilters.pea_eligibility);
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
    primaryExpectedReturn: asNumber(valuation.primary_expected_return),
    normalizationExpectedReturn: asNumber(valuation.mature_normalization_return) ?? asNumber(valuation.no_multiple_expansion_return),
    returnHorizon: asNumber(valuation.return_horizon),
    requiredReturnH: asNumber(priceLadder.required_return_h),
    mosStatus: asEnum<MosStatus>(valuation.margin_of_safety, ['ROBUST', 'ADEQUATE', 'THIN', 'NONE', 'NOT_ASSESSABLE']),
    valuationReliability: asEnum<ValuationReliability>(valuation.valuation_reliability, ['HIGH', 'MEDIUM', 'LOW', 'NOT_ASSESSABLE']),
    scorePermission: asEnum<ScorePermission>(certification.score_permission, ['ALLOWED', 'CONDITIONAL', 'SUSPENDED']),
    investmentConclusionStatus: asEnum<InvestmentConclusionStatus>(certification.investment_conclusion_status, ['CERTIFIED', 'CERTIFIED_WITH_LIMITATIONS', 'NOT_CERTIFIED', 'INSUFFICIENT_DATA']),
    invalidation: invalidationTriggers.length > 0 ? invalidationTriggers.join('\n') : null,
    scoreComponents,
    v2: {
      issuerCountryCode: asString(classification.issuer_country_code),
      primaryListingCountryCode: asString(classification.primary_listing_country_code),
      sector: asString(classification.sector),
      industryGroup: asString(classification.industry_group),
      businessModelPrimary: asString(classification.business_model_primary),
      businessModelSecondary: asString(classification.business_model_secondary),
      economicExposureRegions: asStringArray(classification.economic_exposure_regions),
      taxonomyVersion: asString(classification.taxonomy_version),
      businessDescriptionShort: asString(businessSummary.business_description_short),
      qualityCase: asString(investmentThesis.quality_case),
      valuationCase: asString(investmentThesis.valuation_case),
      keyRisk: asString(investmentThesis.key_risk),
      peaEligibility: pea === 'YES' || pea === 'NO' || pea === 'UNKNOWN' ? pea : null,
    },
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
    return { price: snapshot.referencePrice, priceAsOf: `${referenceDate}T00:00:00Z`, priceSource: 'CANONICAL_REFERENCE_PRICE' };
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
    thesis: canonical.v2.qualityCase,
    main_risk: canonical.v2.keyRisk,
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
  const adaptive = price.price !== null &&
    canonical.referencePrice !== null &&
    canonical.primaryExpectedReturn !== null &&
    canonical.normalizationExpectedReturn !== null &&
    canonical.returnHorizon !== null &&
    canonical.requiredReturnH !== null &&
    canonical.mosStatus !== null &&
    canonical.valuationReliability !== null &&
    canonical.scorePermission !== null &&
    canonical.investmentConclusionStatus !== null &&
    canonical.oqs !== null
    ? computeMarketAdaptiveValuation({
        currentPrice: price.price,
        referencePrice: canonical.referencePrice,
        horizonYears: canonical.returnHorizon,
        primaryExpectedReturnPct: canonical.primaryExpectedReturn,
        normalizationExpectedReturnPct: canonical.normalizationExpectedReturn,
        requiredReturnPct: canonical.requiredReturnH,
        mosStatus: canonical.mosStatus,
        valuationReliability: canonical.valuationReliability,
        scorePermission: canonical.scorePermission,
        investmentConclusionStatus: canonical.investmentConclusionStatus,
        oqs: canonical.oqs,
      })
    : null;
  const companySlug = asString(context.company?.slug) ?? slugify(issuerName);
  const quoteUnit = (asString(context.security.quote_unit) === 'MINOR' ? 'MINOR' : 'MAJOR') as QuoteUnit;

  return {
    id: row.issuer_id,
    security_id: row.security_id,
    slug: companySlug,
    ticker: asString(context.security.ticker) ?? '—',
    name: issuerName,
    exchange: asString(context.security.exchange) ?? '—',
    currency: asString(context.security.trading_currency) ?? asString(context.issuer.reporting_currency) ?? 'USD',
    quote_unit: quoteUnit,
    price_decimals: asNumber(context.security.price_decimals) ?? 2,
    country: canonical.v2.issuerCountryCode ?? asString(context.security.country) ?? asString(context.issuer.country),
    sector: canonical.v2.sector ?? asString(context.company?.sector),
    industry_group: canonical.v2.industryGroup,
    business_model_primary: canonical.v2.businessModelPrimary,
    business_model_secondary: canonical.v2.businessModelSecondary,
    economic_exposure_regions: canonical.v2.economicExposureRegions,
    taxonomy_version: canonical.v2.taxonomyVersion,
    business_description_short: canonical.v2.businessDescriptionShort,
    quality_case: canonical.v2.qualityCase,
    valuation_case: canonical.v2.valuationCase,
    key_risk: canonical.v2.keyRisk,
    pea_eligibility: canonical.v2.peaEligibility,
    market_data_symbol: asString(context.security.market_data_symbol),
    market_data_multiplier: asNumber(context.security.market_data_multiplier) ?? 1,
    price: price.price,
    price_as_of: price.priceAsOf,
    price_source: price.priceSource,
    reference_price: canonical.referencePrice,
    reference_price_date: canonical.referencePriceDate,
    market_price_change_vs_reference_pct: adaptive?.priceChangeVsReferencePct ?? null,
    market_valuation_score: adaptive?.ovs ?? null,
    market_investment_score: adaptive?.investmentScore ?? null,
    market_primary_expected_return: adaptive?.primaryExpectedReturnPct ?? null,
    market_normalization_expected_return: adaptive?.normalizationExpectedReturnPct ?? null,
    market_score_as_of: adaptive ? price.priceAsOf : null,
    market_score_is_live: Boolean(adaptive && price.priceSource !== 'CANONICAL_REFERENCE_PRICE'),
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
    thesis: canonical.v2.qualityCase,
    main_risk: canonical.v2.keyRisk,
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

  const [snapshotsResult, issuersResult, companiesResult, legacyPricesResult] = await Promise.all([
    supabase.from('research_snapshots').select('snapshot_id,dossier_id,issuer_id,security_id,report_id,calculation_date,report_version,method_version,canonical_payload,created_at').in('snapshot_id', snapshotIds),
    supabase.from('issuers').select('*').in('issuer_id', issuerIds),
    supabase.from('companies').select('id,slug,sector').in('id', issuerIds),
    supabase.from('market_prices').select('company_id,price,as_of,source,created_at').in('company_id', issuerIds).order('as_of', { ascending: false }),
  ]);

  if (snapshotsResult.error) throw new Error(`Unable to load canonical snapshots: ${snapshotsResult.error.message}`);
  if (issuersResult.error) throw new Error(`Unable to load canonical issuers: ${issuersResult.error.message}`);
  if (companiesResult.error) throw new Error(`Unable to load company display metadata: ${companiesResult.error.message}`);
  if (legacyPricesResult.error) throw new Error(`Unable to load legacy market prices: ${legacyPricesResult.error.message}`);

  const snapshots = (snapshotsResult.data ?? []) as CanonicalSnapshotRow[];
  const securityIds = snapshots.map((row) => row.security_id);
  const [securitiesResult, securityPricesResult] = await Promise.all([
    supabase.from('securities').select('*').in('security_id', securityIds),
    supabase.from('security_market_prices').select('security_id,price,as_of,source,created_at').in('security_id', securityIds).order('as_of', { ascending: false }),
  ]);
  if (securitiesResult.error) throw new Error(`Unable to load canonical securities: ${securitiesResult.error.message}`);
  if (securityPricesResult.error && !isMissingRelation(securityPricesResult.error, 'security_market_prices')) {
    throw new Error(`Unable to load canonical security market prices: ${securityPricesResult.error.message}`);
  }

  const issuerById = new Map((issuersResult.data ?? []).map((row) => [row.issuer_id as string, row as UnknownRecord]));
  const securityById = new Map((securitiesResult.data ?? []).map((row) => [row.security_id as string, row as UnknownRecord]));
  const companyById = new Map((companiesResult.data ?? []).map((row) => [row.id as string, row as UnknownRecord]));

  const latestLegacyPriceByIssuer = new Map<string, UnknownRecord>();
  for (const row of legacyPricesResult.data ?? []) {
    const issuerId = row.company_id as string;
    if (!latestLegacyPriceByIssuer.has(issuerId)) latestLegacyPriceByIssuer.set(issuerId, row as UnknownRecord);
  }

  const latestSecurityPriceBySecurity = new Map<string, UnknownRecord>();
  for (const row of securityPricesResult.data ?? []) {
    const securityId = row.security_id as string;
    if (!latestSecurityPriceBySecurity.has(securityId)) latestSecurityPriceBySecurity.set(securityId, row as UnknownRecord);
  }

  return snapshots.map((snapshot) => {
    const issuer = issuerById.get(snapshot.issuer_id);
    const security = securityById.get(snapshot.security_id);
    if (!issuer || !security) return null;
    const marketPrice = newestMarketRow(
      latestSecurityPriceBySecurity.get(snapshot.security_id),
      latestLegacyPriceByIssuer.get(snapshot.issuer_id),
    );
    return mapCompanyState(snapshot, {
      issuer,
      security,
      company: companyById.get(snapshot.issuer_id) ?? null,
      marketPrice,
    });
  }).filter((row): row is CompanyState => row !== null).sort((a, b) => a.name.localeCompare(b.name));
}

export async function getCompanyStates(): Promise<CompanyState[]> { return loadPublishedCanonicalStates(); }

export async function getCompanyStateBySlug(slug: string): Promise<CompanyState | null> {
  const states = await loadPublishedCanonicalStates();
  return states.find((row) => row.slug === slug) ?? null;
}

export async function getActiveCompanies(): Promise<ActiveCompanyOption[]> {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase.from('companies').select('id,slug,ticker,name,exchange').eq('active', true).order('name', { ascending: true });
  if (error) throw new Error(`Unable to load companies: ${error.message}`);
  return (data ?? []) as ActiveCompanyOption[];
}

export async function getSnapshotHistory(companyId: string): Promise<SnapshotHistoryRow[]> {
  const supabase = createServerSupabaseClient();
  const { data: dossiers, error: dossierError } = await supabase.from('research_dossiers').select('dossier_id').eq('issuer_id', companyId).eq('active', true).order('created_at', { ascending: false }).limit(1);
  if (dossierError) throw new Error(`Unable to load canonical dossier history: ${dossierError.message}`);
  const dossierId = dossiers?.[0]?.dossier_id;
  if (!dossierId) return [];
  const { data, error } = await supabase.from('research_snapshots').select('snapshot_id,dossier_id,issuer_id,security_id,report_id,calculation_date,report_version,method_version,canonical_payload,created_at').eq('dossier_id', dossierId).order('calculation_date', { ascending: false }).order('created_at', { ascending: false });
  if (error) throw new Error(`Unable to load canonical snapshot history: ${error.message}`);
  return ((data ?? []) as CanonicalSnapshotRow[]).map(mapSnapshotHistory);
}

export function withDistance(company: CompanyState) {
  return { ...company, distance_o90_pct: getDistanceO90(company.price, company.price_o90) };
}
