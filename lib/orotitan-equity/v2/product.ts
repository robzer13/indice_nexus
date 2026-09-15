import taxonomy from "../../../contracts/orotitan-equity/v2/OROTITAN_TAXONOMY_V2.0.json";

export type V2Classification = {
  issuer_country_code: string;
  primary_listing_country_code: string;
  sector: string;
  industry_group: string;
  business_model_primary: string;
  business_model_secondary?: string | null;
  economic_exposure_regions: string[];
  taxonomy_version: "OROTITAN_TAXONOMY_V2.0";
};

export type V2Product = {
  classification: V2Classification;
  business_summary: { business_description_short: string };
  investment_thesis: { quality_case: string; valuation_case: string; key_risk: string };
  portfolio_filters?: {
    pea_eligibility?: "YES" | "NO" | "UNKNOWN";
    pea_eligibility_as_of?: string | null;
    pea_eligibility_source_ref?: string | null;
  };
};

const ISO_ALPHA2 = new Set(`AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW`.split(" "));
const DESCRIPTION_FORBIDDEN = /\b(OQS|OVS|investment score|fair value|undervalued|overvalued|buy|sell|recommend(?:ed|ation)?)\b/i;

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validDate(value: unknown): boolean {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value) && !Number.isNaN(Date.parse(`${value}T00:00:00Z`));
}

export function validateV2ProductSemantics(value: unknown): string[] {
  if (!isObject(value)) return ["v2_product must be an object"];
  const errors: string[] = [];
  const classification = isObject(value.classification) ? value.classification : {};
  const issuerCountry = classification.issuer_country_code;
  const listingCountry = classification.primary_listing_country_code;
  if (typeof issuerCountry !== "string" || !ISO_ALPHA2.has(issuerCountry)) errors.push("classification.issuer_country_code must be a valid ISO 3166-1 alpha-2 code");
  if (typeof listingCountry !== "string" || !ISO_ALPHA2.has(listingCountry)) errors.push("classification.primary_listing_country_code must be a valid ISO 3166-1 alpha-2 code");

  const sectors = new Set<string>(taxonomy.sectors);
  const industryGroups = new Set<string>(taxonomy.industry_groups);
  const businessModels = new Set<string>(taxonomy.business_models);
  const exposureRegions = new Set<string>(taxonomy.economic_exposure_regions);
  if (typeof classification.sector !== "string" || !sectors.has(classification.sector)) errors.push("classification.sector is not in OROTITAN_TAXONOMY_V2.0");
  if (typeof classification.industry_group !== "string" || !industryGroups.has(classification.industry_group)) errors.push("classification.industry_group is not in OROTITAN_TAXONOMY_V2.0");
  if (typeof classification.business_model_primary !== "string" || !businessModels.has(classification.business_model_primary)) errors.push("classification.business_model_primary is not in OROTITAN_TAXONOMY_V2.0");
  if (classification.business_model_secondary !== undefined && classification.business_model_secondary !== null
      && (typeof classification.business_model_secondary !== "string" || !businessModels.has(classification.business_model_secondary))) {
    errors.push("classification.business_model_secondary is not in OROTITAN_TAXONOMY_V2.0");
  }
  if (!Array.isArray(classification.economic_exposure_regions) || classification.economic_exposure_regions.length === 0
      || classification.economic_exposure_regions.some((region) => typeof region !== "string" || !exposureRegions.has(region))) {
    errors.push("classification.economic_exposure_regions contains an invalid controlled value");
  }
  if (classification.taxonomy_version !== taxonomy.taxonomy_version) errors.push("classification.taxonomy_version mismatch");

  const summary = isObject(value.business_summary) ? value.business_summary : {};
  const description = summary.business_description_short;
  if (typeof description !== "string" || description.trim().length === 0 || description.length > 450) {
    errors.push("business_summary.business_description_short must contain 1-450 characters");
  } else if (DESCRIPTION_FORBIDDEN.test(description)) {
    errors.push("business_summary.business_description_short must remain factual and neutral without score, valuation or recommendation language");
  }

  const thesis = isObject(value.investment_thesis) ? value.investment_thesis : {};
  for (const field of ["quality_case", "valuation_case", "key_risk"] as const) {
    const text = thesis[field];
    if (typeof text !== "string" || text.trim().length === 0 || text.length > 240) errors.push(`investment_thesis.${field} must contain 1-240 characters`);
  }
  const thesisKeys = Object.keys(thesis).sort();
  if (thesisKeys.join("|") !== ["key_risk", "quality_case", "valuation_case"].sort().join("|")) {
    errors.push("investment_thesis must contain exactly quality_case, valuation_case and key_risk");
  }

  if (value.portfolio_filters !== undefined) {
    const filters = isObject(value.portfolio_filters) ? value.portfolio_filters : {};
    const eligibility = filters.pea_eligibility;
    if (eligibility !== undefined && !taxonomy.pea_eligibility.includes(eligibility as never)) errors.push("portfolio_filters.pea_eligibility is invalid");
    if (eligibility === "YES" || eligibility === "NO") {
      if (!validDate(filters.pea_eligibility_as_of)) errors.push("portfolio_filters.pea_eligibility_as_of is required for YES/NO");
      if (typeof filters.pea_eligibility_source_ref !== "string" || filters.pea_eligibility_source_ref.trim().length === 0) errors.push("portfolio_filters.pea_eligibility_source_ref is required for YES/NO");
    }
  }
  return errors;
}

export function isV2Product(value: unknown): value is V2Product {
  return validateV2ProductSemantics(value).length === 0;
}
