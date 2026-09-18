import Ajv2020, { type ErrorObject } from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { readFileSync } from "node:fs";
import { validateCanonicalContract, type CanonicalContractInput } from "./contract";
import { assertInvestmentPolicyV1 } from "./investment-policy";
import { selectNormalizationReturn } from "./n-basis";
import { semanticStateSchema } from "./semantic-states";

type JsonObject = Record<string, unknown>;
type ScoreValue = number | { min: number; max: number } | string;

export type ResearchSnapshot = JsonObject;

export type ValidationFailure = {
  ok: false;
  stage: "schema" | "boundary" | "reconciliation";
  errors: string[];
};

export type ValidatedResearchSnapshot = {
  ok: true;
  snapshot: ResearchSnapshot;
  canonicalContract: CanonicalContractInput;
};

export type ResearchSnapshotSchemaValidator = ((input: unknown) => boolean) & {
  errors?: ErrorObject[] | null;
};

const schema = JSON.parse(readFileSync(new URL("../../../contracts/orotitan-equity/v1/04_SCREENER_SCHEMA_V1_PATCHED.json", import.meta.url), "utf8")) as JsonObject;
const ajv = new Ajv2020({ allErrors: true, strict: true, strictRequired: false, strictTypes: false });
// ajv-formats types its plugin against the default Ajv class, while this contract requires the Draft 2020 implementation.
// The runtime plugin API is compatible; bridge only the package-level type mismatch without changing validator behavior.
addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);
for (const keyword of [
  "x-orotitan-authority",
  "x-orotitan-layer-model",
  "x-orotitan-null-semantics",
  "x-orotitan-computed-vs-stored",
  "x-orotitan-deterministic-rules",
  "x-orotitan-server-recomputation-policy",
  "x-storage-precision",
  "x-unit",
]) ajv.addKeyword({ keyword });
ajv.addSchema(schema);
const validateSchema = ajv.compile({ $ref: "urn:orotitan:equity-research:screener-contract:v1#/$defs/researchSnapshot" });

function formatAjvErrors(errors: ErrorObject[] | null | undefined): string[] {
  return (errors ?? []).map((error) => `${error.instancePath || "/"} ${error.message ?? "is invalid"}`);
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isUuid(value: unknown): value is string {
  return typeof value === "string"
    && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
}

function findReversedRanges(value: unknown, path = ""): string[] {
  if (Array.isArray(value)) return value.flatMap((item, index) => findReversedRanges(item, `${path}/${index}`));
  if (!isObject(value)) return [];
  const errors: string[] = [];
  if (typeof value.min === "number" && typeof value.max === "number" && value.min > value.max) {
    errors.push(`${path || "/"}: range min must be <= max`);
  }
  for (const [key, child] of Object.entries(value)) errors.push(...findReversedRanges(child, `${path}/${key}`));
  return errors;
}

function scoreAt(snapshot: JsonObject, path: string[]): ScoreValue {
  let current: unknown = snapshot;
  for (const key of path) current = isObject(current) ? current[key] : undefined;
  return current as ScoreValue;
}

function readRequired(snapshot: JsonObject, path: string[]): unknown {
  const value = scoreAt(snapshot, path);
  if (value === undefined) throw new Error(`Missing deterministic input at ${path.join(".")}`);
  return value;
}

function subtractH(value: unknown, hurdle: number, path: string): number | { min: number; max: number } | string {
  if (typeof value === "number") return value - hurdle;
  if (isObject(value) && typeof value.min === "number" && typeof value.max === "number") {
    return { min: value.min - hurdle, max: value.max - hurdle };
  }
  if (typeof value === "string" && semanticStateSchema.safeParse(value).success) return value;
  throw new Error(`${path} is not a valid return value`);
}

function toContractInput(snapshot: JsonObject): CanonicalContractInput {
  const l2 = readRequired(snapshot, ["l2_research_fundamentals"]) as JsonObject;
  const fundamentals = readRequired(l2, ["fundamental_states"]) as JsonObject;
  const quality = readRequired(l2, ["business_quality"]) as JsonObject;
  const certification = readRequired(l2, ["certification"]) as JsonObject;
  const l3 = readRequired(snapshot, ["l3_investment_valuation"]) as JsonObject;
  const valuation = readRequired(l3, ["valuation"]) as JsonObject;
  const investment = readRequired(l3, ["investment"]) as JsonObject;
  const priceLadder = readRequired(valuation, ["price_ladder"]) as JsonObject;
  const requiredReturnH = readRequired(priceLadder, ["required_return_h"]);
  const strongReturnThreshold = readRequired(priceLadder, ["strong_return_threshold"]);
  const exceptionalReturnThreshold = readRequired(priceLadder, ["exceptional_return_threshold"]);
  assertInvestmentPolicyV1({ requiredReturnH, strongReturnThreshold, exceptionalReturnThreshold });
  const primaryReturn = readRequired(valuation, ["primary_expected_return"]);
  const nReturn = selectNormalizationReturn({
    noMultipleExpansionReturn: readRequired(valuation, ["no_multiple_expansion_return"]),
    matureNormalizationReturn: readRequired(valuation, ["mature_normalization_return"]),
  });
  const l4 = readRequired(snapshot, ["l4_operational_state"]) as JsonObject;
  const orotitan = readRequired(l4, ["orotitan"]) as JsonObject;
  const gates = readRequired(orotitan, ["orotitan_gate_results"]);
  if (!Array.isArray(gates)) throw new Error("l4_operational_state.orotitan.orotitan_gate_results must be an array");

  const eliteGates: JsonObject = {};
  const gateNames: Record<string, string> = {
    CERTIFICATION_GATE: "researchFullyCertified",
    MOAT_ELITE: "moatElite",
    RUNWAY_ELITE: "runwayElite",
    RETURN_QUALITY_ELITE: "returnQualityElite",
    CASH_ECONOMICS_ELITE: "cashEconomicsElite",
    CAPITAL_ALLOCATION_ELITE: "capitalAllocationElite",
    MANAGEMENT_GOVERNANCE_ELITE: "managementGovernanceElite",
    RESILIENCE_ELITE: "resilienceElite",
    MATERIAL_WEAK_LINK_GATE: "materialWeakLink",
    VALUATION_ELITE: "valuationElite",
  };
  for (const gate of gates) {
    if (!isObject(gate) || typeof gate.gate !== "string" || typeof gate.state !== "string") {
      throw new Error("Malformed OroTitan gate result");
    }
    const key = gateNames[gate.gate];
    if (key) eliteGates[key] = gate.state;
  }

  const dimensions = {
    MOAT: readRequired(quality, ["moat_score"]),
    RUNWAY: readRequired(quality, ["runway_score"]),
    RETURN_QUALITY: readRequired(quality, ["return_quality_score"]),
    CASH_ECONOMICS: readRequired(quality, ["cash_economics_score"]),
    CAPITAL_ALLOCATION: readRequired(quality, ["capital_allocation_score"]),
    MANAGEMENT_GOVERNANCE: readRequired(quality, ["management_governance_score"]),
    RESILIENCE_RISK: readRequired(quality, ["resilience_risk_score"]),
  };
  const deterministic = {
    oqsRaw: readRequired(quality, ["oqs_raw"]),
    weakLinkCap: readRequired(quality, ["weak_link_cap"]),
    oqs: readRequired(quality, ["oqs"]),
    ovs: readRequired(valuation, ["ovs"]),
    investmentRaw: readRequired(investment, ["investment_raw"]),
    investmentScore: readRequired(investment, ["investment_score"]),
    orotitanStatus: readRequired(orotitan, ["orotitan_status"]),
  };
  return {
    dimensions,
    evidence: {
      moat: readRequired(fundamentals, ["moat_evidence_state"]),
      runway: readRequired(fundamentals, ["runway_evidence_state"]),
    },
    businessResearchStatus: readRequired(certification, ["business_research_status"]),
    investmentConclusionStatus: readRequired(certification, ["investment_conclusion_status"]),
    scorePermission: readRequired(certification, ["score_permission"]),
    mosStatus: readRequired(valuation, ["margin_of_safety"]),
    valuationReliability: readRequired(valuation, ["valuation_reliability"]),
    primaryExpectedReturnDeltaPercentagePoints: subtractH(primaryReturn, requiredReturnH as number, "primary_expected_return"),
    normalizedExpectedReturnDeltaPercentagePoints: subtractH(nReturn, requiredReturnH as number, "N return"),
    eliteGates,
    deterministic,
  } as CanonicalContractInput;
}

function boundaryErrors(snapshot: JsonObject, dossierId: string): string[] {
  const errors: string[] = [];
  for (const field of ["snapshot_id", "issuer_id", "security_id"]) {
    if (!isUuid(snapshot[field])) errors.push(`${field} must be a UUID compatible with physical persistence`);
  }
  if (!isUuid(dossierId)) errors.push("dossierId must be a UUID compatible with physical persistence");
  errors.push(...findReversedRanges(snapshot));
  return errors;
}

export function validateResearchSnapshotForPersistenceAgainstSchema(
  input: unknown,
  dossierId: string,
  schemaValidator: ResearchSnapshotSchemaValidator,
): ValidationFailure | ValidatedResearchSnapshot {
  if (!schemaValidator(input)) return { ok: false, stage: "schema", errors: formatAjvErrors(schemaValidator.errors) };
  const snapshot = input as JsonObject;
  const boundary = boundaryErrors(snapshot, dossierId);
  if (boundary.length > 0) return { ok: false, stage: "boundary", errors: boundary };
  if (snapshot.execution_mode === "DISCOVER") return { ok: true, snapshot, canonicalContract: {} as CanonicalContractInput };
  try {
    const canonicalContract = toContractInput(snapshot);
    const result = validateCanonicalContract(canonicalContract);
    if (!result.success) return { ok: false, stage: "reconciliation", errors: result.error.issues.map((issue) => `${issue.path.join(".") || "/"}: ${issue.message}`) };
    return { ok: true, snapshot, canonicalContract: result.data };
  } catch (error) {
    return { ok: false, stage: "reconciliation", errors: [error instanceof Error ? error.message : "Canonical reconciliation failed"] };
  }
}

export function validateResearchSnapshotForPersistence(input: unknown, dossierId: string): ValidationFailure | ValidatedResearchSnapshot {
  return validateResearchSnapshotForPersistenceAgainstSchema(input, dossierId, validateSchema as ResearchSnapshotSchemaValidator);
}

export { validateSchema };
