import Ajv2020 from "ajv/dist/2020";
import addFormats from "ajv-formats";
import ledgerSchema from "../../../contracts/orotitan-equity/method-v2/schemas/PRE_CERTIFICATION_QUESTION_LEDGER_V1.0.schema.json";
import reportSchema from "../../../contracts/orotitan-equity/method-v2/schemas/PRE_CERTIFICATION_CHALLENGE_REPORT_V1.0.schema.json";
import { canonicalJson, METHOD_V2_AUTHORITY_SET_SHA256, sha256 } from "./authority";

export const CHALLENGE_FAMILIES = ["CROSS_BLOCK_CONSISTENCY", "GREAT_INVESTOR_DECISION_THINKING", "WORLD_IN_MOTION", "OPERATIONAL_REALITY_SUPPLY_CHAIN", "TECHNOLOGY_AI", "SECOND_ORDER_EFFECTS", "COUNTERFACTUALS", "COMPANY_SPECIFIC_CHALLENGE"] as const;
export type ArtifactRef = { artifact_id: string; version: number; content_sha256: string };
type Question = {
  question_id: string; question_family: string; status: "PASS" | "CONCERN" | "REOPEN" | "FAIL";
  company_specific: boolean; evidence_dates: string[]; decision_impact: string;
  mitigation_or_resolution?: string | null; reopen_target?: string[] | null;
};
type Delta = null | { prior_passing_report_ref: ArtifactRef; prior_passing_ledger_ref: ArtifactRef; impact_map: string; inherited_coverage_revalidated: true };
type Common = {
  run_id: string; company: string; data_cutoff: string; challenge_iteration: number;
  fundamentals_lock_ref: ArtifactRef; valuation_lock_ref: ArtifactRef; mode: "FULL" | "DELTA"; delta_provenance: Delta;
};
type Ledger = Common & { questions: Question[]; dropped_candidates: { question_id: string; reason: string; rationale: string }[] };
type Report = Common & {
  question_ledger_ref: ArtifactRef; total_questions_executed: number; total_candidate_questions_generated: number;
  pass_count: number; concern_count: number; reopen_count: number; fail_count: number;
  dropped_duplicate_count: number; dropped_already_resolved_count: number; dropped_non_material_count: number;
  material_concerns: string[]; company_specific_questions: string[];
  challenge_status: string; ready_for_certification: "YES" | "NO";
  saturation_record: {
    mandatory_families_covered: boolean; coverage_rationale: Record<string, string>;
    company_specific_coverage: "PASS" | "INSUFFICIENT"; company_specific_rationale: string;
    all_material_concerns_dispositioned: boolean; material_questions_remaining: boolean; stop_reason: string;
    final_generation_passes: { pass_id: string; independence_rationale: string; new_material_questions: number; conclusion_scope_sha256: string }[];
  };
};
export type ChallengeContext = {
  analyticalAuthoritySetSha256: string; run_id: string; company: string; data_cutoff: string;
  fundamentals_lock_ref: ArtifactRef; valuation_lock_ref: ArtifactRef;
  question_ledger_ref: ArtifactRef; challenge_report_ref: ArtifactRef;
  // Trusted current lineage supplied by the consuming persisted-artifact resolver.
  // The pure validator itself neither contacts nor modifies a Registry.
  priorPassing?: { report_ref: ArtifactRef; ledger_ref: ArtifactRef };
};
const ajv = new Ajv2020({ allErrors: true, strict: true });
addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);
const validateLedger = ajv.compile(ledgerSchema);
const validateReport = ajv.compile(reportSchema);
const same = (a: unknown, b: unknown): boolean => canonicalJson(a) === canonicalJson(b);
const ensure = (ok: boolean, reason: string): void => { if (!ok) throw new Error(reason); };

export function challengeScopeSha256(context: ChallengeContext): string {
  return sha256(canonicalJson({ run_id: context.run_id, data_cutoff: context.data_cutoff,
    fundamentals_lock_ref: context.fundamentals_lock_ref, valuation_lock_ref: context.valuation_lock_ref,
    question_ledger_ref: context.question_ledger_ref }));
}

/** Offline admission evidence check, never Certification or production activation. */
export function admitMethodV2Certification(context: ChallengeContext, ledgerBytes?: Uint8Array, reportBytes?: Uint8Array): {
  allowed: true; limitations: { question_id: string; decision_impact: string; mitigation_or_resolution: string }[];
} {
  ensure(context.analyticalAuthoritySetSha256 === METHOD_V2_AUTHORITY_SET_SHA256, "METHOD_AUTHORITY_MISMATCH");
  if (!ledgerBytes || !reportBytes) throw new Error("CHALLENGE_ABSENT");
  ensure(sha256(ledgerBytes) === context.question_ledger_ref.content_sha256 && sha256(reportBytes) === context.challenge_report_ref.content_sha256, "CHALLENGE_HASH_MISMATCH");
  const rawLedger: unknown = JSON.parse(Buffer.from(ledgerBytes).toString("utf8"));
  const rawReport: unknown = JSON.parse(Buffer.from(reportBytes).toString("utf8"));
  ensure(validateLedger(rawLedger) === true && validateReport(rawReport) === true, "CHALLENGE_SCHEMA_INVALID");
  const ledger = rawLedger as Ledger;
  const report = rawReport as Report;
  for (const artifact of [ledger, report]) {
    ensure(artifact.run_id === context.run_id && artifact.company === context.company && artifact.data_cutoff === context.data_cutoff
      && same(artifact.fundamentals_lock_ref, context.fundamentals_lock_ref) && same(artifact.valuation_lock_ref, context.valuation_lock_ref), "CHALLENGE_STALE_LINEAGE");
    ensure(artifact.mode === "FULL" ? artifact.delta_provenance === null : !!artifact.delta_provenance && !!context.priorPassing
      && same(artifact.delta_provenance.prior_passing_report_ref, context.priorPassing.report_ref)
      && same(artifact.delta_provenance.prior_passing_ledger_ref, context.priorPassing.ledger_ref), "CHALLENGE_DELTA_PROVENANCE_INVALID");
  }
  ensure(same(report.question_ledger_ref, context.question_ledger_ref) && report.challenge_iteration === ledger.challenge_iteration
    && report.mode === ledger.mode && same(report.delta_provenance, ledger.delta_provenance), "CHALLENGE_LEDGER_MISMATCH");
  const ids = [...ledger.questions, ...ledger.dropped_candidates].map(q => q.question_id);
  ensure(new Set(ids).size === ids.length, "CHALLENGE_DUPLICATE_QUESTION_ID");
  ensure(ledger.questions.every(q => q.evidence_dates.every(date => date <= context.data_cutoff)), "CHALLENGE_POST_CUTOFF_EVIDENCE");
  const count = (status: Question["status"]): number => ledger.questions.filter(q => q.status === status).length;
  const drops = (reason: string): number => ledger.dropped_candidates.filter(q => q.reason === reason).length;
  ensure(report.total_questions_executed === ledger.questions.length
    && report.total_candidate_questions_generated === ids.length
    && report.pass_count === count("PASS") && report.concern_count === count("CONCERN")
    && report.reopen_count === count("REOPEN") && report.fail_count === count("FAIL")
    && report.dropped_duplicate_count === drops("DROP_DUPLICATE")
    && report.dropped_already_resolved_count === drops("DROP_ALREADY_RESOLVED")
    && report.dropped_non_material_count === drops("DROP_NON_MATERIAL"), "CHALLENGE_TELEMETRY_MISMATCH");
  const status = count("FAIL") ? "FAIL" : count("REOPEN") ? "REOPEN" : count("CONCERN") ? "PASS_WITH_CONCERNS" : "PASS";
  ensure(report.challenge_status === status, "CHALLENGE_AGGREGATE_MISMATCH");
  ensure(status !== "FAIL" && status !== "REOPEN", `CHALLENGE_${status}`);
  ensure(report.ready_for_certification === "YES", "CHALLENGE_NOT_READY");
  const saturation = report.saturation_record;
  ensure(saturation.mandatory_families_covered && CHALLENGE_FAMILIES.every(f => saturation.coverage_rationale[f]?.trim())
    && saturation.company_specific_coverage === "PASS" && !!saturation.company_specific_rationale.trim()
    && saturation.all_material_concerns_dispositioned && !saturation.material_questions_remaining
    && saturation.stop_reason === "NO_ADDITIONAL_MATERIAL_DECISION_USEFUL_QUESTION_IDENTIFIED", "CHALLENGE_NOT_SATURATED");
  const passes = saturation.final_generation_passes;
  ensure(passes.length >= 2 && new Set(passes.map(p => p.pass_id)).size === passes.length
    && passes.every(p => !!p.independence_rationale.trim() && p.new_material_questions === 0
      && p.conclusion_scope_sha256 === challengeScopeSha256(context)), "CHALLENGE_SATURATION_PASSES_INVALID");
  const concerns = ledger.questions.filter(q => q.status === "CONCERN");
  ensure(same([...report.material_concerns].sort(), concerns.map(q => q.question_id).sort())
    && concerns.every(q => !!q.mitigation_or_resolution?.trim()), "CHALLENGE_CONCERNS_NOT_PROPAGATED");
  ensure(same([...report.company_specific_questions].sort(), ledger.questions.filter(q => q.company_specific).map(q => q.question_id).sort()), "CHALLENGE_COMPANY_COVERAGE_MISMATCH");
  return { allowed: true, limitations: concerns.map(q => ({ question_id: q.question_id, decision_impact: q.decision_impact,
    mitigation_or_resolution: q.mitigation_or_resolution! })) };
}
