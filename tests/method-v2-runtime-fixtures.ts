import { METHOD_V2_AUTHORITY_SET_SHA256, sha256 } from "../lib/orotitan-equity/method-v2/authority";
import { admitMethodV2Certification, CHALLENGE_FAMILIES, challengeScopeSha256, type ChallengeContext, type FamilyCoverage } from "../lib/orotitan-equity/method-v2/pre-certification";
const ref = (n: number) => ({ artifact_id: `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`, version: 1, content_sha256: "a".repeat(64) });
export function challengeFixture(status: "PASS" | "CONCERN" | "REOPEN" | "FAIL" = "PASS") {
  const common = { company: "Synthetic Company", run_id: ref(1).artifact_id, data_cutoff: "2026-10-01",
    schema_version: "1.1", challenge_version: "1.1", challenge_iteration: 1,
    mode: "FULL", delta_provenance: null, fundamentals_lock_ref: ref(2), valuation_lock_ref: ref(3) };
  const questions = CHALLENGE_FAMILIES.map((family, index) => ({ question_id: `Q${index}`, question_family: family,
    question_text: `Does the specific supplier concentration invalidate the ${family} conclusion?`, company_specific: true,
    origin_trigger: ["supplier-contract-1"], why_material: "Capacity determines the locked growth assumption", novelty_basis: "Independent adversarial review",
    answer_or_judgment: "Cutoff evidence reconciled", status: index === 0 ? status : "PASS", evidence_references: ["E1"],
    evidence_dates: ["2026-09-30"], affected_blocks: ["VALUATION"], decision_impact: "Bounds the investment conclusion",
    mitigation_or_resolution: "Contractual alternative capacity; retain residual limitation" }));
  const ledger = { ...common, artifact_type: "PRE_CERTIFICATION_QUESTION_LEDGER", questions,
    dropped_candidates: [] as { question_id: string; reason: string; rationale: string }[] };
  const report = { ...common, artifact_type: "PRE_CERTIFICATION_CHALLENGE_REPORT", question_ledger_ref: ref(4),
    total_candidate_questions_generated: questions.length, total_questions_executed: questions.length,
    dropped_duplicate_count: 0, dropped_already_resolved_count: 0, dropped_non_material_count: 0,
    pass_count: status === "PASS" ? questions.length : questions.length - 1, concern_count: status === "CONCERN" ? 1 : 0,
    reopen_count: status === "REOPEN" ? 1 : 0, fail_count: status === "FAIL" ? 1 : 0,
    material_concerns: status === "CONCERN" ? ["Q0"] : [], blind_spots_identified: [], company_specific_questions: questions.map(q => q.question_id),
    reopen_required: status === "REOPEN" ? ["VALUATION"] : [], fail_reasons: status === "FAIL" ? ["Q0"] : [], affected_analytical_blocks: ["VALUATION"],
    challenge_status: status === "CONCERN" ? "PASS_WITH_CONCERNS" : status, ready_for_certification: ["REOPEN", "FAIL"].includes(status) ? "NO" : "YES",
    saturation_record: { mandatory_families_covered: true, coverage_rationale: Object.fromEntries(CHALLENGE_FAMILIES.map(f => [f, "Company-specific coverage tied to current locks"])),
      family_coverage: Object.fromEntries(CHALLENGE_FAMILIES.map((f, i) => [f, { disposition: "QUESTION_COVERED", question_ids: [`Q${i}`] }])) as Record<typeof CHALLENGE_FAMILIES[number], FamilyCoverage>,
      company_specific_coverage: "PASS", company_specific_rationale: "Counterparty and capacity specific", all_material_concerns_dispositioned: true,
      material_questions_remaining: false, stop_reason: "NO_ADDITIONAL_MATERIAL_DECISION_USEFUL_QUESTION_IDENTIFIED",
      final_generation_passes: [1, 2].map(i => ({ pass_id: `P${i}`, independence_rationale: `Independent inversion pass ${i} documented separately`, new_material_questions: 0, conclusion_scope_sha256: "" })) } };
  const context: ChallengeContext = { analyticalAuthoritySetSha256: METHOD_V2_AUTHORITY_SET_SHA256, run_id: common.run_id, company: common.company,
    data_cutoff: common.data_cutoff, fundamentals_lock_ref: ref(2), valuation_lock_ref: ref(3), question_ledger_ref: ref(4), challenge_report_ref: ref(5) };
  function bytes() {
    const ledgerBytes = Buffer.from(JSON.stringify(ledger));
    context.question_ledger_ref.content_sha256 = sha256(ledgerBytes);
    report.question_ledger_ref = { ...context.question_ledger_ref };
    for (const pass of report.saturation_record.final_generation_passes) pass.conclusion_scope_sha256 = challengeScopeSha256(context);
    const reportBytes = Buffer.from(JSON.stringify(report));
    context.challenge_report_ref.content_sha256 = sha256(reportBytes);
    return { ledgerBytes, reportBytes };
  }
  function admit() { const b = bytes(); return admitMethodV2Certification(context, b.ledgerBytes, b.reportBytes); }
  return { ledger, report, context, bytes, admit };
}
