import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import { authoritySetSha256, classifyMethodGeneration, methodV2Manifest, METHOD_V2_AUTHORITY_SET_SHA256, METHOD_V2_V1_0_AUTHORITY_SET_SHA256, PRE_CERTIFICATION_EFFECTIVE_SHA256, sha256, verifyMethodV2Authority } from "../lib/orotitan-equity/method-v2/authority";
import { admitMethodV2Certification, CHALLENGE_FAMILIES, challengeScopeSha256, type ChallengeContext, type FamilyCoverage } from "../lib/orotitan-equity/method-v2/pre-certification";
import { resolveContractPin, type ContractPin, type GithubBlobRequest } from "../lib/orotitan-equity/v1/contract-pin-resolver";

const fetchLocal = async ({ repository, path }: GithubBlobRequest): Promise<Uint8Array> => {
  assert.equal(repository, "robzer13/indice_nexus");
  return readFileSync(path);
};
const historical = (versions = {}) => ({ historicalBeforeActivation: true, ...versions });
const ref = (n: number) => ({ artifact_id: `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`, version: 1, content_sha256: "a".repeat(64) });
function fixture(status: "PASS" | "CONCERN" | "REOPEN" | "FAIL" = "PASS") {
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

test("M01 V1 inherited authority is independently resolved from exact canonical bytes", async () => {
  for (const member of methodV2Manifest.members.filter(m => m.role === "INHERITED_BASE")) {
    const bytes = await resolveContractPin(member.pin as ContractPin, fetchLocal);
    assert.equal(sha256(bytes), member.pin.content_sha256);
  }
});
test("M02 scoped successor precedence preserves default and scoring authority", () => {
  assert.equal(methodV2Manifest.precedence.default, "v1_analysis_standard");
  assert.equal(methodV2Manifest.precedence.valuation_date_alignment[0], "valuation_date_alignment");
  assert.equal(methodV2Manifest.precedence.economic_share_count[0], "economic_share_count");
  assert.equal(methodV2Manifest.precedence.scoring[0], "v1_analysis_standard");
  assert.match(methodV2Manifest.members.find(m => m.id === "valuation_date_alignment")!.scope, /date clauses.*only/);
});
test("M03 historical runtime V2 is METHOD_V1", () => assert.equal(classifyMethodGeneration(historical({ runtimeVersion: "2.0" })), "METHOD_V1"));
test("M04 historical runtime V3 is METHOD_V1", () => assert.equal(classifyMethodGeneration(historical({ runtimeVersion: "3.0.2" })), "METHOD_V1"));
test("M05 schema 2.0.0 cannot establish Method-V2", () => {
  assert.equal(classifyMethodGeneration(historical({ schemaVersion: "2.0.0" })), "METHOD_V1");
  assert.throws(() => classifyMethodGeneration({ historicalBeforeActivation: false, schemaVersion: "2.0.0" }), /MISSING_OR_MIXED/);
});
test("M06 exact manifest identifies METHOD_V2; every member and packaging byte verifies", async () => {
  assert.equal(authoritySetSha256(methodV2Manifest), METHOD_V2_AUTHORITY_SET_SHA256);
  await verifyMethodV2Authority(methodV2Manifest, fetchLocal);
  assert.equal(classifyMethodGeneration({ historicalBeforeActivation: false, methodologyGeneration: "METHOD_V2", analyticalAuthoritySetSha256: METHOD_V2_AUTHORITY_SET_SHA256 }), "METHOD_V2");
});
test("M07 missing member fails closed even with recomputed candidate hash", async () => {
  const candidate = structuredClone(methodV2Manifest); candidate.members.pop(); candidate.authority_set_sha256 = authoritySetSha256(candidate);
  await assert.rejects(verifyMethodV2Authority(candidate, fetchLocal), /MANIFEST_MISMATCH/);
});
test("M08 changed member hash or exact bytes fails closed", async () => {
  const candidate = structuredClone(methodV2Manifest); candidate.members[0].pin.content_sha256 = "0".repeat(64);
  await assert.rejects(verifyMethodV2Authority(candidate, fetchLocal), /MANIFEST_MISMATCH/);
  await assert.rejects(verifyMethodV2Authority(methodV2Manifest, async request => request.path === methodV2Manifest.members[0].pin.locator.path ? Buffer.from("tampered") : fetchLocal(request)), /MEMBER_BYTES_MISMATCH/);
});
test("M09 extra analytical authority fails closed", async () => {
  const candidate = structuredClone(methodV2Manifest); candidate.members.push({ ...candidate.members[0], id: "unauthorized" });
  await assert.rejects(verifyMethodV2Authority(candidate, fetchLocal), /MANIFEST_MISMATCH/);
});
test("M10 deterministic scoring draft is explicitly excluded", () => {
  const path = "docs/orotitan-equity/OROTITAN_DETERMINISTIC_SCORING_ENGINE_VNEXT_DRAFT_V0.1.md";
  assert.ok(methodV2Manifest.excluded.includes(path));
  assert.ok(methodV2Manifest.members.every(m => m.pin.locator.path !== path));
});
test("M11 scoring implementation and terminal conjunction are byte-unchanged", () => {
  assert.equal(sha256(readFileSync("lib/orotitan-equity/v1/scoring.ts")), "241761d462903d9ea0209753a66d6d0cb34fce517cfbe52ea82174bda18367bf");
  assert.equal(sha256(readFileSync("lib/orotitan-equity/v1/terminal-gate.ts")), "3f2224b8d78026ea635f8110d60a9670553d4cf5baeda20c3e54c6a8da9410b3");
});
test("M12 Certification authority and implementation are unchanged", () => {
  assert.deepEqual(methodV2Manifest.precedence.certification, ["v1_analysis_standard", "v1_deep_dive_stage"]);
  assert.equal(sha256(readFileSync("lib/orotitan-equity/v1/certification.ts")), "cb82c56b57eb1b94f82637721c2138d70d0841ca755ce1e7497c55fc69a25bac");
});
test("M13 absent Challenge blocks Certification", () => assert.throws(() => admitMethodV2Certification(fixture().context), /CHALLENGE_ABSENT/));
test("M14 REOPEN blocks despite positive questions", () => assert.throws(() => fixture("REOPEN").admit(), /CHALLENGE_REOPEN/));
test("M15 FAIL blocks despite positive questions", () => assert.throws(() => fixture("FAIL").admit(), /CHALLENGE_FAIL/));
test("M16 current saturated PASS admits Certification only", () => assert.deepEqual(fixture().admit(), { allowed: true, limitations: [] }));
test("M17 PASS_WITH_CONCERNS propagates exact limitations", () => {
  const f = fixture("CONCERN"); const result = f.admit();
  assert.deepEqual(result.limitations, [{ question_id: "Q0", decision_impact: f.ledger.questions[0].decision_impact, mitigation_or_resolution: f.ledger.questions[0].mitigation_or_resolution }]);
  f.report.material_concerns = []; assert.throws(f.admit, /CONCERNS_NOT_PROPAGATED/);
});
test("M18 question count is telemetry: below old quota succeeds", () => {
  const f = fixture(); assert.ok(f.ledger.questions.length < 100); assert.equal(f.admit().allowed, true);
});
test("M19 material saturation and independent final passes are mandatory", () => {
  const f = fixture(); f.report.saturation_record.material_questions_remaining = true; assert.throws(f.admit, /NOT_SATURATED/);
  f.report.saturation_record.material_questions_remaining = false; f.report.saturation_record.final_generation_passes[1].new_material_questions = 1;
  assert.throws(f.admit, /SATURATION_PASSES_INVALID/);
  f.report.saturation_record.final_generation_passes = []; assert.throws(f.admit, /SATURATION_PASSES_INVALID/);
});
test("M20 historical runs never silently rebound, including explicit V2 attempts", () => {
  const run = Object.freeze({ ...historical(), methodologyGeneration: "METHOD_V2", analyticalAuthoritySetSha256: METHOD_V2_AUTHORITY_SET_SHA256 });
  assert.throws(() => classifyMethodGeneration(run), /HISTORICAL_AUTHORITY_REBIND_FORBIDDEN/);
  assert.equal(run.methodologyGeneration, "METHOD_V2");
});
test("M21 snapshot writers and all existing migrations remain byte-unchanged", () => {
  const files = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap(e => e.isDirectory() ? files(`${dir}/${e.name}`) : [`${dir}/${e.name}`]);
  const paths = ["migrations", "lib/orotitan-equity/v1", "lib/orotitan-equity/v2", "lib/orotitan-equity/v3"].flatMap(files)
    // Authorized forward migration and independently byte-verified restored production predecessor.
    .filter(p => p !== "migrations/20261005195139_orotitan_registry_v1_13_method_generation.sql"
      && p !== "migrations/20260923_orotitan_registry_v1_12_valuation_date_alignment_successor.sql").sort();
  // Independently computed from all 49 exact source blobs at bd673176...;
  // usable in CI's shallow checkout without fetching historical Git objects.
  assert.equal(sha256(paths.map(p => `${p}|${sha256(readFileSync(p))}`).join("\n") + "\n"), "d634f76852bed441caffb95c8377399b1485e0619acd62e792268b1846e731a4");
});
test("M22 authority/admission checks are offline and have no production mutation surface", async () => {
  const context = fixture(); context.admit();
  // Hash preparation modifies only fixture refs; the validator itself is pure.
  const bytes = context.bytes(); const frozenContext = structuredClone(context.context);
  admitMethodV2Certification(context.context, bytes.ledgerBytes, bytes.reportBytes);
  assert.deepEqual(context.context, frozenContext);
  for (const path of ["authority.ts", "pre-certification.ts"]) {
    const source = readFileSync(`lib/orotitan-equity/method-v2/${path}`, "utf8");
    assert.doesNotMatch(source, /@supabase|node:fs|node:child_process|fetch\(|process\.env/);
  }
  assert.equal(methodV2Manifest.production_active, false);
});
test("adversarial stale lineage, hash, cutoff, coverage and mixed generation reject", () => {
  const f = fixture(); const bytes = f.bytes();
  assert.throws(() => admitMethodV2Certification(f.context, Buffer.from("{}"), bytes.reportBytes), /HASH_MISMATCH/);
  f.context.valuation_lock_ref.version = 2; assert.throws(f.admit, /STALE_LINEAGE/);
  const late = fixture(); late.ledger.questions[0].evidence_dates = ["2026-10-02"]; assert.throws(late.admit, /POST_CUTOFF/);
  const coverage = fixture(); delete coverage.report.saturation_record.coverage_rationale.WORLD_IN_MOTION; assert.throws(coverage.admit, /NOT_SATURATED/);
  assert.throws(() => classifyMethodGeneration({ historicalBeforeActivation: false, methodologyGeneration: "METHOD_V1", analyticalAuthoritySetSha256: METHOD_V2_AUTHORITY_SET_SHA256 }), /MISSING_OR_MIXED/);
});
test("DELTA requires exact prior passing provenance and preserves concern propagation", () => {
  const f = fixture("CONCERN");
  const delta = { prior_passing_report_ref: ref(6), prior_passing_ledger_ref: ref(7), impact_map: "Revalidate supplier change and all dependent valuation assumptions", inherited_coverage_revalidated: true };
  Object.assign(f.ledger, { mode: "DELTA", delta_provenance: delta });
  Object.assign(f.report, { mode: "DELTA", delta_provenance: delta });
  assert.throws(f.admit, /DELTA_PROVENANCE_INVALID/);
  f.context.priorPassing = { report_ref: ref(6), ledger_ref: ref(7) };
  assert.equal(f.admit().limitations[0].question_id, "Q0");
  f.context.priorPassing.report_ref.version = 2;
  assert.throws(f.admit, /DELTA_PROVENANCE_INVALID/);
});
test("execution support is independently byte-pinned and excluded from analytical membership", async () => {
  const support = JSON.parse(readFileSync("contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_EXECUTION_SUPPORT.json", "utf8")) as {
    identity_domain: string; members: { path?: string; sha256?: string; pin?: ContractPin }[];
  };
  assert.equal(support.identity_domain, "EXECUTION_SUPPORT_NOT_ANALYTICAL_IDENTITY");
  for (const member of support.members) {
    if (member.path) {
      assert.equal(sha256(readFileSync(member.path)), member.sha256);
      assert.ok(methodV2Manifest.members.every(m => m.pin.locator.path !== member.path));
    } else await resolveContractPin(member.pin!, fetchLocal);
  }
});

function evidencedExemption(): FamilyCoverage {
  return { disposition: "EVIDENCED_NO_MATERIAL_CHALLENGE",
    rationale: "Cutoff Evidence Ledger records show no material exposure for this company's decision",
    evidence_references: [{ evidence_id: "E1", evidence_ledger_ref: ref(8), evidence_date: "2026-09-30" }] };
}

test("V1.1 rejects empty ledger with self-attested coverage, with or without exemptions", () => {
  const f = fixture(); f.ledger.questions = []; f.report.company_specific_questions = [];
  f.report.total_questions_executed = 0; f.report.total_candidate_questions_generated = 0; f.report.pass_count = 0;
  Reflect.deleteProperty(f.report.saturation_record, "family_coverage");
  assert.throws(f.admit, /SCHEMA_INVALID/);
  f.report.saturation_record.family_coverage = Object.fromEntries(CHALLENGE_FAMILIES.map(family => [family, evidencedExemption()])) as typeof f.report.saturation_record.family_coverage;
  assert.throws(f.admit, /COMPANY_COVERAGE_INSUFFICIENT/);
});

test("V1.1 rejects rationale without structured evidence or evidence dates", () => {
  for (const missing of ["evidence_references", "evidence_date", "evidence_ledger_ref"]) {
    const f = fixture(); const exemption = evidencedExemption();
    assert.equal(exemption.disposition, "EVIDENCED_NO_MATERIAL_CHALLENGE");
    if (exemption.disposition !== "EVIDENCED_NO_MATERIAL_CHALLENGE") throw new Error("fixture");
    if (missing === "evidence_references") Reflect.deleteProperty(exemption, missing);
    else Reflect.deleteProperty(exemption.evidence_references[0], missing);
    f.report.saturation_record.family_coverage.WORLD_IN_MOTION = exemption;
    assert.throws(f.admit, /SCHEMA_INVALID/);
  }
  const f = fixture();
  f.report.saturation_record.family_coverage.WORLD_IN_MOTION = {
    disposition: "EVIDENCED_NO_MATERIAL_CHALLENGE", rationale: "Arbitrary assertion", evidence_references: []
  };
  assert.throws(f.admit, /SCHEMA_INVALID/);
});

test("V1.1 family coverage rejects unknown, dropped, wrong-family and generic questions", () => {
  for (const id of ["UNKNOWN", "DROPPED", "Q1"]) {
    const f = fixture();
    if (id === "DROPPED") {
      f.ledger.dropped_candidates.push({ question_id: id, reason: "DROP_DUPLICATE", rationale: "Already covered" });
      f.report.total_candidate_questions_generated++; f.report.dropped_duplicate_count++;
    }
    f.report.saturation_record.family_coverage.CROSS_BLOCK_CONSISTENCY = { disposition: "QUESTION_COVERED", question_ids: [id] };
    assert.throws(f.admit, /FAMILY_COVERAGE_INVALID/);
  }
  const f = fixture(); f.ledger.questions[0].company_specific = false;
  f.report.company_specific_questions = f.report.company_specific_questions.filter(id => id !== "Q0");
  assert.throws(f.admit, /FAMILY_COVERAGE_INVALID/);
});

test("V1.1 requires an executed company-specific question even with all families exempted", () => {
  const f = fixture(); f.ledger.questions.forEach(q => { q.company_specific = false; });
  f.report.company_specific_questions = [];
  CHALLENGE_FAMILIES.forEach(family => { f.report.saturation_record.family_coverage[family] = evidencedExemption(); });
  assert.throws(f.admit, /COMPANY_COVERAGE_INSUFFICIENT/);
});

test("V1.1 rejects post-cutoff exemption evidence", () => {
  const f = fixture(); const exemption = evidencedExemption();
  if (exemption.disposition !== "EVIDENCED_NO_MATERIAL_CHALLENGE") throw new Error("fixture");
  exemption.evidence_references[0].evidence_date = "2026-10-02";
  f.report.saturation_record.family_coverage.WORLD_IN_MOTION = exemption;
  assert.throws(f.admit, /POST_CUTOFF_EVIDENCE/);
});

test("V1.1 accepts question-backed coverage and evidenced exemption without a family question", () => {
  assert.equal(fixture().admit().allowed, true);
  const f = fixture(); f.ledger.questions = f.ledger.questions.filter(q => q.question_family !== "WORLD_IN_MOTION");
  f.report.total_questions_executed--; f.report.total_candidate_questions_generated--; f.report.pass_count--;
  f.report.company_specific_questions = f.ledger.questions.map(q => q.question_id);
  f.report.saturation_record.family_coverage.WORLD_IN_MOTION = evidencedExemption();
  assert.equal(f.admit().allowed, true);
});

test("V1.1 rejects missing, extra or ambiguous family dispositions", () => {
  const missing = fixture(); Reflect.deleteProperty(missing.report.saturation_record.family_coverage, "WORLD_IN_MOTION");
  assert.throws(missing.admit, /SCHEMA_INVALID/);
  const extra = fixture(); Object.assign(extra.report.saturation_record.family_coverage, { UNKNOWN: evidencedExemption() });
  assert.throws(extra.admit, /SCHEMA_INVALID/);
  const ambiguous = fixture(); Object.assign(ambiguous.report.saturation_record.family_coverage.WORLD_IN_MOTION, evidencedExemption());
  assert.throws(ambiguous.admit, /SCHEMA_INVALID/);
});

test("V1.1 PASS and PASS_WITH_CONCERNS reject either retained negative disposition", () => {
  for (const status of ["PASS", "CONCERN"] as const) {
    for (const field of ["reopen_required", "fail_reasons"] as const) {
      const f = fixture(status); f.report[field] = ["Unresolved valuation defect"];
      assert.throws(f.admit, /NEGATIVE_DISPOSITION_UNRESOLVED/);
      f.report[field] = []; assert.equal(f.admit().allowed, true);
    }
  }
});

test("V1.1 cannot admit V1.0 artifacts under the successor authority identity", () => {
  const f = fixture(); Object.assign(f.ledger, { schema_version: "1.0", challenge_version: "1.0" });
  Object.assign(f.report, { schema_version: "1.0", challenge_version: "1.0" });
  assert.throws(f.admit, /SCHEMA_INVALID/);
});

test("V1.1 retains exact historical V1.0 authority and explicitly scopes successor precedence", async () => {
  const root = "contracts/orotitan-equity/method-v2/";
  const historicalManifest = JSON.parse(readFileSync(`${root}OROTITAN_METHOD_V2_AUTHORITY_MANIFEST.json`, "utf8"));
  assert.equal(authoritySetSha256(historicalManifest), METHOD_V2_V1_0_AUTHORITY_SET_SHA256);
  assert.equal(sha256(readFileSync(`${root}OROTITAN_PRE_CERTIFICATION_CHALLENGE_FREEZE_V1.0.md`)), "6190078e532626d80eaf994e806d4461eece3a7daa3f62cbdd0e908b1f2a654a");
  assert.equal(methodV2Manifest.predecessor_authority_set_sha256, METHOD_V2_V1_0_AUTHORITY_SET_SHA256);
  assert.deepEqual(methodV2Manifest.members.slice(0, historicalManifest.members.length), historicalManifest.members);
  assert.equal(methodV2Manifest.members.length, 18);
  assert.deepEqual(methodV2Manifest.precedence.pre_certification_admission, ["pre_certification_coverage_evidence", "pre_certification_challenge", "v1_deep_dive_stage"]);
  assert.deepEqual(methodV2Manifest.precedence.pre_certification_ledger_schema, ["challenge_ledger_schema_v1_1", "challenge_ledger_schema"]);
  assert.deepEqual(methodV2Manifest.precedence.pre_certification_report_schema, ["challenge_report_schema_v1_1", "challenge_report_schema"]);
  assert.equal(methodV2Manifest.members.find(m => m.id === "pre_certification_coverage_evidence")!.pin.content_sha256, PRE_CERTIFICATION_EFFECTIVE_SHA256);
  await assert.rejects(verifyMethodV2Authority(historicalManifest, fetchLocal), /MANIFEST_MISMATCH/);
  const f = fixture(); f.context.analyticalAuthoritySetSha256 = METHOD_V2_V1_0_AUTHORITY_SET_SHA256;
  assert.throws(f.admit, /METHOD_AUTHORITY_MISMATCH/);
});
