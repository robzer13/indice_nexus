import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import profile from "../contracts/orotitan-equity/method-v2/runtime/OROTITAN_METHOD_V2_CERTIFICATION_PERSISTENCE_PROFILE_V1.0.json";
import {
  METHOD_V2_CERTIFICATION_FORMAT, METHOD_V2_CERTIFICATION_VERSION,
  parseMethodV2CertificationArtifact, serializeMethodV2CertificationArtifact,
  verifyMethodV2CertificationChallengeBinding,
  type MethodV2CertificationArtifact, type MethodV2CertificationExpectedContext,
} from "../lib/orotitan-equity/method-v2/certification-persistence";
import { methodV2Manifest } from "../lib/orotitan-equity/method-v2/authority";
import { resolveContractPin, type ContractPin } from "../lib/orotitan-equity/v1/contract-pin-resolver";

const PROFILE_PATH = "contracts/orotitan-equity/method-v2/runtime/OROTITAN_METHOD_V2_CERTIFICATION_PERSISTENCE_PROFILE_V1.0.json";
const PROFILE_SHA256 = "a9b3eff1930a9a7e6164cfb1f0248c98c8041975ef9adc77f9d152062df0a53c";
const ref = (n: number) => ({ artifact_id: `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`, version: 1, content_sha256: String(n).repeat(64) });
function fixture(concerns = false): { artifact: MethodV2CertificationArtifact; context: MethodV2CertificationExpectedContext } {
  const context: MethodV2CertificationExpectedContext = {
    run_id: ref(1).artifact_id, stage_revision: 1, data_cutoff: "2026-10-01",
    question_ledger_ref: ref(2), challenge_report_ref: ref(3), fundamentals_lock_ref: ref(4), valuation_lock_ref: ref(5),
    challenge_status: concerns ? "PASS_WITH_CONCERNS" : "PASS",
    challenge_limitations: concerns ? [
      { question_id: "Q1", decision_impact: " Exact impact é\n", mitigation_or_resolution: "Bounded residual limitation" },
      { question_id: "Q2", decision_impact: "Another decision impact", mitigation_or_resolution: " Retain exact mitigation " },
    ] : [],
  };
  const { run_id, stage_revision, data_cutoff, ...binding } = structuredClone(context);
  return { context, artifact: { format: METHOD_V2_CERTIFICATION_FORMAT, version: METHOD_V2_CERTIFICATION_VERSION,
    run_id, stage_revision, data_cutoff, certification: { inherited_content: "opaque" }, method_v2_challenge_binding: binding } };
}
const bytes = (value: unknown) => Buffer.from(JSON.stringify(value), "utf8");
const verify = (artifact: MethodV2CertificationArtifact, context: MethodV2CertificationExpectedContext) =>
  verifyMethodV2CertificationChallengeBinding(parseMethodV2CertificationArtifact(bytes(artifact)), context);

test("A correctly serialized PASS", () => {
  const { artifact, context } = fixture();
  assert.deepEqual(verifyMethodV2CertificationChallengeBinding(parseMethodV2CertificationArtifact(serializeMethodV2CertificationArtifact(artifact)), context), { verified: true });
});
test("B PASS_WITH_CONCERNS with multiple exact concerns", () => {
  const { artifact, context } = fixture(true);
  assert.deepEqual(verify(artifact, context), { verified: true });
});
for (const [id, member, replacement] of [
  ["C", "artifact_id", ref(9).artifact_id], ["D", "version", 2], ["E", "content_sha256", "a".repeat(64)],
] as const) test(`${id} wrong Challenge Report ${member}`, () => {
  const { artifact, context } = fixture();
  Object.assign(artifact.method_v2_challenge_binding.challenge_report_ref, { [member]: replacement });
  assert.throws(() => verify(artifact, context), /ARTIFACT_REF_MISMATCH/);
});
for (const [id, key] of [["F", "question_ledger_ref"], ["G", "fundamentals_lock_ref"], ["H", "valuation_lock_ref"]] as const) {
  for (const member of ["artifact_id", "version", "content_sha256"] as const) test(`${id} wrong ${key}.${member}`, () => {
    const { artifact, context } = fixture();
    Object.assign(artifact.method_v2_challenge_binding[key], { [member]: { artifact_id: ref(9).artifact_id, version: 2, content_sha256: "b".repeat(64) }[member] });
    assert.throws(() => verify(artifact, context), /ARTIFACT_REF_MISMATCH/);
  });
}
for (const [id, key, value] of [["I", "run_id", ref(9).artifact_id], ["J", "stage_revision", 2], ["K", "data_cutoff", "2026-10-02"]] as const) test(`${id} wrong ${key}`, () => {
  const { artifact, context } = fixture(); Object.assign(artifact, { [key]: value });
  assert.throws(() => verify(artifact, context), /CONTEXT_MISMATCH/);
});
test("L PASS with nonempty limitations", () => {
  const { artifact, context } = fixture(true); artifact.method_v2_challenge_binding.challenge_status = "PASS";
  assert.throws(() => verify(artifact, context), /PASS_LIMITATIONS_NOT_EMPTY/);
});
test("M missing one concern", () => {
  const { artifact, context } = fixture(true); artifact.method_v2_challenge_binding.challenge_limitations.pop();
  assert.throws(() => verify(artifact, context), /CHALLENGE_LIMITATIONS_MISMATCH/);
});
test("N extra concern", () => {
  const { artifact, context } = fixture(true); artifact.method_v2_challenge_binding.challenge_limitations.push({ question_id: "Q3", decision_impact: "extra", mitigation_or_resolution: "extra" });
  assert.throws(() => verify(artifact, context), /CHALLENGE_LIMITATIONS_MISMATCH/);
});
for (const [id, key] of [["O", "question_id"], ["P", "decision_impact"], ["Q", "mitigation_or_resolution"]] as const) test(`${id} wrong concern ${key}`, () => {
  const { artifact, context } = fixture(true); artifact.method_v2_challenge_binding.challenge_limitations[0][key] += " ";
  assert.throws(() => verify(artifact, context), /CHALLENGE_LIMITATIONS_MISMATCH/);
});
test("R duplicate question_id", () => {
  const { artifact, context } = fixture(true); artifact.method_v2_challenge_binding.challenge_limitations[1].question_id = "Q1";
  assert.throws(() => verify(artifact, context), /DUPLICATE_QUESTION_ID/);
});
test("S concern order has no meaning and deterministic writer emits identical bytes", () => {
  const { artifact, context } = fixture(true); const original = serializeMethodV2CertificationArtifact(artifact);
  artifact.method_v2_challenge_binding.challenge_limitations.reverse();
  assert.deepEqual(verify(artifact, context), { verified: true });
  assert.deepEqual(serializeMethodV2CertificationArtifact(artifact), original);
  assert.equal(artifact.method_v2_challenge_binding.challenge_limitations[0].question_id, "Q2", "writer does not mutate input");
});
test("T invalid UTF-8, malformed JSON, roots, BOM and trailing data", () => {
  for (const raw of [Buffer.from([0xc3, 0x28]), Buffer.from([0xff]), Buffer.from([0xc0, 0xaf])]) {
    assert.throws(() => parseMethodV2CertificationArtifact(raw), /INVALID_UTF8/);
  }
  for (const raw of ['{', '{"x":1,}', '[1,]', 'true false', '{"x":"\\z"}', '{"x":01}', '{"x":"\n"}', '\ufeff{}']) {
    assert.throws(() => parseMethodV2CertificationArtifact(Buffer.from(raw)), /INVALID_JSON/);
  }
  for (const root of [null, [], "text", 1, true]) assert.throws(() => parseMethodV2CertificationArtifact(bytes(root)), /INVALID_ROOT/);
});
test("U every identity-bearing member rejects duplicates including escaped names", () => {
  const { artifact } = fixture(true);
  const root = bytes(artifact).toString();
  const binding = artifact.method_v2_challenge_binding;
  const refValue = binding.challenge_report_ref;
  const limitation = binding.challenge_limitations[0];
  const targets = [
    ...Object.entries(artifact), ...Object.entries(binding), ...Object.entries(refValue), ...Object.entries(limitation),
  ];
  for (const [key, value] of targets) {
    const member = `${JSON.stringify(key)}:${JSON.stringify(value)}`;
    const escapedKey = `"\\u${key.charCodeAt(0).toString(16).padStart(4, "0")}${key.slice(1)}"`;
    for (const duplicate of [member, `${escapedKey}:${JSON.stringify(value)}`]) {
      assert.throws(() => parseMethodV2CertificationArtifact(Buffer.from(root.replace(member, `${member},${duplicate}`))), /DUPLICATE_JSON_MEMBER/, key);
    }
  }
});
test("V additional inherited Certification content is opaque and does not change binding identity", () => {
  const { artifact, context } = fixture(true);
  artifact.certification = { arbitrary_decision: "not interpreted", material_limitations: [{ code: "legacy", rationale: "separately governed" }],
    nested: { run_id: "unrelated", method_v2_challenge_binding: { challenge_status: "FAIL" } }, score_permission: false };
  assert.deepEqual(verify(artifact, context), { verified: true });
});
test("wrong format/version, malformed identity and payload reject", () => {
  for (const patch of [{ format: "wrong" }, { version: "2.0" }, { run_id: "run" }, { stage_revision: 0 }, { stage_revision: 1.5 },
    { stage_revision: Number.MAX_SAFE_INTEGER + 1 }, { data_cutoff: "2026-02-29" }, { data_cutoff: "2026-13-01" },
    { data_cutoff: "2026-10-00" }, { data_cutoff: "2026-1-01" }, { certification: [] }, { extra_root: true }]) {
    assert.throws(() => parseMethodV2CertificationArtifact(bytes({ ...fixture().artifact, ...patch })), /INVALID_/);
  }
  const { artifact, context } = fixture(); artifact.data_cutoff = context.data_cutoff = "2024-02-29";
  assert.deepEqual(verify(artifact, context), { verified: true });
});
test("ArtifactRefs have exactly three members, valid UUID, integer and lowercase SHA", () => {
  for (const patch of [{ artifact_id: "bad" }, { version: "1" }, { version: 0 }, { version: 1.1 }, { content_sha256: "A".repeat(64) },
    { content_sha256: "a".repeat(63) }, { alias: true }]) {
    const { artifact } = fixture(); Object.assign(artifact.method_v2_challenge_binding.challenge_report_ref, patch);
    assert.throws(() => parseMethodV2CertificationArtifact(bytes(artifact)), /INVALID_ARTIFACT_REF/);
  }
  const { artifact } = fixture(); delete (artifact.method_v2_challenge_binding.challenge_report_ref as Partial<typeof artifact.method_v2_challenge_binding.challenge_report_ref>).version;
  assert.throws(() => parseMethodV2CertificationArtifact(bytes(artifact)), /INVALID_ARTIFACT_REF/);
});
test("malformed binding/status/limitations reject", () => {
  for (const patch of [{ challenge_status: "FAIL" }, { challenge_status: "REOPEN" }, { challenge_status: "pass" },
    { challenge_limitations: {} }, { challenge_limitations: [] }, { challenge_limitations: [null] },
    { challenge_limitations: [{ question_id: "Q", decision_impact: "", mitigation_or_resolution: "m" }] },
    { challenge_limitations: [{ question_id: "Q", decision_impact: "i", mitigation_or_resolution: null }] },
    { challenge_limitations: [{ question_id: "Q", decision_impact: "i", mitigation_or_resolution: "m", extra: true }] }, { alias: true }]) {
    const { artifact } = fixture(true); Object.assign(artifact.method_v2_challenge_binding, patch);
    assert.throws(() => parseMethodV2CertificationArtifact(bytes(artifact)), /INVALID_|CONCERNS_ABSENT/);
  }
});
test("status mismatch rejects even when both documents are independently well formed", () => {
  const { artifact } = fixture(true); const { context } = fixture();
  assert.throws(() => verify(artifact, context), /CHALLENGE_STATUS_MISMATCH/);
});
test("verifier revalidates mutable objects and expected context", () => {
  const { artifact, context } = fixture(true);
  context.challenge_limitations.push({ ...context.challenge_limitations[0] });
  assert.throws(() => verify(artifact, context), /DUPLICATE_QUESTION_ID/);
  artifact.stage_revision = 0;
  assert.throws(() => verifyMethodV2CertificationChallengeBinding(artifact, fixture().context), /INVALID_STAGE_REVISION/);
});
test("duplicate inherited members reject structurally; strings and separate nested keys pass", () => {
  const { artifact, context } = fixture();
  artifact.certification = { note: 'text "run_id":1, "run_id":2 \\ slash', nested: { run_id: "opaque" } };
  assert.deepEqual(verify(artifact, context), { verified: true });
  const raw = bytes(artifact).toString().replace('"certification":{', '"certification":{"same":1,"same":2,');
  assert.throws(() => parseMethodV2CertificationArtifact(Buffer.from(raw)), /DUPLICATE_JSON_MEMBER/);
});
test("deterministic writer sorts nested keys, preserves inherited arrays and Unicode, rejects non-JSON values", () => {
  const { artifact } = fixture(true); artifact.certification = { z: [2, 1], a: { z: 0.5, a: "é" } };
  const serialized = serializeMethodV2CertificationArtifact(artifact);
  const alternative = structuredClone(artifact); alternative.certification = { a: { a: "é", z: 0.5 }, z: [2, 1] };
  assert.deepEqual(serializeMethodV2CertificationArtifact(alternative), serialized);
  assert.deepEqual(parseMethodV2CertificationArtifact(serialized).certification, artifact.certification);
  assert.ok(!Buffer.from(serialized).toString().endsWith("\n"));
  for (const value of [undefined, Infinity, NaN, () => 1, new Date()]) {
    artifact.certification = { value };
    assert.throws(() => serializeMethodV2CertificationArtifact(artifact), /NON_JSON_SERIALIZATION_VALUE/);
  }
});
test("profile raw-byte identity and every semantic/scope boundary are frozen", () => {
  assert.equal(createHash("sha256").update(readFileSync(PROFILE_PATH)).digest("hex"), PROFILE_SHA256);
  assert.equal(profile.identity_domain, "METHOD_V2_RUNTIME_SERIALIZATION_NOT_ANALYTICAL_AUTHORITY");
  assert.equal(profile.production_active, false);
  assert.equal(profile.artifact_type, "CERTIFICATION_ARTIFACT");
  assert.equal(profile.applies_to.artifact_lifecycle, "NEW_ARTIFACTS_ONLY");
  assert.ok(Object.values(profile.semantic_boundary).every(value => value === false));
  assert.ok(Object.values(profile.scope_boundary).every(value => value === false));
  assert.ok(methodV2Manifest.members.every(member => member.pin.locator.path !== PROFILE_PATH));
});
test("inherited V1 Deep Dive physical layout is explicitly an implementation detail", async () => {
  const member = methodV2Manifest.members.find(item => item.id === "v1_deep_dive_stage")!;
  const raw = await resolveContractPin(member.pin as ContractPin, async request => readFileSync(request.path));
  const text = Buffer.from(raw).toString("utf8");
  assert.match(text, /Exact physical layout is an implementation detail\./);
});
