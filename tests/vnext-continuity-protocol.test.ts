import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("VNExT continuity state exposes the minimum resume contract", () => {
  const state = JSON.parse(
    readFileSync("docs/continuity/CURRENT_STATE.json", "utf8"),
  );

  for (const key of [
    "protocol_version",
    "resume_id",
    "project",
    "repository",
    "branch",
    "last_verified_head_sha",
    "last_verified_at",
    "current_gate",
    "current_phase",
    "status",
    "objective",
    "completed",
    "active_work",
    "blockers",
    "open_issues",
    "preserved_decisions",
    "authorized_actions",
    "forbidden_actions",
    "authoritative_artifacts",
    "next_action",
    "resume_sequence",
  ]) {
    assert.ok(Object.hasOwn(state, key), `missing continuity key: ${key}`);
  }

  assert.equal(state.project, "OROTITAN_VNEXT");
  assert.equal(state.repository, "robzer13/indice_nexus");
  assert.equal(state.branch, "vnext");
  assert.equal(state.current_gate, 18);
  assert.equal(state.status, "IN_PROGRESS_NOT_FROZEN");
  assert.equal(
    state.next_action,
    "RUN_V1_1_SHADOW_REPLAY_ON_EXISTING_PRIVATE_ARTIFACTS_NO_INFERENCE",
  );
});

test("continuity standing authorization cannot authorize analytical or production work", () => {
  const raw = readFileSync(
    "docs/continuity/OROTITAN_CONTINUITY_STANDING_AUTH_V1.md",
    "utf8",
  );

  for (const prohibited of [
    "analytical methodology changes",
    "frozen contract changes",
    "new model inference",
    "production mutation",
    "publication",
    "retroactive pass",
  ]) {
    assert.match(raw, new RegExp(prohibited, "i"));
  }
});
