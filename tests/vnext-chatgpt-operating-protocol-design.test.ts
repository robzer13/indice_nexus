import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("ChatGPT operating protocol keeps intelligence in ChatGPT and infrastructure bounded", () => {
  const raw = readFileSync(
    "docs/orotitan-equity/OROTITAN_CHATGPT_OPERATING_PROTOCOL_V0.1.md",
    "utf8",
  );

  assert.match(raw, /ChatGPT is allowed to read directly from GitHub, Supabase and Vercel/);
  assert.match(raw, /FREE ANALYTICAL REASONING/);
  assert.match(raw, /CHAT MEMORY\n= CONVENIENCE/);
  assert.match(raw, /PERSISTED ARTIFACT\n= AUTHORITY/);
  assert.match(raw, /LOAD OROTITAN <COMPANY>/);
  assert.match(raw, /CHECKPOINT OROTITAN/);
  assert.match(raw, /SAVE OROTITAN/);
  assert.match(raw, /GO PUBLISH <COMPANY>/);
  assert.match(raw, /PROJECT-SCOPED/);
  assert.match(raw, /READ-ONLY/);
  assert.match(raw, /PROMPT-INJECTION FIREWALL/);
  assert.match(raw, /MATERIAL CHANGE REVALIDATION GATE/);
  assert.match(raw, /AD-HOC UPDATE \/ INSERT/);
  assert.match(raw, /ChatGPT Work is not part of the required V2 workflow/);
});

test("ChatGPT protocol candidate remains non-frozen and non-publishing", () => {
  const d = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_CHATGPT_OPERATING_PROTOCOL_CANDIDATE_001.json",
      "utf8",
    ),
  );

  assert.equal(d.status, "DESIGN_CANDIDATE_READY_FOR_USER_REVIEW");
  assert.equal(d.architecture.chatgpt, "PRIMARY_NON_DETERMINISTIC_ANALYTICAL_BRAIN");
  assert.equal(d.architecture.chatgpt_work, "OUT_OF_SCOPE_UNLESS_CLEAR_FUTURE_GAIN");
  assert.equal(d.does_not_freeze, true);
  assert.equal(d.no_production_mutation, true);
  assert.equal(d.next_action, "USER_REVIEW_CHATGPT_OPERATING_PROTOCOL_V0_1");
});

test("Analytical Engine V2 points to the direct ChatGPT operating protocol", () => {
  const raw = readFileSync(
    "docs/orotitan-equity/OROTITAN_ANALYTICAL_ENGINE_V2_DESIGN_V0.1.md",
    "utf8",
  );

  assert.match(raw, /OROTITAN_CHATGPT_OPERATING_PROTOCOL_V0\.1\.md/);
  assert.match(raw, /PRIMARY NON-DETERMINISTIC ANALYTICAL ENGINE/);
  assert.match(raw, /manual copy\/paste bridge is no longer the target operating model/i);
  assert.match(raw, /ChatGPT Work is optional and currently out of scope/);
});
