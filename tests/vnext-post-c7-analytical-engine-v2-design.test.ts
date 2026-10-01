import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("post-C7 product direction selects Analytical Engine V2 without reopening Phase C", () => {
  const d = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_POST_C7_PRODUCT_DIRECTION_ANALYTICAL_ENGINE_V2_001.json",
      "utf8",
    ),
  );
  const s = JSON.parse(
    readFileSync("docs/continuity/CURRENT_STATE.json", "utf8"),
  );

  assert.equal(
    d.status,
    "ANALYTICAL_ENGINE_V2_SELECTED_AS_POST_C7_DIRECTION",
  );
  assert.deepEqual(d.priority_order, [
    "ANALYTICAL_QUALITY",
    "PROCESS_CORRECTNESS_AND_FLUIDITY",
    "UI_UX_QUALITY",
    "AUTOMATION_RATE",
  ]);
  assert.equal(d.constraints.no_production_mutation, true);
  assert.equal(d.constraints.no_routing_freeze, true);
  assert.equal(d.constraints.no_model_winner_selected, true);
  assert.equal(d.constraints.no_provider_lock_in, true);

  assert.equal(s.phase_c_status, "COMPLETE");
  assert.equal(s.c7_local_candidate_decision_status, "LOCAL_CANDIDATE_REJECTED");
  assert.equal(
    s.post_c7_product_direction_status,
    "ANALYTICAL_ENGINE_V2_SELECTED_AS_POST_C7_DIRECTION",
  );
  assert.equal(s.model_winner_selected, false);
  assert.equal(s.routing_frozen, false);
  assert.equal(s.production_mutation, false);
  assert.equal(s.architectural_planning_phase_status, "CLOSED");
});

test("Analytical Engine V2 design makes sector, cycle, technology and anti-loop execution first-class", () => {
  const raw = readFileSync(
    "docs/orotitan-equity/OROTITAN_ANALYTICAL_ENGINE_V2_DESIGN_V0.1.md",
    "utf8",
  );

  assert.match(raw, /# 5\. COMPANY ECONOMIC DNA/);
  assert.match(raw, /# 7\. SECTOR INTELLIGENCE LIBRARY/);
  assert.match(raw, /# 8\. CYCLICALITY ENGINE/);
  assert.match(raw, /# 9\. TECHNOLOGY ENGINE/);
  assert.match(raw, /# 10\. INDUSTRY STRUCTURE ENGINE/);
  assert.match(raw, /# 11\. CAUSAL GRAPH ENGINE/);
  assert.match(raw, /# 25\. ANTI-LOOP PROCESS ENGINE RULES/);
  assert.match(raw, /# 27\. CHATGPT DIRECT INFRASTRUCTURE PROTOCOL/);
  assert.match(raw, /# 33\. UI \/ UX DESIGN PRINCIPLES/);
  assert.match(raw, /# 40\. IMPLEMENTATION PROGRAM/);
  assert.match(raw, /QUALITY BEFORE AUTOMATION/);
  assert.match(raw, /No production mutation is authorized by this document/);
});

test("post-C7 direction includes French-first workbench and core product functions", () => {
  const d = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_POST_C7_PRODUCT_DIRECTION_ANALYTICAL_ENGINE_V2_001.json",
      "utf8",
    ),
  );

  assert.equal(d.product_requirements.french_first_ui, true);
  assert.equal(d.product_requirements.research_queue_shortlist, true);
  assert.equal(d.product_requirements.stage_and_block_progress_visualization, true);
  assert.equal(d.product_requirements.blockers_and_next_action, true);
  assert.equal(d.product_requirements.market_price_refresh, true);
  assert.equal(d.product_requirements.export_pdf, true);
  assert.equal(d.product_requirements.export_markdown, true);
  assert.equal(d.product_requirements.export_json, true);
  assert.equal(d.product_requirements.export_csv, true);
  assert.equal(d.product_requirements.company_analysis_map, true);
  assert.equal(d.product_requirements.open_questions, true);
  assert.equal(d.product_requirements.evidence_browser, true);
  assert.equal(d.product_requirements.history_and_monitoring, true);
});
