import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const root = new URL('../contracts/orotitan-equity/v4-draft/', import.meta.url);
const checklist = readFileSync(new URL('OROTITAN_PRE_CERTIFICATION_CHECKLIST_SPEC_V4_DRAFT_V0.1.md', root), 'utf8');
const architecture = readFileSync(new URL('OROTITAN_PRE_CERTIFICATION_ARCHITECTURE_IMPACT_V4_DRAFT_V0.1.md', root), 'utf8');
const deepDive = readFileSync(new URL('OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V4_DRAFT_V0.1.md', root), 'utf8');
const ledgerSchema = JSON.parse(readFileSync(new URL('schemas/PRE_CERTIFICATION_QUESTION_LEDGER_V0.1.schema.json', root), 'utf8'));
const reportSchema = JSON.parse(readFileSync(new URL('schemas/PRE_CERTIFICATION_CHECKLIST_REPORT_V0.1.schema.json', root), 'utf8'));

test('V4 draft inserts checklist between valuation and certification without new Registry stage', () => {
  assert.match(deepDive, /PHASE 2 = VALUATION/);
  assert.match(deepDive, /PHASE 2\.5 = PRE_CERTIFICATION_CHECKLIST/);
  assert.match(deepDive, /PHASE 3 = CERTIFICATION_RECONCILIATION/);
  assert.match(architecture, /DO NOT ADD A NEW REGISTRY STAGE_CODE/);
});

test('checklist is non-compensatory and cannot score', () => {
  assert.match(checklist, /No CHECKLIST_SCORE/);
  assert.match(checklist, /REOPEN.*READY_FOR_CERTIFICATION = NO/s);
  assert.match(checklist, /FAIL.*READY_FOR_CERTIFICATION = NO/s);
  assert.match(checklist, /PASS_WITH_CONCERNS.*READY_FOR_CERTIFICATION = YES/s);
});

test('question ledger requires at least 100 executed questions and controlled statuses', () => {
  assert.equal(ledgerSchema.properties.questions.minItems, 100);
  assert.deepEqual(ledgerSchema.properties.questions.items.properties.status.enum, ['PASS','CONCERN','REOPEN','FAIL']);
});

test('report schema enforces checklist status and readiness vocabulary', () => {
  assert.deepEqual(reportSchema.properties.checklist_status.enum, ['PASS','PASS_WITH_CONCERNS','REOPEN','FAIL']);
  assert.deepEqual(reportSchema.properties.ready_for_certification.enum, ['YES','NO']);
  assert.equal(reportSchema.properties.total_questions_executed.minimum, 100);
});

test('historical frozen contracts are explicitly protected', () => {
  assert.match(checklist, /Frozen V2 \/ V3 contracts: UNCHANGED/);
  assert.match(architecture, /DO NOT MODIFY V2\/V3 FROZEN CONTRACTS/);
});
