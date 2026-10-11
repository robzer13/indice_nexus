import assert from 'node:assert/strict';
import test from 'node:test';
import { isAdmissiblePaperValuationQuote } from '../lib/domain/paper-quote-validation';
const now = new Date('2026-10-11T09:00:00Z');
test('paper mark accepts recent valid quotes', () => {
  assert.equal(isAdmissiblePaperValuationQuote(4.5,'2026-10-10T18:00:00Z',now),true);
  assert.equal(isAdmissiblePaperValuationQuote(4.5,'2026-10-11T09:05:00Z',now),true);
});
test('paper mark rejects future/stale/invalid quotes', () => {
  assert.equal(isAdmissiblePaperValuationQuote(4.5,'2026-10-11T09:05:01Z',now),false);
  assert.equal(isAdmissiblePaperValuationQuote(4.5,'2026-10-01T09:00:00Z',now),false);
  assert.equal(isAdmissiblePaperValuationQuote(4.5,'invalid',now),false);
  assert.equal(isAdmissiblePaperValuationQuote(4.5,null,now),false);
  assert.equal(isAdmissiblePaperValuationQuote(0,'2026-10-10T18:00:00Z',now),false);
  assert.equal(isAdmissiblePaperValuationQuote(NaN,'2026-10-10T18:00:00Z',now),false);
});
