import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  assessResearchIntakePreflight,
  PREFLIGHT_BLOCKER_TEXT,
  type ResearchIdentitySnapshot,
} from '../lib/orotitan-equity/product/research-intake-preflight-v1';
import type { ResearchInboxRequest } from '../lib/orotitan-equity/product/local-research-inbox-v1';

const request: ResearchInboxRequest = {
  requestId: '29402dfe-3f7d-43a1-bb08-e0b6c001141a',
  companyId: '00000000-0000-4000-8000-000000000091',
  companyName: 'Synthetic Research Alpha',
  ticker: 'TAL',
  exchange: 'XTEST',
  createdAt: '2026-10-24T13:41:38Z',
  status: 'REQUESTED_NOT_EXECUTED',
  registryRunId: null,
};
const candidate = {
  securityId: '00000000-0000-4000-8000-000000000093',
  issuerId: '00000000-0000-4000-8000-000000000094',
  ticker: 'TAL',
  exchange: 'XTEST',
};

test('an empty Registry never admits execution or invents an issuer', () => {
  const actual = assessResearchIntakePreflight(request, { verified: true, securities: [] });
  assert.equal(actual.verdict, 'BLOCKED_NOT_AUTHORIZED');
  assert.equal(actual.candidateCount, 0);
  assert.equal(actual.candidateIssuerId, null);
  assert.ok(actual.blockerCodes.includes('ISSUER_NOT_FOUND'));
  assert.ok(actual.blockerCodes.includes('EXECUTION_GO_MISSING'));
  assert.equal(actual.requestId, request.requestId);
});

test('a unique security is only a candidate and cannot bypass GO', () => {
  const actual = assessResearchIntakePreflight(request, {
    verified: true, securities: [candidate],
  });
  assert.equal(actual.verdict, 'BLOCKED_NOT_AUTHORIZED');
  assert.equal(actual.candidateCount, 1);
  assert.equal(actual.candidateIssuerId, candidate.issuerId);
  assert.deepEqual(actual.blockerCodes, [
    'ISSUER_REVIEW_REQUIRED', 'MODE_AND_BASELINE_UNRESOLVED',
    'DATA_CUTOFF_MISSING', 'CONTRACT_PINS_MISSING', 'EXECUTION_GO_MISSING',
  ]);
});

test('ticker-only match or fuzzy issuer name is never accepted', () => {
  const actual = assessResearchIntakePreflight(request, {
    verified: true,
    securities: [{ ...candidate, exchange: 'DIFFERENT' }],
  });
  assert.equal(actual.candidateCount, 0);
  assert.ok(actual.blockerCodes.includes('ISSUER_NOT_FOUND'));
});

test('ticker and exchange comparisons normalize case, not identity authority', () => {
  const actual = assessResearchIntakePreflight(request, {
    verified: true,
    securities: [{ ...candidate, ticker: 'tal', exchange: 'xtest' }],
  });
  assert.equal(actual.candidateCount, 1);
  assert.ok(actual.blockerCodes.includes('ISSUER_REVIEW_REQUIRED'));
});

test('ambiguous security resolution blocks approval', () => {
  const actual = assessResearchIntakePreflight(request, {
    verified: true,
    securities: [candidate, {
      ...candidate, securityId: '00000000-0000-4000-8000-000000000095',
    }],
  });
  assert.equal(actual.candidateCount, 2);
  assert.equal(actual.candidateIssuerId, null);
  assert.ok(actual.blockerCodes.includes('ISSUER_AMBIGUOUS'));
});

test('an unavailable or truncated Registry is unknown, never no-match or approved', () => {
  const actual = assessResearchIntakePreflight(request, {
    verified: false, securities: [candidate],
  });
  assert.equal(actual.candidateCount, null);
  assert.equal(actual.candidateIssuerId, null);
  assert.ok(actual.blockerCodes.includes('REGISTRY_LOOKUP_UNAVAILABLE'));
});

test('invalid request or already-linked run is rejected before assessment', () => {
  for (const malformed of [
    { ...request, requestId: 'invalid' },
    { ...request, status: 'APPROVED' },
    { ...request, registryRunId: candidate.issuerId },
    { ...request, exchange: '' },
  ]) {
    assert.throws(() =>
      assessResearchIntakePreflight(malformed as ResearchInboxRequest, {
        verified: true, securities: [candidate],
      }), /PRODUCT_PREFLIGHT_INVALID_REQUEST/);
  }
});

test('all blocker codes have an explicit French operator explanation', () => {
  const known = [
    'REGISTRY_LOOKUP_UNAVAILABLE', 'ISSUER_NOT_FOUND', 'ISSUER_AMBIGUOUS',
    'ISSUER_REVIEW_REQUIRED', 'MODE_AND_BASELINE_UNRESOLVED',
    'DATA_CUTOFF_MISSING', 'CONTRACT_PINS_MISSING', 'EXECUTION_GO_MISSING',
  ];
  for (const code of known) assert.ok(PREFLIGHT_BLOCKER_TEXT[code as keyof typeof PREFLIGHT_BLOCKER_TEXT].length > 20);
});

test('snapshot input remains immutable after assessment', () => {
  const snapshot: ResearchIdentitySnapshot = {
    verified: true, securities: [candidate],
  };
  const before = JSON.stringify(snapshot);
  assessResearchIntakePreflight(request, snapshot);
  assert.equal(JSON.stringify(snapshot), before);
  assert.equal(request.registryRunId, null);
  assert.equal(request.status, 'REQUESTED_NOT_EXECUTED');
});
