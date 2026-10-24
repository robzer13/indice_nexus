import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { ResearchInboxRequest } from '../lib/orotitan-equity/product/local-research-inbox-v1';
import type { ResearchIdentitySnapshot } from '../lib/orotitan-equity/product/research-intake-preflight-v1';
import {
  buildResearchPreRunCard, PRE_RUN_REASON_LABELS,
  type ResearchRegistryRoutingSnapshot,
} from '../lib/orotitan-equity/product/research-pre-run-card-v1';

const request: ResearchInboxRequest = {
  requestId: '29402dfe-3f7d-43a1-bb08-e0b6c001141a',
  companyId: '00000000-0000-4000-8000-000000000091',
  companyName: 'Synthetic Research Alpha',
  ticker: 'TAL', exchange: 'XTEST',
  status: 'REQUESTED_NOT_EXECUTED', registryRunId: null,
  createdAt: '2026-10-24T13:41:38Z',
};
const issuerId = '11111111-1111-1111-1111-111111111111';
const securityId = '22222222-2222-2222-2222-222222222222';
const dossierId = '33333333-3333-3333-3333-333333333333';
const snapshotId = '55555555-5555-5555-5555-555555555555';
const identity: ResearchIdentitySnapshot = {
  verified: true,
  securities: [{ issuerId, securityId, ticker: 'TAL', exchange: 'XTEST' }],
};
const registry: ResearchRegistryRoutingSnapshot = {
  verified: true,
  issuers: [{
    issuerId, displayName: 'Synthetic Issuer', legalName: 'Synthetic Issuer SA',
    country: 'FR', reportingCurrency: 'EUR',
  }],
  dossiers: [], snapshots: [], runs: [],
};

test('anonymous request status never becomes a run / GO or a publication', () => {
  const card = buildResearchPreRunCard(request, identity, registry);
  assert.equal(card.entryPath, 'IMPOSED_COMPANY');
  assert.deepEqual(card.stageSequence, ['RESEARCH','DEEP_DIVE','INTEGRATION']);
  assert.equal(card.candidateIdentityConfirmed, false);
  assert.equal(card.executionAdmitted, false);
  assert.equal(card.executionGoReceived, false);
  assert.equal(card.publicationGoReceived, false);
  assert.equal(card.contractPinsVerified, false);
  assert.equal(card.cutoffValue, null);
  assert.ok(card.blockingReasons.includes('EXPLICIT_EXECUTION_GO_NOT_RECEIVED'));
});

test('no matching title prevents mode invention for the two hosted synthetic companies', () => {
  const card = buildResearchPreRunCard(request, {
    verified: true, securities: [],
  }, registry);
  assert.equal(card.canonicalModeProposal, null);
  assert.equal(card.candidateIssuerId, null);
  assert.ok(card.blockingReasons.includes('NO_EXACT_SECURITY_MATCH'));
  assert.equal(card.executionAdmitted, false);
});

test('single, exactly matched title with no canonical dossier proposes INITIAL only', () => {
  const card = buildResearchPreRunCard(request, identity, registry);
  assert.equal(card.canonicalModeProposal, 'ANALYZE');
  assert.equal(card.runTypeProposal, 'INITIAL');
  assert.equal(card.existingCanonicalDossierId, null);
  assert.equal(card.baselineSnapshotId, null);
  assert.equal(card.candidateIssuerId, issuerId);
  assert.ok(card.blockingReasons.includes('ISSUER_HUMAN_CONFIRMATION_REQUIRED'));
  assert.equal(card.executionAdmitted, false);
});

test('active dossier with linked canonical snapshot proposes REFRESH with baseline ID', () => {
  const card = buildResearchPreRunCard(request, identity, {
    ...registry,
    dossiers: [{ issuerId,dossierId,active:true,currentSnapshotId:snapshotId }],
    snapshots: [{ snapshotId,issuerId,dossierId }],
  });
  assert.equal(card.canonicalModeProposal, 'REFRESH');
  assert.equal(card.runTypeProposal, 'REFRESH');
  assert.equal(card.baselineSnapshotId, snapshotId);
  assert.equal(card.existingCanonicalDossierId, dossierId);
  assert.equal(card.executionAdmitted, false);
  assert.equal(card.cutoffValue, null);
});

test('snapshot belonging to another issuer cannot be accepted as baseline', () => {
  const card = buildResearchPreRunCard(request, identity, {
    ...registry,
    dossiers: [{ issuerId,dossierId,active:true,currentSnapshotId:snapshotId }],
    snapshots: [{
      snapshotId,dossierId,issuerId:'88888888-8888-8888-8888-888888888888',
    }],
  });
  assert.equal(card.canonicalModeProposal, null);
  assert.ok(card.blockingReasons.includes('CANONICAL_BASELINE_INVALID_OR_MISSING'));
});

test('inactive historical dossier never silently resolves as INITIAL', () => {
  const card = buildResearchPreRunCard(request, identity, {
    ...registry,
    dossiers: [{ issuerId,dossierId,active:false,currentSnapshotId:snapshotId }],
  });
  assert.equal(card.canonicalModeProposal, null);
  assert.ok(card.blockingReasons.includes('HISTORICAL_DOSSIER_STATE_NEEDS_REVIEW'));
});

test('multiple active dossiers fail closed even when one has a baseline', () => {
  const card = buildResearchPreRunCard(request, identity, {
    ...registry,
    dossiers: [
      { issuerId,dossierId,active:true,currentSnapshotId:snapshotId },
      { issuerId,dossierId:'44444444-4444-4444-4444-444444444444',
        active:true,currentSnapshotId:null },
    ],
  });
  assert.equal(card.canonicalModeProposal, null);
  assert.ok(card.blockingReasons.includes('MULTIPLE_ACTIVE_CANONICAL_DOSSIERS'));
});

test('incomplete legal issuer details cannot become identity authority', () => {
  const card = buildResearchPreRunCard(request, identity, {
    ...registry, issuers: [{
      issuerId,displayName:'Synthetic Issuer',legalName:null,
      country:null,reportingCurrency:'EUR',
    }],
  });
  assert.equal(card.legalIssuerName, null);
  assert.ok(card.blockingReasons.includes('LEGAL_ISSUER_IDENTITY_INCOMPLETE'));
  assert.equal(card.candidateIdentityConfirmed, false);
});

test('Registry timeout / partial read never means no dossier', () => {
  const card = buildResearchPreRunCard(request, identity, {
    ...registry, verified:false,
  });
  assert.equal(card.canonicalModeProposal, null);
  assert.ok(card.blockingReasons.includes('REGISTRY_CANONICAL_STATE_UNVERIFIED'));
});

test('ticker-only match at a different venue is not an identity match', () => {
  const card = buildResearchPreRunCard(request, {
    verified:true,securities:[{ issuerId,securityId,ticker:'TAL',exchange:'DIFFERENT' }],
  },registry);
  assert.equal(card.candidateIssuerId, null);
  assert.equal(card.canonicalModeProposal, null);
});

test('active prior run is a separate blocker, not idempotency proof', () => {
  const card = buildResearchPreRunCard(request, identity, {
    ...registry, runs:[{
      issuerId,runId:'66666666-6666-6666-6666-666666666666',runStatus:'ACTIVE',
    }],
  });
  assert.equal(card.openRunCount, 1);
  assert.ok(card.blockingReasons.includes('EXISTING_NONTERMINAL_RUN_REVIEW'));
  assert.equal(card.executionAdmitted,false);
});

test('unknown Registry run status fails closed', () => {
  const card = buildResearchPreRunCard(request, identity, {
    ...registry, runs:[{
      issuerId,runId:'66666666-6666-6666-6666-666666666666',runStatus:'NEW_INCOMPATIBLE',
    }],
  });
  assert.equal(card.openRunCount, null);
  assert.ok(card.blockingReasons.includes('RUN_REGISTRY_STATE_INVALID'));
});

test('no client receipt can grant GO or claim registry_run_id', () => {
  for (const tampered of [
    { ...request,requestId:'not-a-uuid' },
    { ...request,status:'APPROVED' },
    { ...request,registryRunId:issuerId },
  ]) assert.throws(() =>
    buildResearchPreRunCard(tampered as ResearchInboxRequest, identity, registry),
    /PRODUCT_PRE_RUN_INVALID_RECEIPT/);
});

test('assessment does not mutate client request, identity or Registry snapshot', () => {
  const before = JSON.stringify([request,identity,registry]);
  buildResearchPreRunCard(request,identity,registry);
  assert.equal(JSON.stringify([request,identity,registry]),before);
});

test('every emitted blocker has an operator-readable explanation', () => {
  const cases = [
    buildResearchPreRunCard(request,identity,registry),
    buildResearchPreRunCard(request,{verified:false,securities:[]},{...registry,verified:false}),
  ];
  for (const card of cases) {
    for (const code of card.blockingReasons) {
      assert.ok(typeof PRE_RUN_REASON_LABELS[code] === 'string');
      assert.ok(PRE_RUN_REASON_LABELS[code].length>20);
    }
  }
});
