import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { unpublishedTrackedCompanies, type PublishedDossierPointer, type PublishedSnapshotIdentity } from '../lib/domain/company-intake';
import type { LegacyCompanyQuoteLink } from '../lib/domain/canonical-quote-identity';
import type { CompanyRecord } from '../lib/domain/types';

const record = (id: string,name: string,active=true): CompanyRecord => ({
  id, name, active, ticker:id.toUpperCase(),slug:id.toLowerCase(),exchange:'Nasdaq',
  currency:'USD', quote_unit:'MAJOR',price_decimals:2,market_data_symbol:id,
  market_data_multiplier:1,country:null,sector:null,created_at:'2026-10-01T00:00:00Z',
  updated_at:'2026-10-01T00:00:00Z',
});
const rows=[record('one','Company One'),record('two','Company Two'),record('new','New action'),
  record('off','Inactive Company',false)];
const links:LegacyCompanyQuoteLink[]=[
  {legacy_company_id:'one',issuer_id:'issuer-one',security_id:'sec-one',dossier_id:'doss-one'},
  {legacy_company_id:'two',issuer_id:'issuer-two',security_id:'sec-two',dossier_id:'doss-two'},
];
const pointers:PublishedDossierPointer[]=[
  {issuer_id:'issuer-one',dossier_id:'doss-one',current_snapshot_id:'snap-one'},
  {issuer_id:'issuer-two',dossier_id:'doss-two',current_snapshot_id:'snap-two'},
];
const snapshots:PublishedSnapshotIdentity[]=[
  {snapshot_id:'snap-one',issuer_id:'issuer-one',security_id:'sec-one',dossier_id:'doss-one'},
  {snapshot_id:'snap-two',issuer_id:'issuer-two',security_id:'sec-two',dossier_id:'doss-two'},
];
test('P0 intake: only tracked, unpublished active company appears when legacy and canonical align',()=>{
  const result=unpublishedTrackedCompanies(rows,links,pointers,snapshots);
  assert.deepEqual(result.map(x=>x.id).sort(),['new']);
});
test('P0 intake: missing snapshots and missing links never imply research publication',()=>{
  assert.deepEqual(unpublishedTrackedCompanies(rows,links,pointers,[]).map(x=>x.id).sort(),['new','one','two']);
  assert.deepEqual(unpublishedTrackedCompanies(rows,[],pointers,snapshots).map(x=>x.id).sort(),['new','one','two']);
  assert.deepEqual(unpublishedTrackedCompanies(rows,[],[],[]).map(x=>x.id).sort(),['new','one','two']);
});
test('P0 intake: issuer, security, dossier and snapshot must all agree; ticker similarity is irrelevant',()=>{
  const badSnapshot=[{...snapshots[0],security_id:'wrong-security'},snapshots[1]];
  assert.deepEqual(unpublishedTrackedCompanies(rows,links,pointers,badSnapshot).map(x=>x.id).sort(),['new','one']);
  const wrongIssuer=[{...links[0],issuer_id:'another-issuer'},links[1]];
  assert.deepEqual(unpublishedTrackedCompanies(rows,wrongIssuer,pointers,snapshots).map(x=>x.id).sort(),['new','one']);
  const wrongDossier=[{...links[0],dossier_id:'old-dossier'},links[1]];
  assert.deepEqual(unpublishedTrackedCompanies(rows,wrongDossier,pointers,snapshots).map(x=>x.id).sort(),['new','one']);
  const wrongCurrent=[{...pointers[0],current_snapshot_id:'superseded'},pointers[1]];
  assert.deepEqual(unpublishedTrackedCompanies(rows,links,wrongCurrent,snapshots).map(x=>x.id).sort(),['new','one']);
});
test('P0 intake: public Screener remains published-only; privileged intake is not in scores or rankings',()=>{
  const page=readFileSync('app/screener/page.tsx','utf8');
  assert.match(page,/isAdminAuthenticated\(\)/);
  assert.match(page,/admin \? await getUnpublishedTrackedCompanies\(\) : \[\]/);
  assert.match(page,/ScreenerTable companies=\{companies\}/);
  assert.match(page,/admin \? <section id="titres-en-attente"/);
  assert.doesNotMatch(page,/ScreenerTable companies=\{\[\.\.\.companies,\.\.\.pending\]\}/);
  const data=readFileSync('lib/data/company-intake.ts','utf8');
  assert.match(data,/legacy_company_identity_map/);
  assert.match(data,/research_dossiers/);
  assert.match(data,/research_snapshots/);
  assert.doesNotMatch(data,/\.insert\(|\.update\(|\.upsert\(/);
});
