/** Readable J1 -> Research routing packet. This is NOT a Research admission or
 * stage execution. Every receiving stage MUST validate persisted run/contract pins.
 */
export type J1VerifiedRun=Readonly<{
 runId:string;issuerName:string;isin:string;exchange:string;
 dataCutoff:string;profileId:'OROTITAN_V3_1';
 contractSetSha256:string;status:'CREATED'|'REPLAYED';
 persistedReadbackVerified:true;receiptPersisted:true;
 stageStarted:false;snapshotCreated:false;published:false;
 environment:'SHADOW_ONLY';productionTouched:false;
}>;
export function buildJ1ResearchHandoff(v:J1VerifiedRun):string{
 if(v.profileId!=='OROTITAN_V3_1'||v.persistedReadbackVerified!==true||
    v.receiptPersisted!==true||v.stageStarted!==false||
    v.snapshotCreated!==false||v.published!==false||
    v.environment!=='SHADOW_ONLY'||v.productionTouched!==false||
    !/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i.test(v.runId)||
    !/^[a-f0-9]{64}$/i.test(v.contractSetSha256))
  throw Error('J1_HANDOFF_INVARIANT_FAILED');
 return [
  'OROTITAN — PILOTAGE → RESEARCH — ROUTING HANDOFF',
  'COMPANY = '+v.issuerName,
  'RUN_ID = '+v.runId,
  'ISIN = '+v.isin,
  'EXCHANGE = '+v.exchange,
  'ENTRY_PATH = IMPOSED_COMPANY',
  'CANONICAL_MODE = ANALYZE',
  'RUN_TYPE = INITIAL',
  'RUN_STATUS = CREATED',
  'DATA_CUTOFF = '+v.dataCutoff,
  'STAGE = RESEARCH',
  'PROFILE_ID = OROTITAN_V3_1',
  'CONTRACT_SET_SHA256 = '+v.contractSetSha256,
  'HIGHER_AUTHORITY_PROCESS = OROTITAN_EXECUTION_PROCESS_V3_1_FREEZE_V3.1',
  'PILOTAGE_CONTRACT = OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_1_FREEZE_V3.1',
  'EXPECTED_STAGE_CONTRACT = OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0',
  'J1_PERSISTENT_RUN_READBACK = VERIFIED',
  'RESEARCH_STAGE_ADMISSION = NOT_YET_EXECUTED',
  'RESEARCH_STAGE_STARTED = NO',
  'PUBLICATION_AUTHORIZED = NO',
  'NEXT_ACTION = Research must independently retrieve the Run, verify exact pinned contract identity and enforce stage admission before work. This handoff is not analytical evidence.',
 ].join('\n');
}
