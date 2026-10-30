import 'server-only';
import {retrievePreferredAsmlQ2OfficialSource} from './research-recovery-official-source-v1';
import {archiveAsmlCandidate} from './research-recovery-private-archive-v1';
import {reviewEconomicShareLedger,reviewHistoricalXamsClose} from './research-recovery-financial-evidence-v1';
import { getHostedShadowRecoveryClient } from './research-inbox-pilot-server';
import {
  runRecoveryCycle, type RecoveryInput, type RecoveryAttempt,
  type RecoveryBlocker, type RecoveryStrategy, type RecoveryPort,
} from './research-recovery-loop-v1';

/**
 * ASML observation from production READ ONLY (2026-10-30).
 * Not an ASML run in SHADOW; never mutate or clear the production Registry.
 */
export const ASML_READONLY_OBSERVATION: RecoveryInput = {
  runId: 'f4ae40f7-eb9b-4472-b124-9fb152b739ce',
  runStatus: 'ACTIVE', stageCode: 'DEEP_DIVE', stageLifecycle: 'IN_PROGRESS',
  runVersion: 6, stageVersion: 2,
  blockers: [
    {code:'ASML-G001',reason:'ECONOMIC_SHARE_COUNT at 2026-09-21 close UNKNOWN'},
    {code:'ASML-G006',reason:'XAMS closing price candidate not source admitted'},
    {code:'DD-PRIMARY-SOURCES',reason:'New original primary source bytes not admitted for Deep Dive'},
    {code:'DD-ROIC-OE',reason:'Operating capital and Owner Earnings not certified'},
  ],
};
const JOURNAL = 'orotitan_product_recovery_attempts_shadow_v1';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
type Snapshot = Readonly<{input:RecoveryInput; origin:'ASML_READ_ONLY_MIRROR'|'SHADOW_REGISTRY'}>;
const parseBlockers = (raw: unknown): RecoveryBlocker[] => {
  if (!Array.isArray(raw) || raw.length > 100) throw Error('RECOVERY_INVALID_BLOCKERS');
  return raw.map(row => {
    if (!row || typeof row !== 'object' || Array.isArray(row)) throw Error('RECOVERY_INVALID_BLOCKERS');
    const x=row as Record<string,unknown>;
    if (typeof x.code !== 'string' || typeof x.reason !== 'string' || !x.code || !x.reason)
      throw Error('RECOVERY_INVALID_BLOCKERS');
    return {code:x.code,reason:x.reason};
  });
};
export async function getRecoverySnapshot(key: string): Promise<Snapshot> {
  // Validate SHADOW gate BEFORE returning even the static fixture.
  const client = getHostedShadowRecoveryClient();
  if (key === 'asml-observation') return {input:ASML_READONLY_OBSERVATION,origin:'ASML_READ_ONLY_MIRROR'};
  if (!UUID.test(key)) throw Error('RECOVERY_INVALID_RUN');
  const {data:run,error} = await client.from('orotitan_runs')
    .select('run_id,run_status,current_stage,state_version')
    .eq('run_id',key).maybeSingle();
  if (error || !run || !run.current_stage || !Number.isSafeInteger(run.state_version))
    throw Error('RECOVERY_RUN_UNAVAILABLE');
  const {data:stage,error:stageError} = await client.from('orotitan_run_stages')
    .select('lifecycle_status,state_version,blocker_summary')
    .eq('run_id',key).eq('stage_code',run.current_stage).maybeSingle();
  if (stageError || !stage || !Number.isSafeInteger(stage.state_version))
    throw Error('RECOVERY_STAGE_UNAVAILABLE');
  return {origin:'SHADOW_REGISTRY',input:{
    runId:run.run_id,runStatus:run.run_status,stageCode:run.current_stage,
    runVersion:run.state_version,stageVersion:stage.state_version,
    stageLifecycle:stage.lifecycle_status,blockers:parseBlockers(stage.blocker_summary),
  }};
}
export async function listRecoveryJournal(input: RecoveryInput): Promise<RecoveryAttempt[]> {
  const c=getHostedShadowRecoveryClient();
  const {data,error}=await c.from(JOURNAL)
    .select('blocker_code,strategy,outcome,detail,evidence_refs')
    .eq('run_id',input.runId).eq('stage_code',input.stageCode)
    .eq('run_state_version',input.runVersion).eq('stage_state_version',input.stageVersion)
    .order('created_at',{ascending:true}).limit(200);
  if (error || !data || data.length>=200) throw Error('RECOVERY_JOURNAL_UNAVAILABLE');
  return data.map(row=>({
    blockerCode:row.blocker_code,strategy:row.strategy,
    outcome:row.outcome,detail:row.detail,evidenceRefs:row.evidence_refs,
  })) as RecoveryAttempt[];
}
export async function executeShadowRecoveryCycle(key: string) {
  const snapshot=await getRecoverySnapshot(key);
  const client=getHostedShadowRecoveryClient();
  const port:RecoveryPort={
    listAttempts:listRecoveryJournal,
    async recordAttempt(input,attempt) {
      const {error}=await client.from(JOURNAL).insert({
        run_id:input.runId,stage_code:input.stageCode,
        run_state_version:input.runVersion,stage_state_version:input.stageVersion,
        blocker_code:attempt.blockerCode,strategy:attempt.strategy,outcome:attempt.outcome,
        detail:attempt.detail,evidence_refs:attempt.evidenceRefs,
      });
      if (error && error.code !== '23505') throw Error('RECOVERY_JOURNAL_WRITE_FAILED');
    },
    async investigate(input,_blocker,strategy:RecoveryStrategy) {
      if (strategy==='REUSE_PRIMARY_ARTIFACTS') {
        // Read-only search of pre-existing SHADOW evidence. This is not source admission.
        const {data,error}=await client.from('orotitan_artifacts')
          .select('artifact_id,artifact_type,content_sha256,size_bytes,availability_state,artifact_status')
          .eq('run_id',input.runId).eq('availability_state','AVAILABLE')
          .eq('artifact_status','SEALED').limit(200);
        if (error || !data || data.length>=200) throw Error('RECOVERY_SOURCE_INDEX_UNAVAILABLE');
        const sources=data.filter(r=>/SOURCE|PRIMARY|ORIGINAL/i.test(r.artifact_type??'') &&
          typeof r.content_sha256==='string' && /^[a-f0-9]{64}$/i.test(r.content_sha256) &&
          typeof r.size_bytes==='number' && r.size_bytes>0);
        if (!sources.length) {
          // An ASML-only, public, dated candidate is allowed for the frozen
          // production OBSERVATION mirror. Never infer an issuer from a SHADOW
          // run UUID, and never use post-valuation Q3 material.
          if (snapshot.origin==='ASML_READ_ONLY_MIRROR') {
            const {candidate:source,bytes}=await retrievePreferredAsmlQ2OfficialSource();
            // Durable private capture, not a canonical artifact or a proof that
            // the exact same bytes existed at the historical valuation cut-off.
            const receipt=await archiveAsmlCandidate(client,source,bytes);
            return {
              outcome:'CANDIDATE_RECORDED' as const,
              detail:`Source primaire ${source.documentId}, publiée ${source.publicationDate}, récupérée ${source.retrievedAt}, ${receipt.byteSize} octets ${source.contentType}, SHA-256 ${receipt.sha256}, archive privée SHADOW create-only et relecture SHA-256 conforme. Octets capturés aujourd'hui, PAS certifiés à la date d'évaluation, aucune admission Registry, ni G001/G006/ROIC résolu.`,
              evidenceRefs:[source.url,'sha256:'+receipt.sha256,
                'storage://'+receipt.bucket+'/'+receipt.path],
            };
          }
          return {outcome:'NO_PROGRESS' as const,
            detail:'Aucune pièce primaire scellée trouvée sur ce run SHADOW. Le connecteur officiel est volontairement limité à observation ASML datée; aucune recherche arbitraire.',
            evidenceRefs:[]};
        }
        return {outcome:'CANDIDATE_RECORDED' as const,
          detail:`${sources.length} candidats binaires identifiés (hash/tailles présents). Provenance et contenu NON admis ; validation distincte obligatoire.`,
          evidenceRefs:sources.slice(0,15).map(r=>String(r.artifact_id))};
      }
      return {outcome:'REQUIRES_AUTHORITY' as const,
        detail:strategy==='RECONCILE_SHARE_COUNT'
          ? reviewEconomicShareLedger(undefined).state==='BLOCKED'
            ? 'G001 : impossible sans nombre de titres émis et autodétenus sourcés, mouvements réglés jusqu’au 21/09/2026 et preuve de couverture exhaustive. Aucun calcul certifié.'
            : 'G001: review required.'
          : strategy==='RECONCILE_HISTORICAL_PRICE'
          ? reviewHistoricalXamsClose(undefined).state==='BLOCKED'
            ? 'G006 : cours de clôture non ajusté du 21/09/2026, ISIN NL0010273215, place XAMS, EUR/action et provenance Euronext à vérifier. Aucune valeur improvisée.'
            : 'G006: review required.'
          : 'Dépend des faits financiers et du capital investi vérifiés ; aucun ROIC/Owner Earnings improvisé.',
        evidenceRefs:[]};
    },
  };
  return {snapshot,report:await runRecoveryCycle(snapshot.input,port,4)};
}
