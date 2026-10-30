/**
 * OroTitan product Research Recovery Loop V1 (SHADOW, non-authoritative).
 * An execution scheduler, NOT a change to frozen analytics or to Registry gates.
 * Never interprets a candidate source, an attempt, or a missing blocker
 * as evidence admission, financial certification, stage transition or GO PUBLISH.
 */
export type RecoveryBlocker = Readonly<{ code: string; reason: string }>;
export type RecoveryInput = Readonly<{
  runId: string;
  runStatus: string;
  stageCode: string;
  stageLifecycle: string;
  runVersion: number;
  stageVersion: number;
  blockers: readonly RecoveryBlocker[];
}>;
export type RecoveryStrategy =
  | 'REUSE_PRIMARY_ARTIFACTS'
  | 'RECONCILE_SHARE_COUNT'
  | 'RECONCILE_HISTORICAL_PRICE'
  | 'REBUILD_ROIC_OWNER_EARNINGS'
  | 'MANUAL_REVIEW';
export type RecoveryOutcome = 'CANDIDATE_RECORDED' | 'NO_PROGRESS' | 'REQUIRES_AUTHORITY';
export type RecoveryAttempt = Readonly<{
  blockerCode: string;
  strategy: RecoveryStrategy;
  outcome: RecoveryOutcome;
  detail: string;
  evidenceRefs: readonly string[];
}>;
export type RecoveryReport = Readonly<{
  status: 'PROGRESSED' | 'BUDGET_REACHED' | 'NO_NEW_AUTOMATED_STEPS' | 'PAUSED' | 'NO_BLOCKERS' | 'ERROR';
  attempted: readonly RecoveryAttempt[];
  remaining: readonly RecoveryBlocker[];
  publicationAuthorized: false;
  registryChanged: false;
  nextAction: string;
}>;

export interface RecoveryPort {
  /** Existing attempts are the authority for idempotency, never chat memory. */
  listAttempts(input: RecoveryInput): Promise<readonly RecoveryAttempt[]>;
  /** Append-only persistent record, idempotent on input revisions+blocker+strategy. */
  recordAttempt(input: RecoveryInput, attempt: RecoveryAttempt): Promise<void>;
  /** Read-only discovery or independently authorized repair; no Registry changes. */
  investigate(input: RecoveryInput, blocker: RecoveryBlocker, strategy: RecoveryStrategy):
    Promise<Pick<RecoveryAttempt,'outcome' | 'detail' | 'evidenceRefs'>>;
}

function strategyFor(code: string): RecoveryStrategy {
  if (code === 'DD-PRIMARY-SOURCES') return 'REUSE_PRIMARY_ARTIFACTS';
  if (code === 'ASML-G001') return 'RECONCILE_SHARE_COUNT';
  if (code === 'ASML-G006') return 'RECONCILE_HISTORICAL_PRICE';
  if (code === 'DD-ROIC-OE') return 'REBUILD_ROIC_OWNER_EARNINGS';
  return 'MANUAL_REVIEW';
}

const ORDER: RecoveryStrategy[] = [
  'REUSE_PRIMARY_ARTIFACTS','RECONCILE_SHARE_COUNT',
  'RECONCILE_HISTORICAL_PRICE','REBUILD_ROIC_OWNER_EARNINGS','MANUAL_REVIEW',
];

function assertInput(x: RecoveryInput) {
  if (!/^[0-9a-f-]{36}$/i.test(x.runId) || !x.stageCode ||
      !Number.isSafeInteger(x.runVersion) || x.runVersion < 0 ||
      !Number.isSafeInteger(x.stageVersion) || x.stageVersion < 0 ||
      x.blockers.some(b => !b.code || !b.reason || b.code.length > 100) ||
      new Set(x.blockers.map(b => b.code)).size !== x.blockers.length) {
    throw new Error('RECOVERY_INVALID_INPUT');
  }
}

/**
 * One bounded, awaitable cycle; the browser action can be reissued at any time.
 * No background promise or sleep. Durable record follows each attempted step.
 * Only strategies without declared authority are eligible for investigation.
 */
export async function runRecoveryCycle(
  input: RecoveryInput, port: RecoveryPort, maxSteps = 4,
): Promise<RecoveryReport> {
  assertInput(input);
  if (!Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 12) {
    throw new Error('RECOVERY_INVALID_BUDGET');
  }
  if (input.runStatus !== 'ACTIVE' || input.stageLifecycle !== 'IN_PROGRESS') {
    return {status:'PAUSED',attempted:[],remaining:input.blockers,
      publicationAuthorized:false,registryChanged:false,
      nextAction:'Run inactive, blocked or paused: respect the authoritative Registry stop.'};
  }
  if (input.blockers.length === 0) {
    return {status:'NO_BLOCKERS',attempted:[],remaining:[],
      publicationAuthorized:false,registryChanged:false,
      nextAction:'No recorded blockers; this is not stage certification.'};
  }
  const seen = await port.listAttempts(input);
  const attemptedKeys = new Set(seen.map(a => a.blockerCode + ':' + a.strategy));
  const attempted: RecoveryAttempt[] = [];
  const queue = [...input.blockers].sort((a,b) =>
    ORDER.indexOf(strategyFor(a.code)) - ORDER.indexOf(strategyFor(b.code)));
  for (const blocker of queue) {
    if (attempted.length >= maxSteps) break;
    const strategy = strategyFor(blocker.code);
    const key = blocker.code + ':' + strategy;
    if (attemptedKeys.has(key)) continue;
    // Unknown blockers are not investigated automatically.
    let result: Pick<RecoveryAttempt,'outcome'|'detail'|'evidenceRefs'>;
    if (strategy === 'MANUAL_REVIEW') {
      result = {outcome:'REQUIRES_AUTHORITY',
        detail:'Unknown blocker: human/contractual review; no automatic override.',
        evidenceRefs:[]};
    } else {
      try { result = await port.investigate(input, blocker, strategy); }
      catch {
        // A transient adapter failure does not exhaust the strategy:
        // leave it retryable and do not claim a persisted resolution.
        return {status:'ERROR',attempted,remaining:input.blockers,
          publicationAuthorized:false,registryChanged:false,
          nextAction:'Adapter temporarily unavailable; resume once connectivity is restored.'};
      }
    }
    if (!['CANDIDATE_RECORDED','NO_PROGRESS','REQUIRES_AUTHORITY'].includes(result.outcome) ||
        !Array.isArray(result.evidenceRefs) ||
        result.evidenceRefs.some(ref => typeof ref !== 'string' || ref.length > 300) ||
        typeof result.detail !== 'string' || result.detail.length > 1500) {
      throw new Error('RECOVERY_INVALID_RESOLVER_OUTPUT');
    }
    const attempt: RecoveryAttempt = {blockerCode:blocker.code,strategy,
      outcome:result.outcome,detail:result.detail,evidenceRefs:result.evidenceRefs};
    await port.recordAttempt(input, attempt);
    attemptedKeys.add(key);
    attempted.push(attempt);
  }
  const progressed = attempted.some(a => a.outcome === 'CANDIDATE_RECORDED');
  const pendingCount = queue.filter(blocker =>
    !attemptedKeys.has(blocker.code + ':' + strategyFor(blocker.code))).length;
  const remaining = input.blockers; // Only the authoritative Registry may clear blockers.
  return {
    status: progressed ? 'PROGRESSED' : pendingCount > 0 ? 'BUDGET_REACHED' : 'NO_NEW_AUTOMATED_STEPS',
    attempted, remaining,
    publicationAuthorized:false,registryChanged:false,
    nextAction: progressed
      ? pendingCount > 0
        ? 'Candidate evidence recorded. Additional unattempted tasks remain; resume the bounded cycle. Registry blockers stay open.'
        : 'Candidate evidence recorded. Validate provenance and Registry gate separately; blockers remain open.'
      : pendingCount > 0
        ? 'Bounded cycle budget reached; unattempted tasks remain. Resume without declaring source exhaustion.'
        : 'No new safe strategy in this cycle. Review recorded attempts; ask for one precise missing input if critical.',
  };
}
