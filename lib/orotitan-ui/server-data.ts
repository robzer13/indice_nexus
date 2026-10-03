import 'server-only';

import { cache } from 'react';
import {
  createServerControlledBridgePort,
  executeServerControlledOperation,
} from '../orotitan-equity/post-c7/chatgpt-supabase-bridge-server';
import type {
  LoadResult as BridgeLoadResult,
  OperationFailure,
} from '../orotitan-equity/post-c7/chatgpt-supabase-bridge';
import {
  buildArtifactCatalog,
  buildCompanyIdentity,
  buildRunSummaries,
  resolveUiIssuer,
  resolveUiRunSelection,
  selectUiSecurity,
} from './read-model';
import type {
  LoadResult,
  RunSummary,
  UiDossier,
  UiDossierShell,
} from './types';

function operationFailureError(result: OperationFailure): Error {
  return new Error(
    'OroTitan controlled LOAD failed [' +
      result.error_class +
      ']: ' +
      result.message,
  );
}

function toUiLoadResult(result: BridgeLoadResult): LoadResult {
  for (const blocker of result.blockers) {
    if (typeof blocker.code !== 'string' || blocker.code.length === 0) {
      throw new Error('OroTitan LOAD_RESULT blocker is missing a code');
    }
  }
  return result as LoadResult;
}

const loadUiDossierShell = cache(
  async (issuerQuery: string): Promise<UiDossierShell | null> => {
    const port = createServerControlledBridgePort();
    const [issuers, securities] = await Promise.all([
      port.listIssuers(),
      port.listSecurities(),
    ]);

    const issuer = resolveUiIssuer(issuerQuery, issuers, securities);
    if (!issuer) return null;

    const security = selectUiSecurity(securities, issuer.issuer_id);
    const runs = await port.listRuns(issuer.issuer_id);

    return {
      identity: buildCompanyIdentity(issuerQuery, issuer, security),
      runSummaries: buildRunSummaries(runs, issuer.issuer_id),
    };
  },
);

export async function getUiDossierShell(
  issuerQuery: string,
): Promise<UiDossierShell | null> {
  return loadUiDossierShell(issuerQuery);
}

export type UiDossierSelection =
  | { kind: 'issuer-not-found' }
  | { kind: 'select'; shell: UiDossierShell }
  | { kind: 'unknown'; shell: UiDossierShell }
  | { kind: 'unavailable'; shell: UiDossierShell; summary: RunSummary }
  | { kind: 'available'; dossier: UiDossier };

export async function getUiDossierSelection(
  issuerQuery: string,
  requestedRun: string | null,
): Promise<UiDossierSelection> {
  const shell = await getUiDossierShell(issuerQuery);
  if (!shell) return { kind: 'issuer-not-found' };

  const selection = resolveUiRunSelection(shell.runSummaries, requestedRun);
  if (selection.kind === 'select') return { kind: 'select', shell };
  if (selection.kind === 'unknown') return { kind: 'unknown', shell };
  if (selection.kind === 'unavailable') {
    return {
      kind: 'unavailable',
      shell,
      summary: selection.summary,
    };
  }

  const result = await executeServerControlledOperation({
    contract_version: '0.1.0',
    operation: 'LOAD',
    issuer_query: issuerQuery,
    run_id: selection.summary.runId,
  });

  if (result.operation === 'OPERATION_FAILURE') {
    throw operationFailureError(result);
  }
  if (result.operation !== 'LOAD_RESULT') {
    throw new Error('OroTitan UI rejected a non-LOAD result');
  }

  const loadResult = toUiLoadResult(result);
  if (loadResult.run_id !== selection.summary.runId) {
    throw new Error('OroTitan UI LOAD_RESULT returned an unexpected run');
  }

  const port = createServerControlledBridgePort();
  const artifactRows = await port.listArtifacts(selection.summary.runId);
  const artifactCatalog = buildArtifactCatalog(loadResult, artifactRows);

  return {
    kind: 'available',
    dossier: {
      ...shell,
      loadResult,
      artifactCatalog,
    },
  };
}
