import 'server-only';

import { cache } from 'react';
import {
  createServerControlledBridgePort,
  executeServerControlledOperation,
} from '../orotitan-equity/post-c7/chatgpt-supabase-bridge-server';
import type {
  ControlledBridgePort,
  IssuerRow,
  LoadResult as BridgeLoadResult,
  OperationFailure,
  SecurityRow,
} from '../orotitan-equity/post-c7/chatgpt-supabase-bridge';
import {
  assertUniqueIssuerSlugs,
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

async function readDossierShell(
  port: ControlledBridgePort,
  issuer: IssuerRow,
  securities: SecurityRow[],
): Promise<UiDossierShell | null> {
  const [dossiers, runs] = await Promise.all([
    port.listDossiers(issuer.issuer_id),
    port.listRuns(issuer.issuer_id),
  ]);

  const activeDossiers = dossiers.filter((row) => row.active);
  if (activeDossiers.length === 0) return null;
  if (activeDossiers.length > 1) {
    throw new Error('OroTitan UI found multiple active dossiers for issuer');
  }

  const activeDossier = activeDossiers[0];
  const runSummaries = buildRunSummaries(
    runs,
    issuer.issuer_id,
    activeDossier.dossier_id,
  );
  if (runSummaries.length === 0) return null;

  const security = selectUiSecurity(securities, issuer.issuer_id);
  return {
    identity: buildCompanyIdentity(issuer, security),
    runSummaries,
  };
}

const loadUiDossierShell = cache(
  async (issuerQuery: string): Promise<UiDossierShell | null> => {
    const port = createServerControlledBridgePort();
    const [issuers, securities] = await Promise.all([
      port.listIssuers(),
      port.listSecurities(),
    ]);

    assertUniqueIssuerSlugs(issuers);
    const issuer = resolveUiIssuer(issuerQuery, issuers, securities);
    if (!issuer) return null;

    return readDossierShell(port, issuer, securities);
  },
);

const loadUiDossierShells = cache(async (): Promise<UiDossierShell[]> => {
  const port = createServerControlledBridgePort();
  const [issuers, securities] = await Promise.all([
    port.listIssuers(),
    port.listSecurities(),
  ]);

  assertUniqueIssuerSlugs(issuers);
  const shells = await Promise.all(
    issuers.map((issuer) => readDossierShell(port, issuer, securities)),
  );

  return shells
    .filter((shell): shell is UiDossierShell => shell !== null)
    .filter((shell) => shell.runSummaries.length > 0)
    .sort((left, right) =>
      left.identity.displayName.localeCompare(right.identity.displayName, 'fr'),
    );
});

export async function getUiDossierShell(
  issuerQuery: string,
): Promise<UiDossierShell | null> {
  return loadUiDossierShell(issuerQuery);
}

export async function listUiDossierShells(): Promise<UiDossierShell[]> {
  return loadUiDossierShells();
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
    issuer_query: shell.identity.legalName,
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
