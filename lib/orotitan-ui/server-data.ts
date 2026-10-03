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
  assertSafeIssuerRoutes,
  buildArtifactCatalog,
  buildCompanyIdentity,
  buildRunSummaries,
  resolveUiIssuer,
  resolveUiRunSelection,
  selectUiSecurity,
  selectUniqueBridgeIssuerQuery,
} from './read-model';
import { ArtifactContentError, resolveVerifiedArtifactContent } from './artifact-reader';
import { readPrivateGithubArtifactBytes } from './artifact-reader-server';
import type {
  LoadResult,
  RunSummary,
  UiDossier,
  UiDossierShell,
  VerifiedArtifactContent,
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

type UiDossierResolution = {
  shell: UiDossierShell;
  bridgeIssuerQuery: string;
};

async function readDossierResolution(
  port: ControlledBridgePort,
  issuer: IssuerRow,
  issuers: IssuerRow[],
  securities: SecurityRow[],
): Promise<UiDossierResolution | null> {
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
  const bridgeIssuerQuery = selectUniqueBridgeIssuerQuery(
    issuer,
    issuers,
    securities,
  );
  return {
    shell: {
      identity: buildCompanyIdentity(issuer, security),
      runSummaries,
    },
    bridgeIssuerQuery,
  };
}

const loadUiDossierResolution = cache(
  async (issuerQuery: string): Promise<UiDossierResolution | null> => {
    const port = createServerControlledBridgePort();
    const [issuers, securities] = await Promise.all([
      port.listIssuers(),
      port.listSecurities(),
    ]);

    assertSafeIssuerRoutes(issuers, securities);
    const issuer = resolveUiIssuer(issuerQuery, issuers, securities);
    if (!issuer) return null;

    return readDossierResolution(port, issuer, issuers, securities);
  },
);

const loadUiDossierResolutions = cache(
  async (): Promise<UiDossierResolution[]> => {
    const port = createServerControlledBridgePort();
    const [issuers, securities] = await Promise.all([
      port.listIssuers(),
      port.listSecurities(),
    ]);

    assertSafeIssuerRoutes(issuers, securities);
    const resolutions = await Promise.all(
      issuers.map((issuer) =>
        readDossierResolution(port, issuer, issuers, securities),
      ),
    );

    return resolutions
      .filter(
        (resolution): resolution is UiDossierResolution =>
          resolution !== null,
      )
      .sort((left, right) =>
        left.shell.identity.displayName.localeCompare(
          right.shell.identity.displayName,
          'fr',
        ),
      );
  },
);

export async function getUiDossierShell(
  issuerQuery: string,
): Promise<UiDossierShell | null> {
  const resolution = await loadUiDossierResolution(issuerQuery);
  return resolution?.shell ?? null;
}

export async function listUiDossierShells(): Promise<UiDossierShell[]> {
  const resolutions = await loadUiDossierResolutions();
  return resolutions.map((resolution) => resolution.shell);
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
  const resolution = await loadUiDossierResolution(issuerQuery);
  if (!resolution) return { kind: 'issuer-not-found' };

  const { shell, bridgeIssuerQuery } = resolution;
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
    issuer_query: bridgeIssuerQuery,
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


export async function getVerifiedUiArtifactContent(input: {
  issuerQuery: string;
  runId: string;
  artifactId: string;
  version: number;
}): Promise<VerifiedArtifactContent> {
  const selection = await getUiDossierSelection(
    input.issuerQuery,
    input.runId,
  );
  if (selection.kind !== 'available') {
    throw new ArtifactContentError(
      'ARTIFACT_NOT_LOAD_AUTHORIZED',
      'Requested run is not available for verified artifact reading',
    );
  }

  const dossier = selection.dossier;
  if (dossier.loadResult.run_id !== input.runId) {
    throw new ArtifactContentError(
      'ARTIFACT_REGISTRY_MISMATCH',
      'Loaded run does not match requested artifact run',
    );
  }

  const port = createServerControlledBridgePort();
  const artifactRows = await port.listArtifacts(input.runId);

  return resolveVerifiedArtifactContent({
    port,
    loadRefs: dossier.loadResult.artifact_index,
    artifactRows,
    runId: input.runId,
    artifactId: input.artifactId,
    version: input.version,
    readPrivateGithub: readPrivateGithubArtifactBytes,
  });
}
