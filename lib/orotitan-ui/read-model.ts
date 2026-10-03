import type {
  ArtifactRow,
  IssuerRow,
  RunRow,
  SecurityRow,
} from '../orotitan-equity/post-c7/chatgpt-supabase-bridge';
import type {
  ArtifactMeta,
  CompanyIdentity,
  LoadResult,
  RunSummary,
} from './types';

const TERMINAL_RUN_STATUSES = new Set(['PUBLISHED', 'CANCELLED']);

function normalized(value: string): string {
  return value.trim().toLocaleLowerCase('en-US');
}

export function issuerSlug(value: string): string {
  const slug = value
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLocaleLowerCase('en-US')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');

  if (!slug) {
    throw new Error('OroTitan UI issuer display name cannot produce a canonical slug');
  }
  return slug;
}

export function assertUniqueIssuerSlugs(issuers: IssuerRow[]): void {
  const ownerBySlug = new Map<string, string>();
  for (const issuer of issuers) {
    const slug = issuerSlug(issuer.display_name);
    const prior = ownerBySlug.get(slug);
    if (prior && prior !== issuer.issuer_id) {
      throw new Error('OroTitan UI canonical issuer slug is ambiguous: ' + slug);
    }
    ownerBySlug.set(slug, issuer.issuer_id);
  }
}

export function resolveUiIssuer(
  query: string,
  issuers: IssuerRow[],
  securities: SecurityRow[],
): IssuerRow | null {
  const needle = normalized(query);
  if (!needle) return null;

  const issuerIds = new Set<string>();
  for (const issuer of issuers) {
    if (
      normalized(issuer.display_name) === needle ||
      issuerSlug(issuer.display_name) === needle ||
      (issuer.legal_name !== null && normalized(issuer.legal_name) === needle)
    ) {
      issuerIds.add(issuer.issuer_id);
    }
  }

  for (const security of securities) {
    if (
      normalized(security.ticker) === needle ||
      (security.market_data_symbol !== null && normalized(security.market_data_symbol) === needle)
    ) {
      issuerIds.add(security.issuer_id);
    }
  }

  if (issuerIds.size === 0) return null;
  if (issuerIds.size > 1) {
    throw new Error('OroTitan UI issuer query is ambiguous');
  }

  const issuerId = [...issuerIds][0];
  const issuer = issuers.find((candidate) => candidate.issuer_id === issuerId);
  if (!issuer) {
    throw new Error('OroTitan UI resolved issuer row is missing');
  }
  return issuer;
}

export function selectUiSecurity(
  rows: SecurityRow[],
  issuerId: string,
): SecurityRow {
  const issuerRows = rows.filter((row) => row.issuer_id === issuerId);
  const primary = issuerRows.filter((row) => row.primary_listing === true);

  if (primary.length === 1) return primary[0];
  if (primary.length > 1) {
    throw new Error('OroTitan UI found multiple primary securities for issuer');
  }
  if (issuerRows.length === 1) return issuerRows[0];
  if (issuerRows.length === 0) {
    throw new Error('OroTitan UI issuer has no resolvable security');
  }
  throw new Error('OroTitan UI issuer has multiple securities and no unique primary listing');
}

export function buildCompanyIdentity(
  issuer: IssuerRow,
  security: SecurityRow,
): CompanyIdentity {
  return {
    slug: issuerSlug(issuer.display_name),
    displayName: issuer.display_name,
    legalName: issuer.legal_name ?? issuer.display_name,
    ticker: security.ticker,
    marketDataSymbol: security.market_data_symbol ?? security.ticker,
  };
}

export function buildRunSummaries(
  rows: RunRow[],
  issuerId: string,
  dossierId?: string,
): RunSummary[] {
  return rows
    .filter((row) => !TERMINAL_RUN_STATUSES.has(row.run_status))
    .filter((row) => dossierId === undefined || row.dossier_id === dossierId)
    .map((row) => {
      if (row.issuer_id !== issuerId) {
        throw new Error('OroTitan UI run issuer does not match resolved issuer');
      }

      return {
        runId: row.run_id,
        runStatus: row.run_status,
        currentStage: row.current_stage,
        dataCutoff: row.data_cutoff,
        stateVersion: row.state_version,
        loadAvailable:
          row.current_stage !== null &&
          row.security_id !== null &&
          row.dossier_id !== null,
      };
    });
}

export type UiRunSelection =
  | { kind: 'select' }
  | { kind: 'available'; summary: RunSummary }
  | { kind: 'unavailable'; summary: RunSummary }
  | { kind: 'unknown' };

export function resolveUiRunSelection(
  runs: RunSummary[],
  requestedRun: string | null,
): UiRunSelection {
  if (!requestedRun) return { kind: 'select' };

  const summary = runs.find((run) => run.runId === requestedRun);
  if (!summary) return { kind: 'unknown' };
  return summary.loadAvailable
    ? { kind: 'available', summary }
    : { kind: 'unavailable', summary };
}

function artifactKey(artifactId: string, version: number): string {
  return artifactId + ':' + version;
}

export function buildArtifactCatalog(
  load: LoadResult,
  rows: ArtifactRow[],
): Record<string, ArtifactMeta> {
  if (!load.run_id) {
    throw new Error('OroTitan UI artifact catalog requires a loaded run');
  }

  const rowsByExactIdentity = new Map<string, ArtifactRow>();
  for (const row of rows) {
    const key = artifactKey(row.artifact_id, row.version);
    if (rowsByExactIdentity.has(key)) {
      throw new Error('OroTitan UI artifact registry contains duplicate exact identities');
    }
    rowsByExactIdentity.set(key, row);
  }

  const versionsByArtifactId = new Map<string, number>();
  const catalog: Record<string, ArtifactMeta> = {};

  for (const ref of load.artifact_index) {
    const priorVersion = versionsByArtifactId.get(ref.artifact_id);
    if (priorVersion !== undefined && priorVersion !== ref.version) {
      throw new Error('OroTitan UI catalog cannot represent two active versions of one artifact ID');
    }
    versionsByArtifactId.set(ref.artifact_id, ref.version);

    const row = rowsByExactIdentity.get(artifactKey(ref.artifact_id, ref.version));
    if (!row) {
      throw new Error('OroTitan UI LOAD-authorized artifact metadata is missing');
    }
    if (row.run_id !== load.run_id) {
      throw new Error('OroTitan UI artifact belongs to a different run');
    }
    if (
      row.artifact_status !== 'SEALED' ||
      row.availability_state !== 'AVAILABLE' ||
      (row.authority_state !== 'AUTHORITATIVE' && row.authority_state !== 'CHECKPOINT')
    ) {
      throw new Error('OroTitan UI artifact metadata is not in an active readable state');
    }
    if (ref.content_sha256 && row.content_sha256 !== ref.content_sha256) {
      throw new Error('OroTitan UI artifact SHA-256 does not match LOAD_RESULT');
    }
    if (
      ref.required_authority_class &&
      row.authority_class !== ref.required_authority_class
    ) {
      throw new Error('OroTitan UI artifact authority class does not match LOAD_RESULT');
    }

    catalog[ref.artifact_id] = {
      artifactId: row.artifact_id,
      stageCode: row.stage_code,
      artifactType: row.artifact_type,
      logicalName: row.logical_name,
      authorityState: row.authority_state,
      artifactStatus: 'SEALED',
      availabilityState: 'AVAILABLE',
      storageBackend: row.storage_backend,
    };
  }

  return catalog;
}
