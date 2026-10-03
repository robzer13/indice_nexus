import type {
  LoadResult as BridgeLoadResult,
  StageCode as BridgeStageCode,
  StageLifecycle as BridgeStageLifecycle,
} from '../orotitan-equity/post-c7/chatgpt-supabase-bridge';

export type StageCode = BridgeStageCode;
export type StageLifecycle = BridgeStageLifecycle;
export type ArtifactRef = BridgeLoadResult['artifact_index'][number];

export type Blocker = {
  code: string;
  classification?: string;
  scope?: string;
  detail?: string;
  diagnostic?: string;
  resolution_required?: string;
  [key: string]: unknown;
};

export type LoadResult = Omit<BridgeLoadResult, 'blockers'> & {
  blockers: Blocker[];
};

export type CompanyIdentity = {
  slug: string;
  displayName: string;
  legalName: string;
  ticker: string;
  marketDataSymbol: string;
};

export type RunSummary = {
  runId: string;
  runStatus: string;
  currentStage: StageCode | null;
  dataCutoff: string;
  stateVersion: number;
  loadAvailable: boolean;
};

export type ArtifactMeta = {
  artifactId: string;
  stageCode: StageCode;
  artifactType: string;
  logicalName: string;
  authorityState: 'AUTHORITATIVE' | 'CHECKPOINT';
  artifactStatus: 'SEALED';
  availabilityState: 'AVAILABLE';
  storageBackend: string;
};

export type UiDossierShell = {
  identity: CompanyIdentity;
  runSummaries: RunSummary[];
};

export type UiDossier = UiDossierShell & {
  loadResult: LoadResult;
  artifactCatalog: Record<string, ArtifactMeta>;
};

export type MockDossier = UiDossier & {
  primaryRunId: string;
};
