export type StageCode = 'RESEARCH' | 'DEEP_DIVE' | 'INTEGRATION';
export type StageLifecycle = 'NOT_STARTED' | 'IN_PROGRESS' | 'PAUSED' | 'BLOCKED' | 'COMPLETE';

export type ArtifactRef = {
  artifact_id: string;
  version: number;
  content_sha256?: string | null;
  required_authority_class?: string | null;
};

export type Blocker = {
  code: string;
  classification?: string;
  scope?: string;
  detail?: string;
  diagnostic?: string;
  resolution_required?: string;
  [key: string]: unknown;
};

export type LoadResult = {
  contract_version: '0.1.0';
  operation: 'LOAD_RESULT';
  mutation_allowed: false;
  issuer_id: string;
  security_id: string | null;
  dossier_id: string;
  run_id: string | null;
  run_status: string | null;
  run_state_version: number | null;
  run_type: string | null;
  canonical_mode: string | null;
  data_cutoff: string | null;
  contract_set_sha256: string | null;
  current_stage: StageCode | null;
  stage: {
    stage_code: StageCode;
    stage_revision: number;
    lifecycle_status: StageLifecycle;
    stage_state_version: number;
    handoff_gate_state: 'NOT_EVALUATED' | 'NO' | 'YES';
    active_manifest: ArtifactRef | null;
  } | null;
  blockers: Blocker[];
  artifact_index: ArtifactRef[];
  process_state_artifact?: ArtifactRef | null;
  context_plan: {
    l0: ArtifactRef[];
    l1: ArtifactRef[];
    l2: ArtifactRef[];
    l3: ArtifactRef[];
  };
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
  currentStage: StageCode;
  dataCutoff: string;
  stateVersion: number;
  detailedMockAvailable: boolean;
};

export type ArtifactMeta = {
  artifactId: string;
  stageCode: StageCode;
  artifactType: string;
  logicalName: string;
  authorityState: 'AUTHORITATIVE' | 'CHECKPOINT';
  artifactStatus: 'SEALED';
  availabilityState: 'AVAILABLE';
  storageBackend: 'PRIVATE_GITHUB';
};

export type MockDossier = {
  identity: CompanyIdentity;
  runSummaries: RunSummary[];
  primaryRunId: string;
  loadResult: LoadResult;
  artifactCatalog: Record<string, ArtifactMeta>;
};
