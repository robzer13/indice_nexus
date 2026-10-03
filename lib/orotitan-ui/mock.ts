import type { ArtifactMeta, ArtifactRef, MockDossier, RunSummary } from './types';

const deepManifest: ArtifactRef = {
  artifact_id: 'c617ed89-b842-44b5-89c8-07d8edb5cf0c',
  version: 1,
  content_sha256: '45c1e22621d1f8af0f734c36b4df6e056543b5cd4dd26dae806ed562f525164e',
  required_authority_class: 'CHECKPOINT_STAGE_OUTPUT',
};

const deepShareCount: ArtifactRef = {
  artifact_id: '82c96b97-5790-4a7f-a36b-21f6ea7d50d4',
  version: 1,
  content_sha256: '9eb0ef5ab8684056192bad416217235a1d8e4ea4f0b5f376fab771210ab7f392',
  required_authority_class: 'CHECKPOINT_STAGE_OUTPUT',
};

const deepFundamentals: ArtifactRef = {
  artifact_id: '42869ed1-608e-4244-9cf3-e71273ef4bb6',
  version: 1,
  content_sha256: 'b7d6cdf72a4b175c6eb85d6f0f914e7785cc4b493c7f3d3644ffaef4e6403e2b',
  required_authority_class: 'CHECKPOINT_STAGE_OUTPUT',
};

const deepValuationBlocker: ArtifactRef = {
  artifact_id: '24e6d95b-ced7-4f0a-bed5-9851e2fbea9f',
  version: 1,
  content_sha256: 'e77a210c6a313c71258b969798fe07dbd943c87b44203f01631288938721de3d',
  required_authority_class: 'CHECKPOINT_STAGE_OUTPUT',
};

const researchRefs: ArtifactRef[] = [
  {
    artifact_id: '2d355b27-d599-49d0-b24e-20049cde3989',
    version: 1,
    content_sha256: '2c6612e0455818a3154173d9216a8b89f317c8399bfc23e00ea4a263c0169b32',
    required_authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
  },
  {
    artifact_id: 'f87f30f6-0e96-4c39-9556-6f4cbb8596b4',
    version: 1,
    content_sha256: 'f17b4b30915b0e8665e60b4858cc6a7586c45afbe7ed91ea1b0c4238d7688960',
    required_authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
  },
  {
    artifact_id: '4ee67a02-d183-4ad8-bc1f-881a3b795f6d',
    version: 1,
    content_sha256: 'a78133a32c36fdd8cf7b13a018a55327982dbab9e25b4b5633f6338d5c13eec4',
    required_authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
  },
  {
    artifact_id: 'ed8f6fc3-3067-4d19-93a7-ec49892ec037',
    version: 1,
    content_sha256: '2ebb1e26ad52faff693aa9428385cf946d7b9d3ce680a39b05680dc7eaa53c6c',
    required_authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
  },
  {
    artifact_id: '49a74184-a24f-44ad-a8f7-793df62eb69d',
    version: 1,
    content_sha256: '610db79948705aa99d24039c6174ca4fef258b87b79e0b578b799a8c17da9567',
    required_authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
  },
  {
    artifact_id: 'd79d9d15-0088-4ffa-8e50-f4cdf18c25ae',
    version: 1,
    content_sha256: '51a1eced006a9016bf64e4ed73372092cafc419a77f5479f86a40836f6685642',
    required_authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
  },
  {
    artifact_id: 'dc32913e-6b55-4328-ae3e-396168ecc200',
    version: 1,
    content_sha256: '5bd4c4407599738a675112d698821e07430673e48236db7efc917695a5027120',
    required_authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
  },
  {
    artifact_id: '58571b3c-e758-4a09-8be4-1715bd37482c',
    version: 1,
    content_sha256: '49fbcd13ccb4ddd81ddc3c7d6486951e398d9494b12648c544595c3bd7fa8596',
    required_authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
  },
];

const meta = (
  artifactId: string,
  stageCode: 'RESEARCH' | 'DEEP_DIVE',
  artifactType: string,
  logicalName: string,
  authorityState: 'AUTHORITATIVE' | 'CHECKPOINT',
): ArtifactMeta => ({
  artifactId,
  stageCode,
  artifactType,
  logicalName,
  authorityState,
  artifactStatus: 'SEALED',
  availabilityState: 'AVAILABLE',
  storageBackend: 'PRIVATE_GITHUB',
});

const artifactCatalog: Record<string, ArtifactMeta> = {
  [deepManifest.artifact_id]: meta(deepManifest.artifact_id, 'DEEP_DIVE', 'DEEP_DIVE_STAGE_MANIFEST', 'deep_dive_stage_manifest', 'CHECKPOINT'),
  [deepShareCount.artifact_id]: meta(deepShareCount.artifact_id, 'DEEP_DIVE', 'ECONOMIC_SHARE_COUNT_CLOSURE_RECORD', 'Veolia Economic Share Count Closure - 2026-09-22', 'CHECKPOINT'),
  [deepFundamentals.artifact_id]: meta(deepFundamentals.artifact_id, 'DEEP_DIVE', 'FUNDAMENTALS_LOCK', 'Veolia Fundamentals Lock - Same-Cutoff Revalidation', 'CHECKPOINT'),
  [deepValuationBlocker.artifact_id]: meta(deepValuationBlocker.artifact_id, 'DEEP_DIVE', 'VALUATION_BLOCKER_CHECK_RECORD', 'valuation_blocker_check_record', 'CHECKPOINT'),
  '2d355b27-d599-49d0-b24e-20049cde3989': meta('2d355b27-d599-49d0-b24e-20049cde3989', 'RESEARCH', 'ANALYSIS_INPUT_LOCK', 'Veolia Analysis Input Lock - Same-Cutoff Revalidation', 'AUTHORITATIVE'),
  'f87f30f6-0e96-4c39-9556-6f4cbb8596b4': meta('f87f30f6-0e96-4c39-9556-6f4cbb8596b4', 'RESEARCH', 'CONFLICT_LEDGER', 'Veolia Conflict Ledger - Same-Cutoff Revalidation', 'AUTHORITATIVE'),
  '4ee67a02-d183-4ad8-bc1f-881a3b795f6d': meta('4ee67a02-d183-4ad8-bc1f-881a3b795f6d', 'RESEARCH', 'DD_INPUT_SUFFICIENCY_RECORD', 'Veolia DD Input Sufficiency - Same-Cutoff Revalidation', 'AUTHORITATIVE'),
  'ed8f6fc3-3067-4d19-93a7-ec49892ec037': meta('ed8f6fc3-3067-4d19-93a7-ec49892ec037', 'RESEARCH', 'EVIDENCE_LEDGER', 'Veolia Evidence Ledger - Same-Cutoff Revalidation', 'AUTHORITATIVE'),
  '49a74184-a24f-44ad-a8f7-793df62eb69d': meta('49a74184-a24f-44ad-a8f7-793df62eb69d', 'RESEARCH', 'MATERIAL_RESEARCH_HYPOTHESIS_REGISTER', 'Veolia Material Research Hypothesis Register - Same-Cutoff Revalidation', 'AUTHORITATIVE'),
  'd79d9d15-0088-4ffa-8e50-f4cdf18c25ae': meta('d79d9d15-0088-4ffa-8e50-f4cdf18c25ae', 'RESEARCH', 'RESEARCH_GAP_REGISTER', 'Veolia Research Gap Register - Same-Cutoff Revalidation', 'AUTHORITATIVE'),
  'dc32913e-6b55-4328-ae3e-396168ecc200': meta('dc32913e-6b55-4328-ae3e-396168ecc200', 'RESEARCH', 'RESEARCH_SOURCE_MANIFEST', 'Veolia Research Source Manifest - Same-Cutoff Revalidation', 'AUTHORITATIVE'),
  '58571b3c-e758-4a09-8be4-1715bd37482c': meta('58571b3c-e758-4a09-8be4-1715bd37482c', 'RESEARCH', 'RESEARCH_STAGE_MANIFEST', 'Veolia Successor Research Stage Manifest', 'AUTHORITATIVE'),
};

export const VEOLIA_MOCK_DOSSIER: MockDossier = {
  identity: {
    slug: 'veolia',
    displayName: 'Veolia',
    legalName: 'Veolia Environnement S.A.',
    ticker: 'VIE',
    marketDataSymbol: 'VIE.PA',
  },
  runSummaries: [
    {
      runId: 'a4cf9002-b52d-4440-8902-adc70bd777dd',
      runStatus: 'BLOCKED',
      currentStage: 'DEEP_DIVE',
      dataCutoff: '2026-09-22',
      stateVersion: 7,
      loadAvailable: true,
    },
    {
      runId: 'd56a0bc9-4e80-48b3-b326-74d1fea15e63',
      runStatus: 'ACTIVE',
      currentStage: 'RESEARCH',
      dataCutoff: '2026-09-22',
      stateVersion: 2,
      loadAvailable: false,
    },
    {
      runId: '76ee7eb1-8658-4038-83bf-e2c951db731a',
      runStatus: 'BLOCKED',
      currentStage: 'DEEP_DIVE',
      dataCutoff: '2026-09-22',
      stateVersion: 8,
      loadAvailable: false,
    },
  ],
  primaryRunId: 'a4cf9002-b52d-4440-8902-adc70bd777dd',
  loadResult: {
    contract_version: '0.1.0',
    operation: 'LOAD_RESULT',
    mutation_allowed: false,
    issuer_id: 'b7dd4445-ae1e-4f33-985b-216d83e3fbde',
    security_id: '71bc5f4a-662e-4628-872f-c86d292a81be',
    dossier_id: '59a08ccd-ceb4-42c2-ab04-aabec5cdc330',
    run_id: 'a4cf9002-b52d-4440-8902-adc70bd777dd',
    run_status: 'BLOCKED',
    run_state_version: 7,
    run_type: 'INITIAL',
    canonical_mode: 'ANALYZE',
    data_cutoff: '2026-09-22',
    contract_set_sha256: '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2',
    current_stage: 'DEEP_DIVE',
    stage: {
      stage_code: 'DEEP_DIVE',
      stage_revision: 1,
      lifecycle_status: 'BLOCKED',
      stage_state_version: 4,
      handoff_gate_state: 'NOT_EVALUATED',
      active_manifest: deepManifest,
    },
    blockers: [
      {
        code: 'ECONOMIC_SHARE_COUNT_UNRESOLVED',
        classification: 'VALUATION_INPUT_UNRESOLVED',
        scope: 'VALUATION_PER_SHARE_AND_DEPENDENT_OUTPUTS',
        detail: 'At the governed valuation/share-count date 2026-09-22, neither an exact economic share count nor a complete finite rigorous bound is provable from cutoff-compliant evidence. The complete 2026-09-16 bounded representation cannot be transported to 2026-09-22.',
        diagnostic: 'STALE_DENOMINATOR_TRANSPORT_FORBIDDEN',
        resolution_required: 'Close every denominator-changing movement class through 2026-09-22 with exact or rigorous hard-bound evidence under the frozen share-count method.',
      },
    ],
    artifact_index: [deepManifest, deepShareCount, deepFundamentals, deepValuationBlocker, ...researchRefs],
    process_state_artifact: null,
    context_plan: {
      l0: [deepManifest],
      l1: [deepManifest, deepShareCount, deepFundamentals, deepValuationBlocker],
      l2: [],
      l3: [],
    },
  },
  artifactCatalog,
};

export function getMockDossier(slug: string): MockDossier | null {
  return slug.toLowerCase() === 'veolia' ? VEOLIA_MOCK_DOSSIER : null;
}

export type MockRunSelection =
  | { kind: 'select' }
  | { kind: 'available'; summary: RunSummary }
  | { kind: 'unavailable'; summary: RunSummary }
  | { kind: 'unknown' };

export function resolveMockRunSelection(
  dossier: MockDossier,
  requestedRun: string | null,
): MockRunSelection {
  if (!requestedRun) return { kind: 'select' };
  const summary = dossier.runSummaries.find((run) => run.runId === requestedRun);
  if (!summary) return { kind: 'unknown' };
  if (requestedRun === dossier.primaryRunId && summary.loadAvailable) {
    return { kind: 'available', summary };
  }
  return { kind: 'unavailable', summary };
}
