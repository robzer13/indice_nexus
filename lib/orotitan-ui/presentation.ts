import type { Blocker, StageCode, StageLifecycle } from './types';

const lifecycleLabels: Record<StageLifecycle, string> = {
  NOT_STARTED: 'Non démarré',
  IN_PROGRESS: 'En cours',
  PAUSED: 'En pause',
  BLOCKED: 'Bloqué',
  COMPLETE: 'Terminé',
};

const stageLabels: Record<StageCode, string> = {
  RESEARCH: 'Recherche',
  DEEP_DIVE: 'Deep Dive',
  INTEGRATION: 'Intégration',
};

const blockerTitles: Record<string, string> = {
  ECONOMIC_SHARE_COUNT_UNRESOLVED: "Nombre économique d'actions non résolu",
};

const stages: StageCode[] = ['RESEARCH', 'DEEP_DIVE', 'INTEGRATION'];

export function lifecycleLabel(value: StageLifecycle): string {
  return lifecycleLabels[value];
}

export function stageLabel(value: StageCode): string {
  return stageLabels[value];
}

export function runStatusLabel(value: string | null): string {
  if (value === null) return 'Indisponible';
  const labels: Record<string, string> = {
    ACTIVE: 'En cours',
    BLOCKED: 'Bloqué',
    PUBLISHED: 'Publié',
    CANCELLED: 'Annulé',
  };
  return labels[value] ?? value.replaceAll('_', ' ');
}

export function blockerTitle(blocker: Blocker): string {
  return blockerTitles[blocker.code] ?? blocker.code;
}

export function shortId(value: string | null | undefined, visible = 8): string {
  if (!value) return '—';
  if (value.length <= visible) return value;
  return value.slice(0, visible) + '…';
}

export function formatCutoff(value: string | null): string {
  if (!value) return '—';
  const parts = value.split('-');
  if (parts.length !== 3) return value;
  return parts[2] + '/' + parts[1] + '/' + parts[0];
}

export function deriveStageStates(
  currentStage: StageCode | null,
  currentLifecycle: StageLifecycle | null,
): Array<{ stage: StageCode; lifecycle: StageLifecycle }> {
  if (!currentStage || !currentLifecycle) {
    return stages.map((stage) => ({ stage, lifecycle: 'NOT_STARTED' }));
  }

  const currentIndex = stages.indexOf(currentStage);
  return stages.map((stage, index) => ({
    stage,
    lifecycle:
      index < currentIndex ? 'COMPLETE' : index === currentIndex ? currentLifecycle : 'NOT_STARTED',
  }));
}

export function buildRunHref(
  path: string,
  runId: string | null,
  extra: Record<string, string | null | undefined> = {},
): string {
  const params: string[] = [];
  if (runId) params.push('run=' + encodeURIComponent(runId));
  for (const [key, value] of Object.entries(extra)) {
    if (value) params.push(encodeURIComponent(key) + '=' + encodeURIComponent(value));
  }
  return params.length > 0 ? path + '?' + params.join('&') : path;
}
