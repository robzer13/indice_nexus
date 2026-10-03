import 'server-only';

import type { ArtifactRow } from '../orotitan-equity/post-c7/chatgpt-supabase-bridge';
import { ArtifactContentError } from './artifact-reader';

const DEFAULT_PRIVATE_REPOSITORY = 'robzer13/real-orotitan';
const REPOSITORY = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;

function allowedRepositories(): Set<string> {
  const configured = process.env.OROTITAN_PRIVATE_GITHUB_REPOSITORIES?.trim();
  return new Set(
    (configured || DEFAULT_PRIVATE_REPOSITORY)
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean),
  );
}

function encodeGithubPath(path: string): string {
  return path
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/');
}

export async function readPrivateGithubArtifactBytes(
  row: ArtifactRow,
): Promise<Uint8Array> {
  const token = process.env.OROTITAN_PRIVATE_GITHUB_TOKEN?.trim();
  if (!token) {
    throw new ArtifactContentError(
      'ARTIFACT_READER_UNCONFIGURED',
      'Private GitHub artifact reader is not configured',
    );
  }

  const repository = row.github_repository;
  const path = row.github_path;
  const commit = row.github_commit_sha;
  if (!repository || !path || !commit || !REPOSITORY.test(repository)) {
    throw new ArtifactContentError(
      'ARTIFACT_REGISTRY_MISMATCH',
      'Private GitHub artifact coordinates are invalid',
    );
  }

  if (!allowedRepositories().has(repository)) {
    throw new ArtifactContentError(
      'ARTIFACT_STORAGE_UNSUPPORTED',
      'Private GitHub repository is outside the configured allow-list',
    );
  }

  const [owner, repo] = repository.split('/');
  const url =
    'https://api.github.com/repos/' +
    encodeURIComponent(owner) +
    '/' +
    encodeURIComponent(repo) +
    '/contents/' +
    encodeGithubPath(path) +
    '?ref=' +
    encodeURIComponent(commit);

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'GET',
      headers: {
        Accept: 'application/vnd.github.raw+json',
        Authorization: 'Bearer ' + token,
        'X-GitHub-Api-Version': '2022-11-28',
      },
      cache: 'no-store',
    });
  } catch {
    throw new ArtifactContentError(
      'ARTIFACT_STORAGE_FETCH_FAILED',
      'Private GitHub artifact fetch failed',
    );
  }

  if (!response.ok) {
    throw new ArtifactContentError(
      'ARTIFACT_STORAGE_FETCH_FAILED',
      'Private GitHub artifact fetch returned HTTP ' + response.status,
    );
  }

  return new Uint8Array(await response.arrayBuffer());
}
