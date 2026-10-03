import { requireAdminSession } from '@/lib/auth/admin-session';
import {
  ArtifactContentError,
  type ArtifactContentErrorCode,
} from '@/lib/orotitan-ui/artifact-reader';
import { getVerifiedUiArtifactContent } from '@/lib/orotitan-ui/server-data';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

function errorStatus(code: ArtifactContentErrorCode): number {
  if (code === 'ARTIFACT_NOT_LOAD_AUTHORIZED') return 404;
  if (code === 'ARTIFACT_READER_UNCONFIGURED') return 503;
  if (code === 'ARTIFACT_STORAGE_FETCH_FAILED') return 502;
  if (code === 'ARTIFACT_STORAGE_UNSUPPORTED') return 422;
  if (
    code === 'ARTIFACT_REGISTRY_MISMATCH' ||
    code === 'ARTIFACT_INTEGRITY_FAILED' ||
    code === 'ARTIFACT_CONTENT_INVALID'
  ) {
    return 409;
  }
  return 500;
}

function json(body: unknown, status = 200): Response {
  return Response.json(body, {
    status,
    headers: {
      'Cache-Control': 'private, no-store, max-age=0',
      Vary: 'Cookie',
    },
  });
}

export async function GET(
  request: Request,
  context: { params: Promise<{ artifactId: string }> },
): Promise<Response> {
  try {
    await requireAdminSession();
  } catch {
    return json(
      {
        error: 'UNAUTHORIZED',
        message: 'Private artifact content requires an authenticated admin session',
      },
      401,
    );
  }

  const { artifactId } = await context.params;
  const query = new URL(request.url).searchParams;
  const issuer = query.get('issuer')?.trim() ?? '';
  const runId = query.get('run')?.trim() ?? '';
  const versionRaw = query.get('version')?.trim() ?? '';
  const version = Number(versionRaw);

  if (
    !issuer ||
    !runId ||
    !artifactId ||
    !Number.isInteger(version) ||
    version < 1
  ) {
    return json(
      {
        error: 'INVALID_REQUEST',
        message: 'issuer, run, artifactId and positive integer version are required',
      },
      400,
    );
  }

  try {
    const content = await getVerifiedUiArtifactContent({
      issuerQuery: issuer,
      runId,
      artifactId,
      version,
    });
    return json(content);
  } catch (error) {
    if (error instanceof ArtifactContentError) {
      return json(
        {
          error: error.code,
          message: error.message,
        },
        errorStatus(error.code),
      );
    }

    return json(
      {
        error: 'ARTIFACT_RESOLUTION_FAILED',
        message: 'Verified artifact resolution failed',
      },
      500,
    );
  }
}
