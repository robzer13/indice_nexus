import { createHash, timingSafeEqual } from 'node:crypto';
import { NextRequest, NextResponse } from 'next/server';
import { refreshMarketPrices } from '@/lib/market/refresh-prices';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';
export const maxDuration = 60;

function safeEqual(a: string, b: string): boolean {
  const left = createHash('sha256').update(a).digest();
  const right = createHash('sha256').update(b).digest();
  return timingSafeEqual(left, right);
}

function isAuthorized(request: NextRequest): boolean {
  const secret = process.env.CRON_SECRET;
  const authorization = request.headers.get('authorization');
  const bearerPrefixPresent = authorization?.startsWith('Bearer ') ?? false;
  const authorized = secret && bearerPrefixPresent
    ? safeEqual(authorization!.slice('Bearer '.length), secret)
    : false;

  // Temporary diagnostic: only booleans, never credentials, token fragments, or hashes.
  // Vercel uses this user-agent for both scheduled and manual Cron Job invocations.
  if (!authorized && request.headers.get('user-agent') === 'vercel-cron/1.0') {
    console.warn('CRON_AUTH_DIAGNOSTIC', {
      secretConfigured: Boolean(secret),
      authorizationHeaderPresent: authorization !== null,
      bearerPrefixPresent,
      secretHasInvalidHeaderChars: /[\r\n\x00-\x1f\x7f]/.test(secret ?? ''),
    });
  }
  return authorized;
}

export async function GET(request: NextRequest) {
  if (!isAuthorized(request)) {
    return NextResponse.json({ ok: false, error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const result = await refreshMarketPrices('CRON');
    return NextResponse.json(result, { status: result.failed === 0 ? 200 : 207 });
  } catch (error) {
    return NextResponse.json({
      ok: false,
      error: error instanceof Error ? error.message : 'Unexpected market refresh failure',
    }, { status: 500 });
  }
}
