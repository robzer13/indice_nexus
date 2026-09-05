import { createHash, timingSafeEqual } from 'node:crypto';
import { NextRequest, NextResponse } from 'next/server';
import { getMarketDataCompanies, insertMarketPrice } from '@/lib/data/market-prices';
import { fetchTwelveDataPrice } from '@/lib/market/twelve-data';

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
  if (!secret) return false;
  const authorization = request.headers.get('authorization');
  if (!authorization?.startsWith('Bearer ')) return false;
  return safeEqual(authorization.slice('Bearer '.length), secret);
}

export async function GET(request: NextRequest) {
  if (!isAuthorized(request)) {
    return NextResponse.json({ ok: false, error: 'Unauthorized' }, { status: 401 });
  }

  const startedAt = new Date().toISOString();
  const companies = await getMarketDataCompanies();
  const results: Array<{
    slug: string;
    symbol: string;
    status: 'inserted' | 'failed';
    price?: number;
    error?: string;
  }> = [];

  for (const company of companies) {
    try {
      if (!company.market_data_symbol) {
        results.push({ slug: company.slug, symbol: '', status: 'failed', error: 'Missing market_data_symbol' });
        continue;
      }
      if (!Number.isFinite(company.market_data_multiplier) || company.market_data_multiplier <= 0) {
        results.push({ slug: company.slug, symbol: company.market_data_symbol, status: 'failed', error: 'Invalid market_data_multiplier' });
        continue;
      }

      const provider = await fetchTwelveDataPrice(company.market_data_symbol);
      const normalizedPrice = provider.providerPrice * company.market_data_multiplier;
      if (!Number.isFinite(normalizedPrice) || normalizedPrice <= 0) {
        throw new Error('Normalized price is invalid');
      }

      await insertMarketPrice({
        companyId: company.id,
        price: normalizedPrice,
        asOf: provider.fetchedAt,
        source: 'TWELVE_DATA',
        raw: provider.raw,
      });
      results.push({ slug: company.slug, symbol: company.market_data_symbol, status: 'inserted', price: normalizedPrice });
    } catch (error) {
      results.push({
        slug: company.slug,
        symbol: company.market_data_symbol,
        status: 'failed',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  const inserted = results.filter((result) => result.status === 'inserted').length;
  const failed = results.length - inserted;

  return NextResponse.json({
    ok: failed === 0,
    startedAt,
    finishedAt: new Date().toISOString(),
    companies: results.length,
    inserted,
    failed,
    results,
  }, { status: failed === 0 ? 200 : 207 });
}
