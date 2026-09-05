import { formatPrice } from '@/lib/domain/format-price';
import type { QuoteUnit } from '@/lib/domain/types';

export function PriceDisplay({ value, currency, quoteUnit, priceDecimals, className = '' }: { value: number | null; currency: string; quoteUnit: QuoteUnit; priceDecimals: number; className?: string }) {
  return <span className={`font-mono tabular-nums ${className}`}>{formatPrice({ value, currency, quoteUnit, priceDecimals })}</span>;
}
