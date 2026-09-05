import type { QuoteUnit } from '@/lib/domain/types';

export interface PriceFormatInput {
  value: number | null;
  currency: string;
  quoteUnit: QuoteUnit;
  priceDecimals: number;
}

function formatNumber(value: number, decimals: number): string {
  return new Intl.NumberFormat('fr-FR', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

export function formatPrice({
  value,
  currency,
  quoteUnit,
  priceDecimals,
}: PriceFormatInput): string {
  if (value === null || !Number.isFinite(value) || value <= 0) {
    return 'Non disponible';
  }

  const formatted = formatNumber(value, priceDecimals);

  if (quoteUnit === 'MINOR') {
    if (currency === 'GBP') {
      return `${formatted}p`;
    }
    return `${formatted} ${currency} (unité mineure)`;
  }

  switch (currency) {
    case 'EUR':
      return `${formatted} €`;
    case 'USD':
      return `$${formatted}`;
    case 'GBP':
      return `£${formatted}`;
    case 'NOK':
      return `${formatted} NOK`;
    default:
      return `${formatted} ${currency}`;
  }
}
