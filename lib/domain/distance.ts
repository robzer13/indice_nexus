export function getDistanceO90(
  currentPrice: number | null,
  priceO90: number | null,
): number | null {
  if (
    currentPrice === null ||
    priceO90 === null ||
    !Number.isFinite(currentPrice) ||
    !Number.isFinite(priceO90) ||
    currentPrice <= 0 ||
    priceO90 <= 0
  ) {
    return null;
  }

  return (priceO90 / currentPrice - 1) * 100;
}
