export function getFairValueUpsidePct(
  currentPrice: number | null,
  fairValueBase: number | null,
): number | null {
  if (
    currentPrice === null ||
    fairValueBase === null ||
    !Number.isFinite(currentPrice) ||
    !Number.isFinite(fairValueBase) ||
    currentPrice <= 0 ||
    fairValueBase <= 0
  ) {
    return null;
  }

  return (fairValueBase / currentPrice - 1) * 100;
}
