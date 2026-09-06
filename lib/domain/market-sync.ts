export const MARKET_SYNC_COOLDOWN_MS = 65_000;

export function getMarketSyncCooldownRemainingMs(
  lastFinishedAt: string,
  nowMs = Date.now(),
): number {
  const lastFinishedMs = new Date(lastFinishedAt).getTime();
  if (!Number.isFinite(lastFinishedMs)) return 0;
  return Math.max(0, MARKET_SYNC_COOLDOWN_MS - (nowMs - lastFinishedMs));
}
