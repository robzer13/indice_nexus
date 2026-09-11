import type { ScoreRange } from "./semantic-states";

const EXPECTED_RETURN_ANCHORS = [
  [-10, 0], [-8, 10], [-6, 20], [-4, 35], [-2, 50], [0, 70],
  [2, 82], [4, 90], [6, 95], [8, 100],
] as const;

export function scoreExpectedReturnDelta(deltaPercentagePoints: number): number {
  if (!Number.isFinite(deltaPercentagePoints)) throw new Error("Expected-return delta must be finite");
  if (deltaPercentagePoints <= -10) return 0;
  if (deltaPercentagePoints >= 8) return 100;
  for (let index = 1; index < EXPECTED_RETURN_ANCHORS.length; index += 1) {
    const [rightDelta, rightScore] = EXPECTED_RETURN_ANCHORS[index];
    const [leftDelta, leftScore] = EXPECTED_RETURN_ANCHORS[index - 1];
    if (deltaPercentagePoints <= rightDelta) {
      return leftScore + ((deltaPercentagePoints - leftDelta) / (rightDelta - leftDelta)) * (rightScore - leftScore);
    }
  }
  throw new Error("Unreachable expected-return interval");
}

export function scoreExpectedReturnRange(delta: number | ScoreRange): number | ScoreRange {
  if (typeof delta === "number") return scoreExpectedReturnDelta(delta);
  return { min: scoreExpectedReturnDelta(delta.min), max: scoreExpectedReturnDelta(delta.max) };
}
