export const OROTITAN_INVESTMENT_POLICY_V1_0_0 = {
  policyVersion: "OROTITAN_INVESTMENT_POLICY_V1.0.0",
  requiredReturnH: 10.0,
  strongReturnThreshold: 12.5,
  exceptionalReturnThreshold: 15.0,
} as const;

export interface InvestmentPolicyV1Input {
  requiredReturnH: unknown;
  strongReturnThreshold: unknown;
  exceptionalReturnThreshold: unknown;
}

function assertExactPolicyNumber(name: string, actual: unknown, expected: number): void {
  if (typeof actual !== "number" || !Number.isFinite(actual) || actual !== expected) {
    throw new Error(`${name} must equal ${expected} under ${OROTITAN_INVESTMENT_POLICY_V1_0_0.policyVersion}`);
  }
}

export function assertInvestmentPolicyV1(input: InvestmentPolicyV1Input): void {
  assertExactPolicyNumber("price_ladder.required_return_h", input.requiredReturnH, OROTITAN_INVESTMENT_POLICY_V1_0_0.requiredReturnH);
  assertExactPolicyNumber("price_ladder.strong_return_threshold", input.strongReturnThreshold, OROTITAN_INVESTMENT_POLICY_V1_0_0.strongReturnThreshold);
  assertExactPolicyNumber("price_ladder.exceptional_return_threshold", input.exceptionalReturnThreshold, OROTITAN_INVESTMENT_POLICY_V1_0_0.exceptionalReturnThreshold);
}
