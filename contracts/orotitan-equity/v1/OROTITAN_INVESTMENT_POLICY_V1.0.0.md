# OROTITAN_INVESTMENT_POLICY_V1.0.0

## Status

```text
POLICY_VERSION
= OROTITAN_INVESTMENT_POLICY_V1.0.0

STATUS
= AUTHORITATIVE / LOCKED
```

This artifact fixes only the project investment-policy inputs required by the OroTitan Equity Research V1 execution contract. It does not reopen or modify frozen research methodology.

## Policy values

```text
REQUIRED_RETURN_H
= 10.0%

STRONG_RETURN_THRESHOLD
= 12.5%

EXCEPTIONAL_RETURN_THRESHOLD
= 15.0%
```

Required hierarchy:

```text
10.0 < 12.5 < 15.0
= PASS
```

## Execution semantics

`REQUIRED_RETURN_H` is the investor required-return input used by the frozen OVS expected-return delta and by the Price Ladder. It is not the economic discount rate.

`STRONG_RETURN_THRESHOLD` and `EXCEPTIONAL_RETURN_THRESHOLD` are Price Ladder / opportunity-threshold policy inputs. They do not alter the frozen OVS anchor curve, OVS weights, caps, Investment Score formula, certification architecture, or OroTitan terminal gate.

All three values are global and versioned for V1. Company-specific analysis may calculate expected returns and prices, but may not replace these policy values.

## Protected invariants

```text
ECONOMIC DISCOUNT RATE
!= INVESTOR REQUIRED RETURN

OVS FORMULA
= UNCHANGED

OVS ANCHORS / CAPS
= UNCHANGED

INVESTMENT SCORE FORMULA
= UNCHANGED

OROTITAN TERMINAL GATE
= UNCHANGED

PHASE-4 PAYLOAD SHAPE
= UNCHANGED
```

No Supabase data mutation is authorized by this policy artifact.
