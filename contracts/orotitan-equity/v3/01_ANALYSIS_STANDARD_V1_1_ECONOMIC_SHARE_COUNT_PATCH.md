# OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1 — FREEZE V1.0

**Logical Contract Pin:** analysis_standard
**Base incorporated by reference:** 01_ANALYSIS_STANDARD_V1 @ SHA256 9283a0df395d4596c93cf6cfad9644ce114b343ee80e6e0a81e8f94f15d1e3df
**Governance class:** GLOBAL_VALUATION_METHODOLOGY
**Subject:** POINT_IN_TIME_ECONOMIC_SHARE_COUNT_UNCERTAINTY
**Status:** FROZEN — V1.0
**Classification:** G2 — RIGOROUS_BOUNDED_DENOMINATOR_PERMITTED
**Scope of supersession:** only uncertain point-in-time ECONOMIC_SHARE_COUNT definition, admission, bound construction and downstream propagation. All other base Analysis Standard clauses remain unchanged.

## 1. Governing policy

The representation order is:

EXACT_SCALAR
-> if exact closure fails, attempt RIGOROUS_BOUND
-> if rigorous bound fails, UNKNOWN.

A rigorous bound may admit Valuation. Distributional/probability-weighted share counts are not authorized.

MIDPOINT_AS_POINT_ESTIMATE = FORBIDDEN
ENDPOINT_AS_POINT_ESTIMATE = FORBIDDEN
STALE_EXACT_SUBSTITUTION = FORBIDDEN
WEIGHTED_AVERAGE_EPS_SUBSTITUTION = FORBIDDEN
VENDOR_ESTIMATE_SUBSTITUTION = FORBIDDEN

## 2. ECONOMIC_SHARE_COUNT

ECONOMIC_SHARE_COUNT(t) is the number of ordinary-share-equivalent residual equity units economically outstanding to external holders at the close of t, on the analyzed security/economic-equivalence basis.

Exclude issuer/group treasury or own shares and cancelled/retired shares. Apply exact split, consolidation, share-class and depositary-receipt conversion factors.

This object is distinct from issued shares, treasury shares, basic or diluted weighted-average EPS shares, fully diluted/if-converted shares, authorized shares and vendor market-cap denominators.

Unsettled options, RSUs, convertibles and other potential ordinary shares are not inserted into the current point-in-time denominator merely because they are dilutive for EPS. They enter only when share delivery/issuance becomes economically effective. Future dilution remains a forecast/claim input.

## 3. Effective date

Keep distinct DATA_CUTOFF, REFERENCE_PRICE_DATE, CALCULATION_DATE and SHARE_COUNT_AS_OF_DATE.

For the current OroTitan valuation architecture:

VALUATION_DATE = REFERENCE_PRICE_DATE
SHARE_COUNT_AS_OF_DATE = REFERENCE_PRICE_DATE.

This same current denominator basis governs instantaneous market capitalization, intrinsic value per share, expected return, Reverse DCF, Price Ladder and MOS.

CALCULATION_DATE does not move the measurement date. DATA_CUTOFF limits admissible knowledge.

## 4. Share-state model and exhaustive movement taxonomy

Use conceptual buckets:
U = unissued/available;
X = externally economically outstanding;
T = treasury/own shares excluded from external ownership;
R = retired/cancelled.

ECONOMIC_SHARE_COUNT = X.

Canonical movements:
- ordinary issuance / primary placement: U->X, +q;
- settled issuer buyback: X->T, -q;
- treasury disposal/reissue: T->X, +q;
- cancellation of treasury shares: T->R, zero change to X;
- direct external cancellation/redemption: X->R, -q;
- employee plan release: T->X or U->X according to delivery source;
- cash-settled award: zero;
- option exercise / RSU settlement: T->X or U->X according to source;
- convertible settlement in shares: T/U->X; cash settlement: zero;
- scrip/bonus/stock dividend: exact new units or exact factor;
- acquisition consideration shares: T/U->X;
- rights issue: actual issued shares only; authorization is not execution;
- split/reverse split/consolidation: exact multiplicative transformation;
- ADR/depositary creation/cancellation: no issuer economic-share movement; apply exact ratio for per-security basis;
- other corporate actions: map to exact bucket transition before use.

State transitions, not labels, determine sign and prevent double counting.

## 5. Exactness and complete bridge

An exact item requires exact quantity/conversion, correct security/share class, effective/settlement date, economic source/destination effect, qualifying primary/root provenance and no unresolved scope ambiguity.

Rounded disclosures such as 384.1m are not exact.

Exact closure requires:
EXACT_ANCHOR
+ ALL denominator-changing movements through target date
= ECONOMIC_SHARE_COUNT(target).

Every relevant movement class must be EXACT_COMPLETE or proven NOT_APPLICABLE/ZERO. Silence is not proof of zero.

## 6. Rigorous bound

Let Omega be all share-state paths consistent with the exact anchor, hard constraints, dates, state conservation and correlations.

S_LOW = min over Omega of X_target.
S_HIGH = max over Omega of X_target.

Bound admission requires:
- non-empty feasible set;
- finite positive endpoints;
- BOUND_EFFECTIVE_DATE = SHARE_COUNT_AS_OF_DATE;
- BOUND_COMPLETENESS = COMPLETE;
- UNBOUNDED_MOVEMENT_CLASSES = 0;
- provenance and constraint IDs retained.

If any relevant movement class is unbounded, BOUNDED_DENOMINATOR = NOT_PROVABLE and ECONOMIC_SHARE_COUNT = UNKNOWN.

### Bound inputs

PERMITTED_HARD_BOUND:
- exact executed repurchases;
- exact issue/share-capital/register notices;
- statutory issuance ceilings for the specifically identified legal channel when exact base/headroom, dates and amendments are verified;
- AGM authorization ceilings under the same channel-specific conditions;
- named buyback/program maxima with exact remaining capacity and validity;
- named employee-plan maxima with exact remaining capacity and delivery-channel treatment.

An authorization ceiling constrains only that exact channel. It proves neither actual execution nor issuer-level completeness while another channel is unbounded.

PERMITTED_SUPPORT_ONLY:
- generic rounded interim disclosures;
- company expectations;
- management intentions;
- historical behavior.

A rounded disclosure becomes a hard interval only if the source explicitly supplies a deterministic rounding convention; that explicit convention is a separate hard constraint.

PROHIBITED_AS_BOUND:
- vendor estimates.

## 7. Correlated movements

Preserve state conservation and shared causal dependencies. Do not independently add gross movements when one movement supplies another.

Examples:
X->T buyback then T->X employee release;
X->T then T->R cancellation;
option settlement from T rather than U.

If the same transaction also changes cash/debt used in EV, preserve the joint feasible set downstream.

## 8. Propagation

The uncertainty object is Omega. Reported intervals are closed output hulls over Omega.

For scalar non-negative reference price P and shares [L,H]:
MARKET_CAP = [P*L, P*H].

For scalar non-negative intrinsic equity value E:
INTRINSIC_VALUE_PER_SHARE = [E/H, E/L].

For independent equity-value range [E_L,E_H]:
PER_SHARE_VALUE = [E_L/H, E_H/L].
Correlated variables require joint optimization, not marginal Cartesian products.

Expected return:
if every feasible state has a finite valid shareholder-return solution, report [min IRR, max IRR]; otherwise NOT_ASSESSABLE.

Reverse DCF:
solve the implied variable for every feasible state. Report an interval only if the valid-domain solution is unique and bounded for every state; otherwise NOT_ASSESSABLE.

Price Ladder:
each H / STRONG / EXCEPTIONAL threshold produces a price interval. No midpoint/end point scalarization.

MOS:
if all feasible states map to the same frozen MOS category, emit it; otherwise NOT_ASSESSABLE.

VALUATION_RELIABILITY:
no new automatic HIGH/MEDIUM/LOW cap is created. If required denominator propagation cannot produce finite/determinate required outputs, reliability = NOT_ASSESSABLE. Otherwise apply the existing reliability doctrine with the range retained as a limitation.

OVS:
no formula change. Existing I2 return-range endpoint propagation applies only when MOS and reliability are assessable scalar states. If either is NOT_ASSESSABLE, numeric OVS is prohibited.

INVESTMENT_SCORE:
existing I2 range propagation applies.

INVESTMENT_CLASS:
if the entire score interval lies inside one frozen class band, emit that class. If it crosses a class boundary, emit NOT_AVAILABLE with denominator-uncertainty reason.

Affected gate/predicate:
PASS if true for every feasible state;
FAIL if false for every feasible state;
NOT_ASSESSABLE if truth varies across feasible states.

Existing terminal OroTitan logic remains unchanged.

## 9. Materiality

There is no de-minimis scalarization in V1.0. Any non-zero denominator interval remains a range.

Any interval capable of changing MOS, the required-return hurdle, Investment Class, an OroTitan gate, readiness or another canonical categorical state is decision-material regardless of width.

## 10. UNKNOWN

Allowed primary reasons:
NO_EVIDENCE;
PARTIAL_EVIDENCE;
EXACT_ANCHOR_BUT_INCOMPLETE_BRIDGE;
BOUND_NOT_PROVABLE;
METHODOLOGY_NOT_APPLICABLE.

## 11. Valuation admission

EXACT scalar with complete exact state/bridge -> YES.
RIGOROUS_BOUND with finite positive endpoints, complete provenance and UNBOUNDED_MOVEMENT_CLASSES=0 -> YES.
Any UNKNOWN reason -> NO.
MISSING/malformed/contradictory -> NO.

## 12. Versioning

This is a MATERIAL_METHOD_CHANGE because it newly authorizes rigorous bounded denominator admission.

Existing run contract_pins, contract_set_sha256 and DATA_CUTOFF are immutable. This authority must not be injected into a run pinned to an earlier Contract Set.

A methodology replay requires a controlled successor run with parent_run_id pointing to the historical run. Pure replay preserves the historical DATA_CUTOFF. Consuming later evidence requires an explicit METHODOLOGY_REPLAY_PLUS_INFORMATION_REFRESH classification and later governed cutoff.

No issuer-specific exception is authorized.
