# OROTITAN_VNEXT_ANALYTICAL_MODEL_PROVIDER_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE — LIVE AZURE CALL PENDING  
**Methodology change:** NO  
**Depends on:** Gates 7, 11, 12  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

Gate 13 introduces a replaceable headless model-provider boundary.

The model is an execution dependency for bounded analytical work. It is not an authority over:

```text
RUN STATE
STAGE STATE
REGISTRY
ARTIFACT AUTHORITY
LINEAGE
PUBLICATION
DETERMINISTIC FORMULAS
FROZEN METHODOLOGY
```

Core rule:

```text
ANALYTICAL MODEL
= BOUNDED STRUCTURED INFERENCE PROVIDER
!= ORCHESTRATOR
!= REGISTRY WRITER
!= PUBLICATION AUTHORITY
```

## 1. Provider abstraction

Implementation:

```text
runtime/vnext/analytical-model-provider.ts
```

Provider-neutral contract:

```text
AnalyticalModelProvider
  providerId
  invokeStructured(request)
```

The request carries only:

```text
operation identity
system prompt
input
JSON Schema name
JSON Schema
maximum output tokens
non-secret metadata
```

The response carries:

```text
provider identity
deployment identity
provider request identity
provider-reported model when available
schema-validated output
raw JSON output text
provider token usage
rate-limit snapshot
traceable cost receipt
```

No VNext module contract is intrinsically coupled to Azure.

A future provider may implement the same interface without changing analytical methodology.

## 2. Azure adapter

Implementation:

```text
runtime/vnext/azure-provider.ts
```

The V0.1 Azure adapter uses the Azure OpenAI-compatible v1 Responses endpoint:

```text
POST /openai/v1/responses
```

Supported endpoint boundary:

```text
HTTPS only
*.openai.azure.com
*.services.ai.azure.com
```

The deployment name is configuration, not hard-coded methodology.

Authentication uses:

```text
api-key
```

The key is accepted only from server-side environment configuration.

No key is returned in receipts or successful output.



## 2.1 Model-selection neutrality

Passing Gate 13 with one Azure deployment does **not** select that physical model as the permanent OroTitan analytical model.

The broadened calibration universe is versioned separately at:

```text
calibration/vnext/OROTITAN_MODEL_ROUTING_ARCHITECTURE_V0.2.md
```

That v0.2 candidate architecture defines the expanded multi-provider competition, functional tiers, routing inputs, budget-governor semantics, error taxonomy and benchmark protocol. Physical model selection is explicitly deferred to later OroTitan calibration.

For Gate 13:

```text
LIVE AZURE MODEL
= ADAPTER / CAPABILITY PROOF
!= ROUTING POLICY FREEZE
!= BEST-MODEL DECLARATION
```

GPT-5.6 Sol High may serve as a reference-quality baseline for difficult reasoning modules, but Gate 13 does not declare it ground truth or the permanent winner.

## 3. Strict structured output

Every Azure analytical call in this adapter uses:

```text
text.format.type   = json_schema
text.format.strict = true
```

The supplied JSON Schema is sent to Azure.

Provider compliance alone is not accepted as final assurance.

After the response returns, VNext independently:

```text
extracts output text
parses JSON
validates JSON again with AJV 2020
fails closed on any mismatch
```

Therefore:

```text
PROVIDER SAYS STRUCTURED
+
LOCAL SCHEMA VALIDATION
= REQUIRED
```

## 4. Usage traceability

The adapter requires provider-reported usage.

Minimum fields:

```text
input_tokens
cached_input_tokens when supplied
output_tokens
total_tokens
```

If required usage is absent or internally inconsistent, the call fails closed.

The provider request ID is retained for later operational reconciliation.

## 5. Quota / rate-limit traceability

The adapter captures Azure response headers when present:

```text
x-ratelimit-limit-requests
x-ratelimit-limit-tokens
x-ratelimit-remaining-requests
x-ratelimit-remaining-tokens
x-ratelimit-reset-requests
x-ratelimit-reset-tokens
retry-after-ms
```

These are operational diagnostics only.

They do not become analytical evidence or score inputs.

## 6. Cost traceability

Azure model responses expose token usage, not the customer's final invoice amount.

Gate 13 therefore separates:

```text
MEASURED PROVIDER USAGE
from
PINNED RETAIL COST ESTIMATE
from
FINAL AZURE BILLING / CREDIT SETTLEMENT
```

The cost receipt records:

```text
usage source
pricing source
pricing effective date
currency
uncached input tokens
cached input tokens
output tokens
input USD / 1M
cached input USD / 1M
output USD / 1M
component costs
total estimated USD cost
```

Pricing is never silently hard-coded.

The exact rate snapshot used for a live call must be provided explicitly in environment configuration and preserved in the smoke-test evidence.

The V0.1 receipt is therefore named:

```text
PINNED_RETAIL_ESTIMATE
```

It must not be represented as an Azure invoice.

## 7. Environment variables

Required only when Azure is selected for a live analytical call:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_DEPLOYMENT
AZURE_OPENAI_API_KEY

AZURE_OPENAI_PRICING_SOURCE
AZURE_OPENAI_PRICING_EFFECTIVE_DATE
AZURE_OPENAI_INPUT_USD_PER_1M
AZURE_OPENAI_CACHED_INPUT_USD_PER_1M
AZURE_OPENAI_OUTPUT_USD_PER_1M
```

No real API key is committed to Git.

The adapter is constructed only when Azure is explicitly invoked.

Absence of Azure credentials does not alter the VNext state machine and does not grant any fallback provider implicit authority.

## 8. Authority isolation

The provider implementation has no import dependency on:

```text
Supabase
Registry writer
Run Controller
State machine
Post-Stage Certification
Publication
```

It therefore cannot directly mutate those systems.

A caller must separately pass provider output through the ordinary module, artifact, certification, and state-transition boundaries.

Model output is data, not state authority.

## 9. Failure behavior

Examples of fail-closed provider errors:

```text
invalid endpoint
missing deployment
missing API key
invalid request
HTTP/provider failure
timeout
missing provider request ID
missing usage
invalid token accounting
missing output text
non-JSON output
JSON Schema mismatch
invalid price schedule
missing cost provenance
```

Provider failure does not create a run-state transition by itself.

Recovery classification remains under Gate 11.

## 10. Live smoke command

Implementation:

```text
runtime/vnext/azure-gate13-smoke.ts
npm run test:vnext:azure-live
```

The smoke request is synthetic.

It asks the deployment to return exactly:

```json
{
  "gate": 13,
  "status": "PASS",
  "provider_role": "ANALYTICAL_MODEL_ONLY",
  "publication_authority": false
}
```

The command prints only secret-safe evidence:

```text
provider
deployment
provider request ID
provider-reported model
strict structured output
token usage
rate-limit snapshot
pinned cost receipt
```

It never prints the API key.

## 11. Deterministic tests

Implementation:

```text
tests/vnext-analytical-model-provider.test.ts
```

Covered cases:

```text
strict JSON Schema request                         PASS
independent local schema validation                PASS
non-JSON provider output                           REJECT
provider usage capture                             PASS
quota/rate-limit capture                           PASS
cached/uncached/output cost decomposition          PASS
price provenance required                          PASS
non-Azure endpoint                                 REJECT
HTTP failure                                       REJECT
secret absent from surfaced error                  PASS
missing usage                                      REJECT
replaceable non-Azure implementation               PASS
no state/registry/publication dependency           PASS
```

## 12. Gate 13 acceptance matrix

```text
G13-01 provider-neutral interface                      PASS
G13-02 Azure is adapter, not methodology dependency    PASS
G13-03 Azure v1 Responses request                      PASS
G13-04 strict JSON Schema requested                    PASS
G13-05 local independent schema validation             PASS
G13-06 provider usage required                         PASS
G13-07 token accounting checked                        PASS
G13-08 Azure rate-limit headers captured               PASS
G13-09 pricing provenance explicit                     PASS
G13-10 cost receipt deterministic                      PASS
G13-11 API key secret-safe                             PASS
G13-12 endpoint allow-list                             PASS
G13-13 no Registry/state/publication authority         PASS
G13-14 provider replaceability tested                  PASS
G13-15 deterministic CI                               PENDING
G13-16 real Azure model deployed                       PENDING
G13-17 real Azure API call successful                  PENDING
G13-18 real call strict JSON locally validates         PENDING
G13-19 real token usage captured                       PENDING
G13-20 real quota headers captured where supplied      PENDING
G13-21 live cost receipt traceable                     PENDING
```

## 13. Gate condition

Gate 13 remains pending until one real Azure deployment call is executed with a model actually available to the user's Azure subscription. The model used for this smoke test is a capability probe only; permanent model routing remains outside Gate 13.

The live evidence must show:

```text
HTTP SUCCESS
+
STRICT JSON OUTPUT
+
LOCAL SCHEMA PASS
+
PROVIDER REQUEST ID
+
TOKEN USAGE
+
RATE-LIMIT / QUOTA HEADERS WHEN PROVIDED
+
PINNED PRICE SOURCE
+
TRACEABLE COST RECEIPT
```

If the Azure Student subscription cannot expose a usable deployment, this is recorded as provider unavailability rather than an analytical-methodology failure.

Another provider may then implement `AnalyticalModelProvider` without changing the Gate 7 architecture.

## 13.1 Azure quota-request status

As of 2026-09-22:

```text
SUBSCRIPTION CLASS          = Azure for Students
TARGET MODEL                = gpt-5.6-sol
TARGET DEPLOYMENT TYPE      = Global Standard
TARGET REGION               = Switzerland North
CURRENT QUOTA BEFORE REQUEST= 0 kTPM
REQUESTED TOTAL QUOTA       = 10 kTPM
REQUEST STATUS              = SUBMITTED / PENDING MICROSOFT ALLOCATION
```

Microsoft's quota-request confirmation states that requests are typically processed the next business day and may take up to two business days, with no guarantee of approval.

This is an external-capacity dependency only. It does not change the Gate 13 provider contract, model-selection neutrality, or deterministic CI state.

If the request is denied or remains unavailable, Gate 13 may use another Azure model/deployment that satisfies the same capability probe. Such a substitution must be explicit and does not select that model as a permanent OroTitan routing winner.

## 14. Out of scope

Gate 13 does not:

- change analytical methodology;
- define P0 analytical modules;
- change scoring;
- change valuation;
- authorize publication;
- write production;
- write shadow Registry state;
- make ChatGPT Work a runtime API;
- resolve Gate 14 human escalation.
