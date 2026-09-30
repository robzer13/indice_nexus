# OroTitan VNExT — Open Issues

## OI-001 — Completed/stopped candidate calibration evidence
Status: RETAINED_IMMUTABLE

## OI-002 — Granite 4.1 first Constellation C4
Status: COMPLETE_ENGINEERING_PASS_HUMAN_QUALITY_CRITICAL_FAILURE

Expansion: STOPPED
Retry: NOT_AUTHORIZED

## OI-003 — Llama 3.2 3B pinned download
Status: COMPLETE_PASS

Exact digest:
`a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`

## OI-004 — Llama 3.2 context4096 load-only preflight
Status: BLOCKED_PRE_EXECUTION_RUNTIME_UNREACHABLE
Priority: P0

The Ollama loopback API on port 11434 was unreachable before exact-tag verification. No model load or inference was attempted.

The existing single-run context4096 authorization remains unconsumed.

Required action:
restore Ollama runtime reachability, verify the local API, then execute the same authorized load-only preflight.

## OI-005 — Model winner and routing
Status: OPEN_GUARDED
