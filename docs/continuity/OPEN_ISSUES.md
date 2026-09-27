# OroTitan VNExT — Open Issues

## OI-001 — v1.1 validation architecture

Status: COMPLETE

## OI-002 — Phi-4 C4 Constellation qualification

Status: COMPLETE

Disposition:

`ENGINEERING_PASS_HUMAN_QUALITY_CRITICAL_FAILURE`

Phi-4 C4 expansion is stopped and retained as calibration evidence.

## OI-003 — Qwen3.5 4B hardware qualification

Status: COMPLETE_FOR_TARGET_CONTEXT

Load-only PASS at:

- context 4096;
- context 8192;
- context 16384.

The target context 16384 is loadable. RAM pressure is high and CPU offload is material.

## OI-004 — First Qwen3.5 C4 discriminator

Status: READY_LOCAL_EXECUTION  
Priority: P0

Selected cell:

`Constellation Software / SERIAL_ACQUIRER`

Frozen parameters:

- context 16384;
- max output 1024;
- temperature 0;
- timeout 600 seconds;
- loopback transport;
- validation v1.1;
- one run;
- no automatic retry.

## OI-005 — Human-quality adjudication

Status: BLOCKED_BY_OI-004

After the first Qwen3.5 cell, review exact evidence grounding, atomicity, polarity, qualifications, conflict handling, weak links, unresolved points, judgment boundary, and priority selection.

## OI-006 — Model winner and routing

Status: OPEN_GUARDED

No winner is selected and routing remains unfrozen.
