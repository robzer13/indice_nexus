# OroTitan VNExT — Latest Handoff

## Granite 4.1 3B — context16384 load-only result

Status: `PASS_LOAD_ONLY_MEASURED`

Measured:
- before free RAM: 1.06 GiB;
- loaded free RAM: 0.84 GiB;
- after free RAM: 2.16 GiB;
- loaded VRAM used/free: 2305 / 1658 MiB;
- processor split: 38%/62% CPU/GPU;
- Ollama reported loaded size: 3.8 GB;
- load-only confirmed;
- explicit unload complete;
- post-unload VRAM used/free: 0 / 3962 MiB.

Interpretation:

`PASS_WITH_USEFUL_HEADROOM`

The context16384 load-only authorization is consumed.

Compared with Granite 4 3B at context16384, VRAM use and processor split are identical while loaded free RAM is 0.43 GiB higher. RAM differences remain host-state-sensitive.

Authoritative post-context disposition:

`calibration/vnext/OROTITAN_GATE18_GRANITE4_1_POST_CONTEXT16384_DISPOSITION_001.json`

## Exact next action

`REVIEW_GRANITE4_1_POST_CONTEXT16384_DISPOSITION`
