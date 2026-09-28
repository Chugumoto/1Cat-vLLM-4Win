# Windows Gate 3 — Qwen3.6-35B-A3B AWQ + NVFP4 (+ native MTP)

**Date:** 2026-09-28  
**Status:** Spec approved; implementation in progress (`docs/superpowers/plans/2026-09-28-windows-gate3-qwen36-35b-a3b.md`)  
**Parent:** [2026-09-27-windows-gate2-qwen36-27b-nvfp4-design.md](2026-09-27-windows-gate2-qwen36-27b-nvfp4-design.md)  
**Track:** B continued — Windows 1× V100 perf/capability (not TP/NCCL)

## Summary

Add a Gate 3 smoke path for **`QuantTrio/Qwen3.6-35B-A3B-AWQ`** and **`nvidia/Qwen3.6-35B-A3B-NVFP4`** (MoE: ~35B total / ~3B active) on **1× Tesla V100-32GB**:

1. AWQ baseline serve (no speculative) → soft floor **N35**
2. Native MTP with `num_speculative_tokens=1` → **Nmtp1** or documented FAIL
3. If MTP-1 green: MTP with `num_speculative_tokens=2` → **Nmtp2** or documented FAIL
4. NVFP4 ModelOpt baseline (`nvidia/...-NVFP4`, **not** GGUF) → soft floor **N35nv**; optional MTP after green
5. Quality canary + speed summary row(s) in `PERF_GATE.md`

Context for expectations: Reddit V100 ~40–50+ tok/s for dense Qwen3.6-27B usually includes **MTP / llama.cpp**; our Gate 2 dense 27B-AWQ **no-MTP** measured **~29 tok/s**, matching published no-MTP baselines. MoE 35B-A3B is expected to differ (often higher tok/s at similar or lower VRAM pressure than dense 27B).

## Goals

| Priority | Goal |
|---|---|
| P0 | Scripted smoke-perf for `QuantTrio/Qwen3.6-35B-A3B-AWQ` on 1× V100; JSON + **N35** |
| P0 | Scripted smoke-perf for `nvidia/Qwen3.6-35B-A3B-NVFP4` on 1× V100; JSON + **N35nv** |
| P0 | Attempt native MTP (`method=mtp`, `num_speculative_tokens=1`); record OK/**Nmtp1** or FAIL |
| P1 | If MTP-1 green: run `num_speculative_tokens=2`; record **Nmtp2** or FAIL |
| P1 | Update `PERF_GATE.md` speed summary + quality canary for these models |
| P2 | Contract tests for new scripts/docs markers |

## Non-goals

- TP / NCCL / multi-GPU
- Matching llama.cpp Reddit headlines or Linux 4× TP4 numbers
- DFlash2 / external draft for this model (native MTP only)
- Commit / publish unless the user explicitly asks

## Constraints

- Hardware: 1× `Tesla V100-SXM2-32GB`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`
- Stack: Python 3.12, CUDA **12.8**, Torch **2.10+cu128**, this Windows fork
- Port **8004** (8001/8002/8003 reserved by earlier gates)
- `HF_HUB_DISABLE_XET=1`; no `vcvars` required for runtime
- Teardown: `taskkill /F /T` on serve PID tree
- Soft floors: `max(1, floor(0.7 × first green e2e_output_tok_s))` per successful phase

## Approach (approved)

**A — Dedicated smoke script with MTP env toggles**, mirroring Gate 2 NVFP4/DFlash pattern:

| Phase | Env | Speculative config |
|---|---|---|
| Baseline | MTP unset / 0 | none |
| MTP-1 | `VLLM_PERF_ENABLE_MTP=1`, `VLLM_PERF_MTP_NUM_TOKENS=1` (default when enabled) | `{"method":"mtp","num_speculative_tokens":1}` |
| MTP-2 | `VLLM_PERF_ENABLE_MTP=1`, `VLLM_PERF_MTP_NUM_TOKENS=2` | `{"method":"mtp","num_speculative_tokens":2}` |

Script: `scripts/windows/smoke_perf_qwen36_35b_a3b_awq.{ps1,cmd}`  
JSON prefix: `perf_qwen36_35b_a3b_awq_*.json` with fields `mtp_enabled`, `mtp_num_tokens`, `speculative_config`.

## Serve contract (draft)

### Baseline

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
HF_HUB_DISABLE_XET=1

vllm serve QuantTrio/Qwen3.6-35B-A3B-AWQ
  --host 127.0.0.1 --port 8004
  --dtype float16
  --kv-cache-dtype fp8_e5m2
  --max-model-len 8192
  --gpu-memory-utilization 0.90
  --max-num-seqs 1
  --trust-remote-code
  --limit-mm-per-prompt {"image":0,"video":0}
```

On OOM: retry `--max-model-len 4096` (document). On `fp8_e5m2` failure: `kv-cache-dtype auto`.

### MTP

Same flags plus:

```text
--speculative-config {"method":"mtp","num_speculative_tokens":<1|2>}
```

If serve rejects MTP on this AWQ/Windows build: structured FAIL row; keep baseline numbers. Do not invent alternate draft models in this gate.

## Benchmark / quality

| Item | Value |
|---|---|
| Metric | `e2e_output_tok_s` after warmup; timed request `ignore_eos` when accepted |
| Timed | `max_tokens=256` default |
| Quality | Include model in `smoke_quality_matrix` (or one-shot canary with same 3 prompts) |
| Summary | Rows: baseline, MTP-1, MTP-2 (FAIL allowed) |

## Deliverables

| Path | Role |
|---|---|
| `scripts/windows/smoke_perf_qwen36_35b_a3b_awq.{ps1,cmd}` | Baseline + MTP toggles |
| `tests/windows/test_smoke_perf_gate3_contract.py` | Marker contracts |
| `docs/windows/PERF_GATE.md` | Gate 3 runbook + summary |
| `docs/windows/perf_results/perf_qwen36_35b_a3b_awq_*.json` | Artifacts |
| This spec | Status + floors after greens |

Optional: extend `smoke_quality_matrix.ps1` model profiles for port 8004 / len 8192.

## Success criteria

1. Baseline greens with **N35** written, or clear blocked reason after OOM retries.
2. MTP-1 attempted; OK+**Nmtp1** or documented FAIL.
3. MTP-2 attempted only after MTP-1 green; OK+**Nmtp2** or documented FAIL.
4. `PERF_GATE.md` summary updated; quality canary recorded for baseline (at least).
5. Contract tests PASS.

## Risks

| Risk | Mitigation |
|---|---|
| MoE AWQ path broken on Windows | Capture traceback; FAIL documented |
| MTP unsupported for AWQ/QuantTrio weights | FAIL MTP rows; keep baseline |
| OOM at 8192 | Drop to 4096 |
| Thinking/`<think>` inflates e2e wall | Same metric as Gate 1/2; note in PERF_GATE if needed |
| Large HF download | `HF_HUB_DISABLE_XET=1`; log under `perf_results/` |

## Spec self-review

- Sequencing matches approved approach A and MTP 1→2 rule.
- Non-goals exclude DFlash2/external draft and TP.
- Floors intentionally TBD until measurement.
- No contradiction with Gate 2 ports/scripts.

## References

- Model: [QuantTrio/Qwen3.6-35B-A3B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ)
- MTP recipe shape: `{"method":"mtp","num_speculative_tokens":N}` (vLLM Qwen3.5/3.6 guides)
- Prior gates: `docs/windows/PERF_GATE.md`, Gate 2 design
