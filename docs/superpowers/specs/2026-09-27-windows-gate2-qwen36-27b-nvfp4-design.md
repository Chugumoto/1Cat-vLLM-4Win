# Windows Gate 2 — 27B-AWQ, NVFP4 (± DFlash2), speed summary

**Date:** 2026-09-27  
**Status:** Implemented; Phase 1 green at 29.328 tok/s (N27=20), Phase 2a NVFP4 target-only green at 30.949 tok/s (Nnv=21), Phase 2b DFlash2 failed readiness timeout (Ndflash unset), quality matrix **PASS** after teardown/UTF-8 fixes (`quality_matrix_20260928-022124.json`)  
**Parent:** [2026-09-27-windows-perf-gate-qwen35-awq-design.md](2026-09-27-windows-perf-gate-qwen35-awq-design.md) (Gate 1 green; N=43)  
**Track:** B continued — capability / perf beyond Gate 1 on Windows (still not TP/NCCL track A)

## Summary

After Gate 1 (`QuantTrio/Qwen3.5-9B-AWQ` on 1× Tesla V100-32GB), extend the Windows perf path in order:

1. Bring up **`QuantTrio/Qwen3.6-27B-AWQ`** (smoke-perf + soft floor **N27**).
2. Try **`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`** **target-only** on the same **1×32GB** V100.
3. If (2) is green, attach **`incoai/Qwen3.8-27B-DFlash2`** and measure again.
4. Publish a **speed summary** for all models that actually ran (9B-AWQ, 27B-AWQ, NVFP4 ± DFlash2), plus a **minimal quality canary** (2–3 greedy prompts).

Habr’s 2×16GB · TP2 · Qwen3.6-27B-AWQ article remains **context only** (~45 tok/s there). Linux 1Cat **4× V100 · TP4 · ~260 tok/s** NVFP4+DFlash2 headlines are **not** the pass bar for this Windows 1× gate.

## Goals

| Priority | Goal |
|---|---|
| P0 | Scripted smoke-perf for `QuantTrio/Qwen3.6-27B-AWQ` on 1× V100; JSON + soft **N27** |
| P0 | Attempt `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` target-only on 1×32GB; record OK or structured fail |
| P1 | If NVFP4 target green: enable DFlash2 draft `incoai/Qwen3.8-27B-DFlash2`; measure tok/s |
| P1 | Speed summary table in `docs/windows/PERF_GATE.md` for all successful (and failed) rows |
| P2 | Minimal quality matrix: 2–3 fixed greedy prompts + degeneration canary on models that served |

## Non-goals

- Tensor / pipeline parallelism and NCCL-on-Windows (track A)
- Matching Linux 4× TP4 ~260 tok/s or Habr 2×16GB numbers
- Full 1Cat README bench suites (MBPP/HumanEval matrices, 128K/256K exact decode)
- Commit / publish / GitHub automation unless the user explicitly asks

## Constraints

- **Hardware:** 1× `Tesla V100-SXM2-32GB` via `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0` (do not TP across CMP/RTX)
- **Stack:** Python 3.12, CUDA **12.8**, Torch **2.10+cu128**, this Windows fork
- **HF:** Prefer `HF_HUB_DISABLE_XET=1` on Windows (Gate 1 lesson)
- **Runtime:** Do not require `env_build.cmd` / `vcvars` for smoke (MSVC is build-only)
- **Ports:** 8001 reserved Gate 1; **8002** for 27B-AWQ; **8003** for NVFP4 (± DFlash2)

## Approach (approved)

**A — Separate smoke scripts per phase**, then a thin quality/summary step:

| Phase | Script (planned) | Model(s) |
|---|---|---|
| 1 | `scripts/windows/smoke_perf_qwen36_27b_awq.{ps1,cmd}` | `QuantTrio/Qwen3.6-27B-AWQ` |
| 2a | `scripts/windows/smoke_perf_qwen38_nvfp4.{ps1,cmd}` | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` (no draft) |
| 2b | same script + env (e.g. `VLLM_PERF_ENABLE_DFLASH2=1`) | + `incoai/Qwen3.8-27B-DFlash2` |
| 3 | `scripts/windows/smoke_quality_matrix.{ps1,cmd}` and/or doc update after runs | Models that reached `SMOKE PERF OK` |

Mirror Gate 1 patterns: `%TEMP%` cwd, System32 PowerShell wrapper, free-VRAM preflight (≥12 GiB), warmup + timed decode with `ignore_eos` on the timed request, JSON under `docs/windows/perf_results/`, soft floor after first green.

## Serve contracts (draft — tune on first OOM / backend error)

### Phase 1 — Qwen3.6-27B-AWQ

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
HF_HUB_DISABLE_XET=1

vllm serve QuantTrio/Qwen3.6-27B-AWQ
  --host 127.0.0.1 --port 8002
  --dtype float16
  --kv-cache-dtype fp8_e5m2
  --max-model-len 4096
  --gpu-memory-utilization 0.90
  --max-num-seqs 1
  --trust-remote-code
  --limit-mm-per-prompt {"image":0,"video":0}
```

On OOM: retry `--max-model-len 2048` (then document). On `fp8_e5m2` failure: `kv-cache-dtype auto`. Soft floor **N27 = max(1, floor(0.7 × first green e2e_output_tok_s))**.

### Phase 2a — Qwen3.8-27B-NVFP4 (target only)

```text
vllm serve QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4
  --host 127.0.0.1 --port 8003
  --max-model-len 2048   # start conservative on 1×32GB
  --gpu-memory-utilization 0.90
  --max-num-seqs 1
  --trust-remote-code
  # quantization / dtype flags: follow what this fork expects for QUASAR NVFP4 on SM70;
  # record exact argv in JSON; adjust only with PERF_GATE notes
```

**Expected risk:** Linux evidence is mostly **TP4 · 4×16GB**. On **1×32GB** the run may OOM, fail to select kernels, or need flags not yet wired on Windows. Structured fail (exit ≠ 0 + reason in log/JSON) is an acceptable Gate 2 outcome if documented in the summary table.

Soft floor **Nnv** only after a green target-only run.

### Phase 2b — + DFlash2 (only if 2a green)

Attach draft via vLLM speculative config (exact JSON shape as supported by this fork), e.g. draft model `incoai/Qwen3.8-27B-DFlash2`, method `dflash` (or fork-equivalent). Re-measure `e2e_output_tok_s`; set **Ndflash** the same 0.7× rule if green. If speculative path errors, keep 2a row and mark DFlash2 as failed with reason — do not block the summary of AWQ models.

## Benchmark / quality contract

| Item | Value |
|---|---|
| Metric | `e2e_output_tok_s` = completion_tokens / wall seconds after warmup; timed request uses `ignore_eos` when the API accepts it |
| Warmup | 1× short completion |
| Timed | `max_tokens=256` (or lower if 27B/NVFP4 needs it; record override) |
| Quality | 2–3 fixed prompts (EN short, EN factual; optional RU), `temperature=0`; canary fail on empty / obvious degeneration (extreme repetition) |
| Summary | Table in `PERF_GATE.md`: model, flags, tok/s, N, quality pass/fail, notes |

## Deliverables

| Path | Role |
|---|---|
| `scripts/windows/smoke_perf_qwen36_27b_awq.{ps1,cmd}` | Phase 1 |
| `scripts/windows/smoke_perf_qwen38_nvfp4.{ps1,cmd}` | Phase 2a/2b |
| `scripts/windows/smoke_quality_matrix.{ps1,cmd}` | Phase 3 canaries (or folded into docs if runs stay manual-first) |
| `docs/windows/PERF_GATE.md` | Gate 2 runbooks + speed summary + floors |
| `docs/windows/perf_results/perf_qwen36_27b_awq_*.json` | Phase 1 artifacts |
| `docs/windows/perf_results/perf_qwen38_nvfp4_*.json` | Phase 2 artifacts |
| `tests/windows/test_smoke_perf_gate2_contract.py` | Existence/string contracts |
| This spec | Status, floors, and sequencing |

## Success criteria

1. Phase 1 exits 0 with `SMOKE PERF OK` and **N27** written into docs (or blocked with a clear Windows/runtime reason after reasonable OOM retries).
2. Phase 2a either greens with **Nnv** or produces a documented fail row (not a silent skip).
3. Phase 2b runs only after 2a green; success or documented fail.
4. `PERF_GATE.md` contains a speed summary covering Gate 1 9B + whatever of 27B-AWQ / NVFP4 / DFlash2 completed.
5. Minimal quality canaries recorded for models that served.

Gate 2 Task 7 outcome (2026-09-28): Phase 2a passed at 30.949 tok/s on
1× V100-32GB, establishing **Nnv=21**. Phase 2b was attempted only after that
green result, but failed because the DFlash2-enabled server did not become
ready within 90×10 seconds; no **Ndflash** was set. `FLASH_ATTN_V100` was
accepted in Phase 2a, so no backend override was required.

Gate 2 Task 8 outcome (2026-09-28): the quality matrix failed. The 9B-AWQ
model passed two prompts before the Russian prompt returned HTTP 400; the
27B-AWQ and target-only NVFP4 quality servers each exceeded the 900-second
readiness window. DFlash2 was intentionally excluded from quality because its
performance server had already failed readiness. Results are recorded in
`docs/windows/perf_results/quality_matrix_20260928-020510.json`.

## Risks

| Risk | Mitigation |
|---|---|
| 27B-AWQ OOM at 4096 | Drop to 2048; util tweak; record |
| NVFP4 path missing / broken on Windows SM70 | Capture traceback; fail row; no TP workaround in this track |
| DFlash2 unsupported or needs newer vLLM bits | Skip 2b; keep target-only row |
| Huge HF downloads | `HF_HUB_DISABLE_XET=1`; long-running logs under `perf_results/` |
| Stale VRAM | Preflight free ≥12 GiB; kill leftover python/vllm |
| Mixing GPUs | Never set `CUDA_VISIBLE_DEVICES` to non-V100 for this gate |

## Spec self-review

- No TBD left in executable sequencing; numeric floors intentionally filled only after green runs.
- Non-goals do not contradict the approved inclusion of NVFP4/DFlash2 **as attempt phases**.
- Habr remains example-only; Linux TP4 numbers are reference, not pass criteria.
- Placeholder scan: draft serve flags for NVFP4 may need one-line argv fixes on first bring-up — allowed if logged in JSON/`PERF_GATE.md`.

## References

- Gate 1 design: `docs/superpowers/specs/2026-09-27-windows-perf-gate-qwen35-awq-design.md`
- Gate 1 ops: `docs/windows/PERF_GATE.md`
- Models: [QuantTrio/Qwen3.6-27B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ), [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4), [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
- Upstream headlines: repo `README.md` (4× V100 TP4 context)
