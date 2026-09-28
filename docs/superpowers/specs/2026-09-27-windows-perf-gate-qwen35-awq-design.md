# Windows perf gate 1 — Qwen3.5-9B-AWQ on 1× V100

**Date:** 2026-09-27  
**Status:** Gate 1 implemented; N=43 from first green run 2026-09-27T23:01:51+03:00 (`e2e_output_tok_s=62.059`)
**Parent:** [2026-09-26-1cat-vllm-windows-design.md](2026-09-26-1cat-vllm-windows-design.md) (MVP smoke green)  
**Track:** B — performance / capability beyond MVP on Windows (not TP/NCCL)

## Summary

After MVP (`opt-125m` serve smoke), establish a **repeatable Windows performance gate** on **1× Tesla V100-32GB** using **`QuantTrio/Qwen3.5-9B-AWQ`**: start OpenAI-compatible serve, warm up, measure steady **decode tok/s**, record a JSON result. Throughput floor **N** is set after the first successful run (soft floor), not guessed up front.

This is **not** parity with Linux 1Cat headline benches (mostly **4× V100 · TP4 · NVFP4 ± DFlash2**). It is also **not** a clone of the Habr 2×16GB · TP2 · Qwen3.6-27B-AWQ setup ([habr.com/ru/articles/1043956](https://habr.com/ru/articles/1043956/)); that article is context only (~45 tok/s decode on 2×16GB). Same *class* of stack (1Cat + AWQ + V100), different hardware and model.

## Goals

| Priority | Goal |
|---|---|
| P0 | Documented serve command for `QuantTrio/Qwen3.5-9B-AWQ` on Windows 1× V100 |
| P0 | Scripted smoke-perf: warmup + fixed decode workload + JSON artifact |
| P0 | Record first green run; set soft decode tok/s floor **N** from that run |
| P1 | Short `docs/windows/PERF_GATE.md` with how to re-run and interpret results |
| P2 (later, gate 2) | Quality + throughput matrix for models that fit in 1×32GB |

## Non-goals (gate 1)

- Tensor / pipeline parallelism and NCCL-on-Windows (track A)
- Matching Linux README numbers (260 tok/s DFlash2, 128K/256K decode, etc.)
- NVFP4 / DFlash2 27B routes on this gate
- Reproducing Habr 2×16GB · TP2 · 65k · Hermes agent setup
- Publishing wheels or GitHub release automation

## Constraints

- **Hardware:** 1× `Tesla V100-SXM2-32GB` via `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`
- **Stack:** Python 3.12, CUDA **12.8**, Torch **2.10+cu128**, this Windows fork (MSVC-built)
- **Model:** `QuantTrio/Qwen3.5-9B-AWQ` (HF; ~12GB download; AWQ 4-bit)
- **No CUDA 13** build/runtime baseline (Volta)
- Kill leftover GPU holders before runs (`nvidia-smi`); avoid parallel `pip`/`run_build_mvp` that corrupt `site-packages`

## Approach (approved)

**B — Windows smoke-perf script** mirroring `scripts/windows/smoke_serve.*`:

1. Ensure env (`env_build.cmd` / same vars as MVP smoke)
2. Start `vllm serve QuantTrio/Qwen3.5-9B-AWQ` with a pinned flag set
3. Wait until `/v1/completions` or `/v1/chat/completions` responds
4. Warmup request(s)
5. Timed decode request(s); compute output tok/s
6. Write JSON result; print `SMOKE PERF OK` or fail with clear reason
7. Tear down server

Prefer HTTP against a real serve process (same path users run) over in-process `LLM()` for gate 1.

## Serve contract (draft — tune on first OOM / backend error)

Exact flags may need small adjustments on first bring-up; changes must be logged in the JSON/`PERF_GATE.md`.

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
PYTHONUTF8=1

vllm serve QuantTrio/Qwen3.5-9B-AWQ
  --host 127.0.0.1 --port 8001
  --dtype float16
  --kv-cache-dtype fp8_e5m2
  --max-model-len 8192
  --gpu-memory-utilization 0.90
  --max-num-seqs 1
  --trust-remote-code
  --limit-mm-per-prompt {"image":0,"video":0}
```

Notes:

- Port **8001** by default so it does not collide with MVP smoke on 8000.
- Multimodal weights exist on this checkpoint; disable image/video for text-only gate.
- If `fp8_e5m2` or attention backend fails on Windows, fall back is documented in the result JSON (`kv_cache_dtype`, `attention_backend`, error string) and `PERF_GATE.md` updated — do not silently change the contract without recording it.
- Habr used `VLLM_DISABLE_PYNCCL=1` and `--disable-custom-all-reduce` for **TP2**; for TP=1 these are optional. Enable only if bring-up requires them.

## Benchmark contract (gate 1)

| Field | Value |
|---|---|
| Endpoint | `/v1/completions` (or chat if completions unsupported for this arch) |
| Warmup | 1 request, `max_tokens=32`, discarded from timing |
| Measured | ≥1 request, fixed prompt, `max_tokens=256`, greedy or temperature=0 |
| Prompt | Fixed short English string (pinned in script; e.g. 64–128 tokens after tokenize if easy, else fixed text) |
| Metric | `output_tokens / wall_seconds` after first output token preferred; if streaming is hard, use full-request wall time and label metric `e2e_output_tok_s` |
| Pass | Server stays up; metric recorded; after first green run, `metric >= N` where **N** is written into this spec and `PERF_GATE.md` |

First green run procedure:

1. Run script once to completion with no floor check (or floor=0)
2. Read measured tok/s
3. Set **N = max(1, floor(0.7 × measured))** (or user-chosen round number) and update docs
4. Subsequent CI/local gates use that N

## Deliverables

| Path | Role |
|---|---|
| `scripts/windows/smoke_perf_qwen35_awq.ps1` | Main gate script |
| `scripts/windows/smoke_perf_qwen35_awq.cmd` | CMD wrapper (full path to PowerShell; like `smoke_serve.cmd`) |
| `docs/windows/PERF_GATE.md` | How to run, contract, current **N**, known Windows caveats |
| Result JSON (e.g. `docs/windows/perf_results/` or `%TEMP%`) | Timestamp, model, flags, tok/s, GPU free/used snapshot |

Optional: extend `README.windows.md` with a one-paragraph pointer to `PERF_GATE.md`.

## Gate 2 (out of scope for this spec’s implementation plan, listed for sequencing)

- Quality + throughput matrix for models that fit **1×32GB** (start with same 9B-AWQ; optionally Qwen3.6-27B-AWQ on 1×32GB)
- Datasets / scores in the spirit of 1Cat README (subset, not full 4× TP4 matrix)
- Separate design + plan when gate 1 is green

## Success criteria

1. On a clean V100 (no stale process holding VRAM), script exits 0 and prints `SMOKE PERF OK`
2. JSON artifact exists with model id, flags, and tok/s
3. Spec and `PERF_GATE.md` contain concrete **N** after first green run
4. Re-run on the same machine without code changes still passes **N** (allow ~10% noise band if documented)

## Risks

| Risk | Mitigation |
|---|---|
| AWQ / Qwen3.5 path broken on Windows build | Capture full traceback; minimal kernel/flag fixes only; stay on CUDA 12.8 |
| OOM at max-model-len 8192 | Lower to 4096; record in JSON |
| HF download / disk | Document cache dir; allow `VLLM_SMOKE_MODEL` / local path override |
| Stale GPU memory | Preflight `nvidia-smi`; fail early if free &lt; threshold |
| Broken `site-packages` from concurrent pip | Document; use restore procedure from MVP notes |

## References

- Parent Windows MVP design: `docs/superpowers/specs/2026-09-26-1cat-vllm-windows-design.md`
- Model: [QuantTrio/Qwen3.5-9B-AWQ](https://huggingface.co/QuantTrio/Qwen3.5-9B-AWQ)
- Context (2×16GB TP2, not this gate): [Habr 1043956](https://habr.com/ru/articles/1043956/)
- Upstream 1Cat benches (4× TP4 headlines): repo `README.md`
