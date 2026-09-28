# Windows performance gate (1× V100)

## Gate 1 — Qwen3.5-9B-AWQ

- Model: `QuantTrio/Qwen3.5-9B-AWQ`
- Hardware: 1× Tesla V100 via `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`
- Script: `scripts/windows/smoke_perf_qwen35_awq.cmd`
- Metric: `e2e_output_tok_s` (decode tok/s — completion_tokens / wall seconds after warmup)
- Soft floor **N**: **43** tok/s (`VLLM_PERF_TOK_S_FLOOR=43`)
- First green (2026-09-27): `e2e_output_tok_s=62.059` → N = floor(0.7×62.059) = 43  
  Artifact: `docs/windows/perf_results/perf_qwen35_awq_20260927-230151.json`

### Run

Runtime (no MSVC/`vcvars` required):

```bat
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set HF_HUB_DISABLE_XET=1
set VLLM_PERF_TOK_S_FLOOR=43
scripts\windows\smoke_perf_qwen35_awq.cmd
```

Optional MTP probe (floor TBD until measured):

```bat
set VLLM_PERF_ENABLE_MTP=1
set VLLM_PERF_MTP_NUM_TOKENS=1
scripts\windows\smoke_perf_qwen35_awq.cmd
```

The timed decode request sets OpenAI-compatible `ignore_eos: true` so generation runs for the full `max_tokens` (256 by default), improving repeatability of `e2e_output_tok_s`.

Optional: `call scripts\windows\env_build.cmd` only from an **x64 Native Tools** / `vcvarsall` shell (it fails otherwise). Not needed for serve/smoke-perf.

Optional env overrides: `VLLM_PERF_MODEL`, `VLLM_PERF_PORT` (default 8001), `VLLM_PERF_MAX_MODEL_LEN`, `VLLM_PERF_KV_CACHE_DTYPE`, `VLLM_PERF_GPU_MEM_UTIL`, `VLLM_PERF_MAX_TOKENS`, `VLLM_PERF_TOK_S_FLOOR`.

Results JSON: `docs/windows/perf_results/perf_qwen35_awq_*.json`.

### Before you run

1. `nvidia-smi` — free V100 memory (script fails if free < 12 GiB).
2. Do not run from a cmd session whose PATH was wiped by `run_build_mvp` without PowerShell (wrapper uses System32 path).
3. First download of the HF model can take a long time (~12 GB). Prefer `HF_HUB_DISABLE_XET=1` on Windows if Hub download stalls on 0-byte `.incomplete` stubs.

### Out of scope

TP/NCCL, Linux 4× NVFP4/DFlash2 parity, Habr 2×16GB agent setup.

## Gate 2 — 27B-AWQ + NVFP4 (± DFlash2)

### Phase 1 — Qwen3.6-27B-AWQ

- Model: `QuantTrio/Qwen3.6-27B-AWQ`
- Script: `scripts/windows/smoke_perf_qwen36_27b_awq.cmd`
- Port: 8002
- Soft floor **N27**: **20** tok/s (`VLLM_PERF_TOK_S_FLOOR=20`)
- First green (2026-09-28): `e2e_output_tok_s=29.328` → N27 = floor(0.7×29.328) = 20  
  Artifact: `docs/windows/perf_results/perf_qwen36_27b_awq_20260928-001845.json`
- Start: `max-model-len=4096` (fallback 2048 on OOM)

```bat
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set HF_HUB_DISABLE_XET=1
set VLLM_PERF_TOK_S_FLOOR=20
scripts\windows\smoke_perf_qwen36_27b_awq.cmd
```

Optional MTP probe (floor TBD until measured):

```bat
set VLLM_PERF_ENABLE_MTP=1
set VLLM_PERF_MTP_NUM_TOKENS=1
scripts\windows\smoke_perf_qwen36_27b_awq.cmd
```

### Phase 2a — Qwen3.8-27B-NVFP4 (target only)

- Model: `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`
- Script: `scripts/windows/smoke_perf_qwen38_nvfp4.cmd`
- Port: 8003
- Soft floor **Nnv**: **21** tok/s (`VLLM_PERF_TOK_S_FLOOR=21`)
- First green (2026-09-28): `e2e_output_tok_s=30.949` → Nnv = floor(0.7×30.949) = 21
  Artifact: `docs/windows/perf_results/perf_qwen38_nvfp4_20260928-010021.json`
- `FLASH_ATTN_V100` was accepted; no attention-backend override was needed.
- Note: Linux headlines use 4× TP4; this gate tries **1×32GB** only.

```bat
set HF_HUB_DISABLE_XET=1
set VLLM_PERF_TOK_S_FLOOR=21
scripts\windows\smoke_perf_qwen38_nvfp4.cmd
```

### Phase 2b — + DFlash2 (only if 2a green)

```bat
set VLLM_PERF_ENABLE_DFLASH2=1
scripts\windows\smoke_perf_qwen38_nvfp4.cmd
```

Draft: `incoai/Qwen3.8-27B-DFlash2`. **FAIL**: the server did not become
ready within 90×10 seconds; no throughput JSON or **Ndflash** was produced.
Log: `docs/windows/perf_results/smoke_perf_qwen38_nvfp4_dflash2_20260928-010624.log`.

### Quality matrix

```bat
scripts\windows\smoke_quality_matrix.cmd
```

The 2026-09-28 matrix first run **FAIL**ed (RU prompt HTTP 400 under PS 5.1
default encoding; later models hit readiness timeouts from incomplete
teardown). After `taskkill /F /T` teardown + UTF-8 JSON bodies, rerun
**SMOKE QUALITY OK** (all three models, 3/3 prompts each).

Artifact: `docs/windows/perf_results/quality_matrix_20260928-022124.json`
(log: `quality_matrix_rerun_20260928-020843.log`).

### Speed summary

| Model | tok/s | Floor | Quality | Notes |
|---|---|---|---|---|
| Qwen3.5-9B-AWQ | 62.059 | N=43 | PASS | Gate 1 |
| Qwen3.6-27B-AWQ | 29.328 | N27=20 | PASS | 4096 max model length |
| Qwen3.8-27B-NVFP4 | 30.949 | Nnv=21 | PASS | 1×32GB; target-only green |
| Qwen3.8-27B-NVFP4+DFlash2 | FAIL | Ndflash=unset | Not run | server readiness timeout after 900 s; no JSON; quality canary not required |

## Gate 3 — Qwen3.6-35B-A3B AWQ + NVFP4 (+ MTP)

### Phase 3a — AWQ

- Model: `QuantTrio/Qwen3.6-35B-A3B-AWQ`
- Script: `scripts/windows/smoke_perf_qwen36_35b_a3b_awq.cmd`
- Port: 8004
- Soft floor **N35**: **44** (= `floor(0.7 × 63.234)`)
- Soft floor **Nmtp1**: **39** (= `floor(0.7 × 56.440)`)
- Soft floor **Nmtp2**: **45** (= `floor(0.7 × 64.526)`)
- Hybrid Mamba: scripts pass `--max-num-batched-tokens 8192` (block_size ~2096 must be ≤ batched tokens)

#### Baseline

```bat
set HF_HUB_DISABLE_XET=1
set VLLM_PERF_TOK_S_FLOOR=44
scripts\windows\smoke_perf_qwen36_35b_a3b_awq.cmd
```

First green (floor=0): **63.234** tok/s — `perf_qwen36_35b_a3b_awq_20260928-120950.json`
#### MTP-1

```bat
set VLLM_PERF_ENABLE_MTP=1
set VLLM_PERF_MTP_NUM_TOKENS=1
set VLLM_PERF_TOK_S_FLOOR=39
scripts\windows\smoke_perf_qwen36_35b_a3b_awq.cmd
```

First MTP-1 green (floor=0): **56.440** tok/s — `perf_qwen36_35b_a3b_awq_20260928-122142.json` (below no-MTP on this host; acceptance TBD)
#### MTP-2 (only if MTP-1 green)

```bat
set VLLM_PERF_ENABLE_MTP=1
set VLLM_PERF_MTP_NUM_TOKENS=2
set VLLM_PERF_TOK_S_FLOOR=45
scripts\windows\smoke_perf_qwen36_35b_a3b_awq.cmd
```

First MTP-2 green (floor=0): **64.526** tok/s — `perf_qwen36_35b_a3b_awq_20260928-122625.json`

### Phase 3b — NVFP4 (ModelOpt; not GGUF)

- Model: `nvidia/Qwen3.6-35B-A3B-NVFP4` (vLLM ModelOpt NVFP4; **not** `simhadrig/...-NVFP4-Q8_0` GGUF)
- Script: `scripts/windows/smoke_perf_qwen36_35b_a3b_nvfp4.cmd`
- Port: 8005
- Defaults: `max_model_len=4096`, `--max-num-batched-tokens 4096`, `--attention-backend FLASH_ATTN_V100`
- Soft floor **N35nv**: **68** (= `floor(0.7 × 97.454)`)
- Soft floor **Nmtp1nv**: **42** (= `floor(0.7 × 60.559)`)
- Soft floor **Nmtp4nv**: **48** (= `floor(0.7 × 69.242)`)
- Local weights (curl; HF Xet hung): `C:\Users\Chugumoto\.cache\huggingface\hub\models--nvidia--Qwen3.6-35B-A3B-NVFP4\manual`

#### Baseline

```bat
set HF_HUB_DISABLE_XET=1
set VLLM_PERF_MODEL=C:\Users\Chugumoto\.cache\huggingface\hub\models--nvidia--Qwen3.6-35B-A3B-NVFP4\manual
set VLLM_PERF_TOK_S_FLOOR=68
scripts\windows\smoke_perf_qwen36_35b_a3b_nvfp4.cmd
```

First green (floor=0): **97.454** tok/s — `perf_qwen36_35b_a3b_nvfp4_20260928-131946.json`
#### MTP-1

```bat
set VLLM_PERF_MODEL=C:\Users\Chugumoto\.cache\huggingface\hub\models--nvidia--Qwen3.6-35B-A3B-NVFP4\manual
set VLLM_PERF_ENABLE_MTP=1
set VLLM_PERF_MTP_NUM_TOKENS=1
set VLLM_PERF_TOK_S_FLOOR=42
scripts\windows\smoke_perf_qwen36_35b_a3b_nvfp4.cmd
```

First MTP-1 green (floor=0): **60.559** tok/s — `perf_qwen36_35b_a3b_nvfp4_20260928-132922.json` (below no-MTP; same pattern as AWQ MTP-1)

#### MTP-4 (table-comparable k; not 4× TP4 parity)

```bat
set VLLM_PERF_MODEL=C:\Users\Chugumoto\.cache\huggingface\hub\models--nvidia--Qwen3.6-35B-A3B-NVFP4\manual
set VLLM_PERF_ENABLE_MTP=1
set VLLM_PERF_MTP_NUM_TOKENS=4
set VLLM_PERF_TOK_S_FLOOR=48
scripts\windows\smoke_perf_qwen36_35b_a3b_nvfp4.cmd
```

First MTP-4 green (floor=0): **69.242** tok/s — `perf_qwen36_35b_a3b_nvfp4_20260928-134134.json` (above MTP-1, still below no-MTP 97; Linux table 174 is 4× TP4)

### Gate 3 summary

| Profile | tok/s | Floor | Quality | Notes |
|---|---:|---|---|---|
| Qwen3.6-35B-A3B-AWQ baseline | 63.234 | N35=44 | TBD | First green 63.234; floor recheck 73.140 OK |
| Qwen3.6-35B-A3B-AWQ MTP-1 | 56.440 | Nmtp1=39 | TBD | Slower than baseline on this 1× run; still green |
| Qwen3.6-35B-A3B-AWQ MTP-2 | 64.526 | Nmtp2=45 | TBD | Near baseline; better than MTP-1 |
| Qwen3.6-35B-A3B-NVFP4 baseline | 97.454 | N35nv=68 | TBD | nvidia ModelOpt; local curl path; 1×32GB |
| Qwen3.6-35B-A3B-NVFP4 MTP-1 | 60.559 | Nmtp1nv=42 | TBD | Slower than NVFP4 baseline (like AWQ MTP-1) |
| Qwen3.6-35B-A3B-NVFP4 MTP-4 | 69.242 | Nmtp4nv=48 | TBD | Better than MTP-1; still &lt; no-MTP; not 4×174 |

## Gate 4 — Qwen2.5-Coder-32B (coding canary; prefer NVFP4)

### Phase 4a — NVFP4 (community compressed-tensors)

- Model: `drawais/Qwen2.5-Coder-32B-Instruct-NVFP4` (~20.7GB; `nvfp4-pack-quantized`)
- **Not** an official `nvidia/` ModelOpt Coder-32B (none found). Fallback AWQ: `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`
- Script: `scripts/windows/smoke_perf_qwen25_coder_32b_nvfp4.cmd`
- Port: 8006
- Soft floor **Ncoder**: _TBD after baseline green_
- Prefetch: `scripts\windows\_prefetch_qwen25_coder_32b_nvfp4_curl.cmd` → local `...\manual`
- Risk: community checkpoint on SM70; if serve fails, fall back to official AWQ scripts

#### Baseline

```bat
set HF_HUB_DISABLE_XET=1
set VLLM_PERF_TOK_S_FLOOR=0
set VLLM_PERF_MODEL=C:\Users\Chugumoto\.cache\huggingface\hub\models--drawais--Qwen2.5-Coder-32B-Instruct-NVFP4\manual
scripts\windows\smoke_perf_qwen25_coder_32b_nvfp4.cmd
```

### Phase 4b — AWQ fallback (official Qwen)

- Model / script: `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` / `smoke_perf_qwen25_coder_32b_awq.cmd`
- Prefetch: `_prefetch_qwen25_coder_32b_awq_curl.cmd` (paused when switching to NVFP4)

### Gate 4 summary

| Profile | tok/s | Floor | Quality | Notes |
|---|---:|---|---|---|
| Qwen2.5-Coder-32B-Instruct-NVFP4 | TBD | Ncoder=TBD | TBD | drawais community; ~21GB; 1×32GB |
| Qwen2.5-Coder-32B-Instruct-AWQ | TBD | TBD | TBD | Official Qwen AWQ fallback |

