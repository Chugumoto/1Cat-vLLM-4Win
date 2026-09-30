# 1Cat-vLLM for Windows (Tesla V100 / SM70)

Native **Windows** build + performance gates for [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) on **1× NVIDIA Tesla V100 (SM70)**.

This repository (`Chugumoto/1Cat-vLLM-4Win`) is a **Windows port / packaging fork** of upstream 1Cat SM70 work. It is **not** a claim of Linux 4× V100 · TP4 · ~260 tok/s parity.

## Based on / how it was made

| Layer | Source | What we took |
|---|---|---|
| SM70 engine, FlashAttention-V100, NVFP4/DFlash2 research path | [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) | Core vLLM fork optimized for Volta |
| Windows MSVC / CUDA build & runtime patterns | [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows) (`vllm-for-windows`) | `winloop`, FileStore, ZMQ TCP, MSVC flags, Windows requirements — **without** dropping SM70 or moving to CUDA 13 |
| This repo | Local Windows MVP + gates | Native build scripts, smoke/perf gates, Windows wheels under GitHub Releases |

**Build stack (pinned):** Windows 10/11 x64 · Python **3.12** · CUDA Toolkit **12.8** (CUDA 13 drops Volta) · Torch **2.10+cu128** · `TORCH_CUDA_ARCH_LIST=7.0` · VS 2022 MSVC (`vcvars`).

**How to build:** `scripts\windows\run_build_mvp.cmd` (after `env_build.cmd` / preflight). Details below.

**Runtime wheels (example release):** [v1.5.0.1w](https://github.com/Chugumoto/1Cat-vLLM-4Win/releases/tag/v1.5.0.1w) — `vllm-*-win_amd64.whl`, `flash_attn_v100-*-win_amd64.whl` (scheme: `1.5.0w` = clean 1Cat 1.5.0 Windows port; `1.5.0.1w` = patched).

**Limits (this fork’s gates):** **1× GPU only** — no TP / NCCL in Windows MVP gates. Upstream Linux headlines that use 4× TP4 are **context only**.

---

## Measured performance (1× Tesla V100-32GB)

Metric: `e2e_output_tok_s` = completion_tokens / wall seconds after warmup (`ignore_eos`, default `max_tokens=256`). Soft floors = `floor(0.7 × first green)`.

Full runbooks + JSON artifacts: [docs/windows/PERF_GATE.md](docs/windows/PERF_GATE.md) · `docs/windows/perf_results/`.

### Speed summary

| Gate | Profile | tok/s | Soft floor | Result |
|---|---|---:|---:|---|
| 1 | `QuantTrio/Qwen3.5-9B-AWQ` | **62.1** | 43 | PASS |
| 2 | `QuantTrio/Qwen3.6-27B-AWQ` | **29.3** | 20 | PASS |
| 2 | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` (target-only) | **30.9** | 21 | PASS |
| 2 | same + DFlash2 draft | — | — | **FAIL** (readiness timeout) |
| 3 | `QuantTrio/Qwen3.6-35B-A3B-AWQ` baseline | **63.2** | 44 | PASS |
| 3 | 35B-A3B-AWQ MTP-1 / MTP-2 | 56.4 / **64.5** | 39 / 45 | PASS |
| 3 | `nvidia/Qwen3.6-35B-A3B-NVFP4` baseline | **97.5** | 68 | PASS (best 1× decode here) |
| 3 | 35B-A3B-NVFP4 MTP-1 / MTP-4 | 60.6 / **69.2** | 42 / 48 | PASS (MTP &lt; no-MTP on this host) |
| 4 | `drawais/...-Coder-32B-…-NVFP4` | **18.0** | 12 | PASS (after QPN4 `split_k`-by-K) |
| 4 | `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` | **28.4** | 19 | PASS (faster decode) |

**Quality canary (Gate 2):** after teardown/`taskkill /T` + UTF-8 HTTP bodies — **SMOKE QUALITY OK** for 9B-AWQ + 27B-AWQ + 27B-NVFP4 (`quality_matrix_20260928-022124.json`).

**Notes:**

- Hybrid MoE 35B-A3B needs `--max-num-batched-tokens` ≥ Mamba block (~2096); scripts use 8192/4096.
- Linux table rows like ~174 tok/s (35B NVFP4+MTP4) or ~260 tok/s (27B+DFlash2) are **4× TP4**; do not treat them as the Windows 1× pass bar.
- Prefer `HF_HUB_DISABLE_XET=1`; large ModelOpt repos may need curl prefetch (see `scripts/windows/_prefetch_*.cmd`).

### How tests are run

```bat
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set HF_HUB_DISABLE_XET=1
scripts\windows\smoke_perf_qwen35_awq.cmd
scripts\windows\smoke_perf_qwen36_27b_awq.cmd
scripts\windows\smoke_perf_qwen38_nvfp4.cmd
scripts\windows\smoke_perf_qwen36_35b_a3b_awq.cmd
scripts\windows\smoke_perf_qwen36_35b_a3b_nvfp4.cmd
scripts\windows\smoke_perf_qwen25_coder_32b_awq.cmd
scripts\windows\smoke_quality_matrix.cmd
```

---

## Download & serve models (1× V100)

Use this when you want a **chat / Hermes / Codex** server (not the smoke-perf harness). Full gate details: [docs/windows/PERF_GATE.md](docs/windows/PERF_GATE.md).

### 1. Install runtime (wheels or local build)

Either build (`scripts\windows\run_build_mvp.cmd`) or install release wheels from [v1.5.0.1w](https://github.com/Chugumoto/1Cat-vLLM-4Win/releases/tag/v1.5.0.1w) into Python 3.12 + Torch 2.10+cu128.

### 2. Prefetch weights

Prefer `HF_HUB_DISABLE_XET=1`. Large ModelOpt NVFP4 repos often hang on Hub/Xet — use the **curl** prefetch scripts (resume-friendly).

| Model | Prefetch | Local path (after curl) |
|---|---|---|
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | `scripts\windows\_prefetch_35b_nvfp4_curl.cmd` | `%USERPROFILE%\.cache\huggingface\hub\models--nvidia--Qwen3.6-35B-A3B-NVFP4\manual` |
| `QuantTrio/Qwen3.6-35B-A3B-AWQ` | Hub `snapshot_download` / smoke script auto-pull | HF cache under `models--QuantTrio--...` |
| `QuantTrio/Qwen3.6-27B-AWQ` | same | HF cache |
| `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | Hub / smoke | HF cache |
| `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` | `scripts\windows\_prefetch_qwen25_coder_32b_awq_curl.cmd` | `...\models--Qwen--Qwen2.5-Coder-32B-Instruct-AWQ\manual` |
| `drawais/Qwen2.5-Coder-32B-Instruct-NVFP4` | `scripts\windows\_prefetch_qwen25_coder_32b_nvfp4_curl.cmd` | `...\models--drawais--...\manual` |

For 35B NVFP4: if `config.json` / tokenizer files are missing after the curl shard download, copy them from a Hub snapshot under the same `models--nvidia--Qwen3.6-35B-A3B-NVFP4\` tree (the prefetch script does this when a snapshot already exists).

### 3. Serve (interactive)

Best measured decode here (~97 tok/s baseline):

```bat
scripts\windows\start-Qwen3.6-35B-A3B-NVFP4.cmd
```

What that script does:

- `cd %TEMP%` before `python -m vllm...` (avoids Windows locking `vllm\_C.pyd` when cwd is the repo)
- `VLLM_SM70_GDN_DECODE_FLASHQLA=0` and `--gdn-prefill-backend triton` (avoids tilelang/TVM WinError 127 on GDN)
- `--served-model-name qwen36-35b-nvfp4` — **clients must use this id**, not the HF path
- `--host 127.0.0.1 --port 8005`, `--max-model-len 262144`, tools: `--enable-auto-tool-choice --tool-call-parser qwen3_xml`

Quick API check (PowerShell):

```powershell
curl.exe http://127.0.0.1:8005/v1/models
curl.exe http://127.0.0.1:8005/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen36-35b-nvfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":64}"
```

Hermes / Codex: set model to `qwen36-35b-nvfp4` and base URL `http://127.0.0.1:8005/v1`.

### 4. Perf / quality gates (automated)

Same models via `scripts\windows\smoke_perf_*.cmd` — see table above and PERF_GATE.md. Those scripts manage start/teardown and write JSON under `docs/windows/perf_results/`.

---

## Hard pins

| Component | Required | Notes |
|---|---|---|
| OS | Windows 10/11 x64 | Native (not WSL) for this MVP |
| Python | **3.12** | Tested with `C:\Python312` |
| CUDA Toolkit | **12.8** | **CUDA 13 unsupported** — Volta/SM70 removed |
| PyTorch | **2.10 + cu128** | Match CUDA 12.8 |
| GPU arch | `TORCH_CUDA_ARCH_LIST=7.0` | V100 only for this MVP |
| MSVC | VS 2022 + CUDA host compiler | Prefer a recent `vcvars` toolset |

## Limits (MVP)

- Single GPU only — **no TP / NCCL** in this MVP
- Serve smoke uses a small model (`facebook/opt-125m` by default)
- `torch.distributed` gloo can fail on some Hyper-V/VPN host setups; world_size=1 falls back to a fake process group

## Prerequisites

1. Install **CUDA 12.8** toolkit and drivers that expose the V100.
2. Install **Visual Studio 2022** with C++ workload; open a VS developer / `vcvars` shell when building.
3. Install Python 3.12 and [uv](https://github.com/astral-sh/uv) (optional but recommended for installs).
4. Ensure **Git** is on `PATH` (including `usr\bin` for `patch` during CMake FetchContent).

## Build

From a Developer / `vcvars` Command Prompt:

```bat
cd /d <repo-root>
scripts\windows\env_build.cmd
scripts\windows\run_build_mvp.cmd
```

Typical sequence (wrapped by `run_build_mvp.cmd`):

1. `env_build.cmd` — CUDA/arch/device env  
2. `python scripts\windows\preflight_env.py` — fail-closed check  
3. Install deps: `requirements\windows.txt` (+ Torch 2.10+cu128 if missing)  
4. Build FlashAttention-V100, then install vLLM  

## Smoke tests (MVP)

```bat
scripts\windows\env_build.cmd
python scripts\windows\preflight_env.py
python scripts\windows\smoke_imports.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\smoke_serve.ps1
```

Expected: `PREFLIGHT OK` · `SMOKE IMPORTS OK` · `SMOKE SERVE OK`.

## Layout helpers

| Path | Role |
|---|---|
| `scripts/windows/env_build.cmd` | Common env for build/smoke |
| `scripts/windows/preflight_env.py` | Contract checks |
| `scripts/windows/run_build_mvp.cmd` | Full MVP build driver |
| `scripts/windows/smoke_*.{cmd,ps1}` | Serve / perf / quality gates |
| `scripts/windows/_prefetch_*_curl.cmd` | Curl weight download (Hub/Xet workaround) |
| `scripts/windows/start-Qwen3.6-35B-A3B-NVFP4.cmd` | Interactive serve for 35B NVFP4 (port 8005) |
| `docs/windows/PERF_GATE.md` | Full gate runbook + floors |
| `docs/windows/perf_results/` | JSON/log artifacts |
| `docs/windows/INVENTORY.md` | Port checklist vs vllm-windows |

## Attribution

- Upstream SM70 / FlashAttention-V100: [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)  
- Windows build patterns: [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows)  
- Windows packaging / gates / releases: this fork (`1Cat-vLLM-4Win`)
