# 1Cat-vLLM for Windows (Tesla V100 / SM70)

Native Windows build and smoke path for [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) on **NVIDIA Tesla V100 (SM70)**.

Windows build/runtime adaptations are informed by [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows) (`vllm-for-windows` branch). This fork keeps Volta/SM70 and FlashAttention-V100 as first-class targets; it is **not** a drop-in replacement for Ampere+ Windows users (use SystemPanic for those GPUs).

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
4. Ensure **Git** is on `PATH` (including `usr\bin` for `patch` during CMake FetchContent). Hermes Git under `%LOCALAPPDATA%\hermes\git` works if added to `PATH`.

## Build

From a Developer / `vcvars` Command Prompt:

```bat
cd C:\Users\<you>\Projects\1Cat-vLLM-4Win
scripts\windows\env_build.cmd
```

Typical sequence (already wrapped by `scripts\windows\run_build_mvp.cmd`):

1. `scripts\windows\env_build.cmd` — sets CUDA/arch/device env
2. `python scripts\windows\preflight_env.py` — fail-closed env check
3. Install deps: `requirements\windows.txt` (+ Torch 2.10+cu128 if missing)
4. Build FlashAttention-V100, then install vLLM (`pip install -e .` / project build script)
5. Do **not** start a second `run_build_mvp.cmd` while one is writing the build log — Windows will report that the log file is in use

One-shot:

```bat
scripts\windows\run_build_mvp.cmd
```

If the log path is locked, set an alternate:

```bat
set BUILD_MVP_LOG=%TEMP%\build_mvp_alt.log
scripts\windows\run_build_mvp.cmd
```

## Smoke tests

```bat
scripts\windows\env_build.cmd
python scripts\windows\preflight_env.py
python scripts\windows\smoke_imports.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\smoke_serve.ps1
```

Expected markers:

- `PREFLIGHT OK`
- `SMOKE IMPORTS OK`
- `SMOKE SERVE OK` (HTTP `POST /v1/completions` against `127.0.0.1:8000`)

Serve notes:

- Sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and defaults `CUDA_VISIBLE_DEVICES=0`
- Uses UTF-8 (`PYTHONUTF8=1`) to avoid cp125x banner encode errors
- Starts `vllm` with working directory `%TEMP%` so the source tree does not shadow the installed package (extensions live under `site-packages`)

Optional model override:

```bat
set VLLM_SMOKE_MODEL=facebook/opt-125m
```

## Layout helpers

| Path | Role |
|---|---|
| `scripts/windows/env_build.cmd` | Common env for build/smoke |
| `scripts/windows/preflight_env.py` | Contract checks (Python/CUDA/Torch/arch) |
| `scripts/windows/run_build_mvp.cmd` | Full MVP build driver |
| `scripts/windows/smoke_imports.py` | Import + version smoke |
| `scripts/windows/smoke_serve.ps1` | Single-GPU OpenAI serve smoke |
| `requirements/windows.txt` | `winloop`, `triton-windows`, `portalocker`, `llguidance`, `xgrammar` |
| `docs/windows/INVENTORY.md` | Port checklist vs vllm-windows |

## Performance gate (1× V100)

See [docs/windows/PERF_GATE.md](docs/windows/PERF_GATE.md) for Gate 1 Qwen3.5-9B-AWQ, Gate 2 27B-AWQ/NVFP4, and Gate 3 Qwen3.6-35B-A3B-AWQ (+ MTP) performance gates.

## Attribution

- Upstream SM70 / FlashAttention-V100 engineering: [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)
- Windows build patterns (MSVC flags, winloop, FileStore, ZMQ TCP, etc.): [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows)
