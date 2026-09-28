# 1Cat-vLLM for Windows — Design

**Date:** 2026-09-26  
**Workspace:** `C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win`  
**Status:** Implemented per plan 2026-09-26-1cat-vllm-windows.md (MVP smoke green)

## Summary

Create a Windows-capable fork of [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) by porting Windows build/runtime patches from [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows) (`vllm-for-windows` branch), while keeping Tesla V100 / SM70 as the primary target.

Deliverables: (1) a local tree that builds and smoke-tests on native Windows with one V100, then (2) a public GitHub repository.

## Goals

| Priority | Goal |
|---|---|
| P0 | Build from source on Windows (MSVC + CUDA 12.8 + PyTorch 2.10) |
| P0 | Smoke on **1× Tesla V100** (native Windows, not WSL): imports + short `vllm serve` |
| P1 | Publish GitHub fork with Windows install/build docs |
| P2 (later) | Multi-GPU (NCCL / TP), full feature parity with Linux 1Cat, wheel releases |

## Non-goals (MVP)

- Tensor / pipeline parallelism and NCCL-on-Windows
- Matching Linux 1Cat performance (DFlash2 / NVFP4 / long-context benches) in the first release
- Migrating to CUDA 13 / newer Torch from current vllm-windows releases (**impossible for V100:** CUDA 13 dropped Volta/SM70)
- Replacing or competing with SystemPanic/vllm-windows for non-V100 Windows users
- Automated CI on real V100 hardware in MVP

## Sources and version pins

| Source | Role |
|---|---|
| `1CatAI/1Cat-vLLM` (`main`) | **Base tree** — V100 / FlashAttention-V100 / SM70 engineering |
| `SystemPanic/vllm-windows` (`vllm-for-windows`) | **Patch reference only** — MSVC/CMake/setup, `requirements/windows.txt`, Win32 platform fixes |

**Pinned stack (MVP):**

- Python 3.12
- CUDA **12.8** (last practical major line that still targets SM70)
- PyTorch 2.10 (matching 1Cat; built against CUDA 12.x)
- `TORCH_CUDA_ARCH_LIST=7.0` / SM70

**Hard constraint — no CUDA 13:** NVIDIA CUDA Toolkit 13 removed support for pre-Turing architectures (Maxwell, Pascal, **Volta**). Tesla V100 is SM70 / Volta, so CUDA 13 toolkits, nvcc, and CUDA 13–built wheels (including current SystemPanic vllm-windows releases aimed at Ampere/Ada/Blackwell) **cannot** be the build or runtime baseline for this fork. Windows patches are ported from `vllm-for-windows`; the CUDA/Torch *version* stays on the 1Cat 12.8 / 2.10 pin.

Do **not** adopt the vllm-windows release stack (CUDA 13 + newer Torch).

## Approach

**Chosen:** Base = 1Cat-vLLM; port Windows patches on top (not the reverse).

Rejected alternatives:

1. Base = vllm-windows, port SM70/1Cat — faster “Windows builds,” but fights the pinned stack and drops/rewrites most of 1Cat’s custom kernels.
2. Three-way rebase on a shared upstream ancestor — cleaner long-term history, too slow for MVP smoke.

## Repository layout

- Working copy: full clone of 1Cat into `1Cat-vLLM-4Win`.
- Note: any design-only git history created before the import is disposable; implementation starts by replacing the workspace with a real 1Cat clone (or fetching 1Cat `main` as the root tree), then applying Windows ports as commits on top.
- Remotes (intended):
  - `origin` — this Windows fork (GitHub user/org decided at first push)
  - `upstream` — `1CatAI/1Cat-vLLM`
  - `windows-ref` — `SystemPanic/vllm-windows` (reference; not merged wholesale)

Change layers (surgical, not a full tree merge):

1. **Build / platform** — MSVC (`vcvarsall`), CMake/setuptools Windows paths, `requirements/windows.txt` (e.g. `winloop`, `triton-windows`, `xformers`, `portalocker` as needed), env vars (`DISTUTILS_USE_SDK`, `VLLM_TARGET_DEVICE=cuda`, arch list).
2. **CUDA extensions** — compile 1Cat SM70 extensions (esp. FlashAttention-V100) with MSVC; gate or remove Linux-only build deps (e.g. `patchelf`).
3. **Runtime stubs** — only blockers for single-GPU import/serve (patterns from vllm-windows where FlashInfer / Win32 paths fail).
4. **Docs** — Windows build + smoke; credit both upstream forks; document MVP limits.

## Porting and verification workflow

1. **Inventory** — list Windows-specific deltas in `vllm-for-windows` (setup/CMake/csrc/`win32` branches/requirements).
2. **Build skeleton** — minimal port so `pip install . --no-build-isolation` starts under MSVC + CUDA 12.8.
3. **SM70 extensions** — get `flash_attn_v100` (and required siblings) compiling for arch 7.0.
4. **Runtime fixes** — only what single-GPU serve needs.
5. **Smoke gate** — see below.
6. **Publish** — GitHub + README; wheel optional for “published.”

Rule: each failed build gets a minimal fix and retry; no scope creep into multi-GPU or unrelated Linux parity.

## Smoke criteria (MVP done)

On the user’s machine (native Windows, 1× V100):

1. Environment reports Python 3.12, CUDA 12.8, Torch 2.10, GPU name contains V100.
2. Imports succeed in the spirit of 1Cat’s README verification (`vllm`, `flash_attn_v100` / related CUDA modules).
3. Short `vllm serve` on **one GPU** completes one minimal chat/completions request without crash. Prefer the smallest model the user already has locally that 1Cat can load on a single V100 16GB; if none, use a tiny publicly available HF causal LM suitable for smoke (exact ID chosen during implementation plan). Cap `--max-model-len` low enough to fit. Multi-GPU flags must not be required.

**Ready to publish when:** smoke is green locally; README states stack pins and known limits (no TP in MVP).

## Publishing

- Create/push public GitHub repo after smoke (account/org chosen at push time).
- README: Windows install from source, pinned stack, smoke steps, limitations.
- Attribution: based on 1CatAI/1Cat-vLLM; Windows adaptation informed by SystemPanic/vllm-windows.
- Prebuilt wheel in Releases: desirable, not required for MVP “published.”

## Risks

| Risk | Mitigation |
|---|---|
| MSVC cannot compile 1Cat CUDA kernels as-is | Port SystemPanic MSVC fixes; fix SM70 code only as needed for compile |
| Linux-only deps (`patchelf`, some FlashInfer paths) | Conditional skip; Windows package substitutes from `requirements/windows.txt` |
| Accidental pull of CUDA 13 / new Torch | **Blocked by arch support:** CUDA 13 dropped Volta; keep hard pin 12.8/2.10 in docs and requirements |
| V100 driver / CUDA visibility on Windows | Verify `torch.cuda` before investing in long compiles |
| Large version skew between forks | Inventory + cherry-pick mindset; no wholesale merge |

## Success metrics

- MVP: smoke checklist above passes on 1× V100 Windows.
- Publish: public repo + Windows README exist; clone/build instructions are reproducible on the same stack.

## Next step

Implementation plan: `docs/superpowers/plans/2026-09-26-1cat-vllm-windows.md`. Choose subagent-driven or inline execution.
