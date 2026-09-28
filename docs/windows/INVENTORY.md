# Windows port inventory (from SystemPanic/vllm-windows @ `windows-ref/vllm-for-windows`)

Checklist for porting Windows build support into the 1Cat V100 (CUDA 12.8 / SM70) fork. Generated from `git grep` against `windows-ref/vllm-for-windows` (2026-09-26).

**Local gap:** 1Cat `setup.py` and `CMakeLists.txt` currently have **no** `IS_WINDOWS` / `WIN32` branches; items below are absent upstream of this fork until ported.

---

## Do port

| Source file | Why | Ref hits (summary) |
|---|---|---|
| `setup.py` | Build entry on Win32: `IS_WINDOWS`, FA3 disable (`VLLM_FORCE_FA3_WINDOWS_BUILD`), path/`PYTHONPATH` joins, sccache vs ccache, append `windows.txt` reqs | L81, L91, L115–116, L276, L295–339, L1374–1375 |
| `CMakeLists.txt` | MSVC CUDA host flags (`/Zc:__cplusplus`, `WIN32_LEAN_AND_MEAN`, `-Xcompiler=…`); invoke `fix_cutlass_msvc.py` after FetchContent cutlass | L488–509, L541–544, L1493 |
| `cmake/external_projects/vllm_flash_attn.cmake` | Flash-attn subproject: `WIN32` preprocessor flags + cutlass MSVC patch | L63, L74, L85 |
| `fix_cutlass_msvc.py` | Patch cutlass headers (`platform.h`, `cuda_host_adapter.hpp`) for MSVC | repo root |
| `requirements/windows.txt` | Runtime deps: winloop, triton-windows, portalocker, xformers | 4 packages |

### Additional cmake / tooling hits (port with build stack)

| Source file | Why |
|---|---|
| `cmake/external_projects/flashkda.cmake` | `WIN32` branch; calls `fix_cutlass_msvc.py` on flashkda cutlass tree |
| `cmake/external_projects/deepgemm.cmake` | `WIN32`-guarded build paths |
| `cmake/external_projects/qutlass.cmake` | Multiple `WIN32` / `NOT WIN32` branches |
| `cmake/external_projects/triton_kernels.cmake` | `WIN32` guard |
| `tools/build_deepgemm_C.py` | MSVC defines (`/DWIN32_LEAN_AND_MEAN`, `/utf-8`, `/Zc:preprocessor`, …) |

### Runtime / diagnostics (low risk, cherry-pick if needed)

| Source file | Why |
|---|---|
| `vllm/collect_env.py` | `where`, `nvidia-smi`, `wmic` paths on win32 |
| `vllm/distributed/utils.py` | `USE_SCHED_YIELD` excludes win32 |

---

## Do NOT port

| Source file / pattern | Why |
|---|---|
| `fix_cuda_13_align.py` | CUDA 13 tensor-map alignment only; V100 MVP targets CUDA 12.8 |
| Torch/CUDA 13 pins in `requirements/build/cuda.txt` and `requirements/cuda.txt` (`torch==…+cu130`, `torchvision==…+cu130`, `sys_platform == "win32"`) | Breaks SM70 / CUDA 12.8 stack; keep 1Cat CUDA 12 pins |
| Win32-only wheels in their `requirements/cuda.txt` (SystemPanic flashinfer/humming forks, older flashinfer-cubin, cudnn-frontend cap, tilelang downgrade, skip instanttensor / cutlass-dsl cu13 extras on win32) | Fork-specific CUDA 13 ecosystem; re-derive for CUDA 12.8 Windows instead of copying |
| NCCL multi-GPU / distributed hardening beyond Linux MVP | Out of MVP scope (ref still uses NCCL throughout `vllm/v1/worker/*`; do not expand scope to match) |

---

## 1Cat-specific (no upstream Windows analog)

| File | Windows work |
|---|---|
| `flash-attention-v100/setup.py` | Replace GCC `-O3` / `-std=c++17` with MSVC `/O2` / `/std:c++17` when `os.name == "nt"` (currently GCC flags only) |
| `pyproject.toml` `patchelf` build-require | Make Windows-conditional or omit on win32 (`patchelf>=0.19.1.0` today) |
| `setup.py` portable wheel `patchelf --remove-rpath` | Guard or replace on Windows (no patchelf on win32) |
| `requirements/cuda.txt` packages tagged `[cu13]` / SM70 CUDA 12 resolution in `setup.py` | Gate or skip on Windows MVP if they pull CUDA 13-only wheels; keep `[cu12]` / stripped extras on CUDA 12 builds |

---

## Raw scan command (repeatable)

```powershell
Set-Location C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win
git grep -n -i "WIN32\|IS_WINDOWS\|windows.txt\|fix_cutlass_msvc\|VLLM_FORCE_FA3" `
  windows-ref/vllm-for-windows -- setup.py CMakeLists.txt cmake requirements vllm/envs.py
```

Fallback probe clone: `C:\Users\Chugumoto\Projects\_tmp_fork_probe\win`.
