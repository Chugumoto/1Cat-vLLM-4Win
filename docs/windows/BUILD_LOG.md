# Windows SM70 Build Log

Environment: Windows, Python 3.12.7, Torch 2.10.0+cu128, CUDA 12.8,
Tesla V100-SXM2-32GB, Visual Studio 2026 x64 tools.

## Preflight

- `scripts/windows/preflight_env.py`: `PREFLIGHT OK`.
- Build environment uses `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=0`, CUDA 12.8, and `TORCH_CUDA_ARCH_LIST=7.0`.

## Dependency installation

- `requirements/build/cuda.txt`: installed successfully.
- `requirements/cuda.txt` initially failed while resolving
  `nvidia-cutlass-dsl[cu13]==4.7.0`: its required
  `nvidia-cutlass-dsl-libs-base==4.7.0` had no matching distribution on
  Windows (only 4.8.0 was available).
- Fix: gated the explicitly CUDA 13-only `nvidia-cutlass-dsl[cu13]` and
  `humming-kernels[cu13]` entries for this CUDA 12.8 Windows MVP, as allowed
  by the build brief. The rest of the CUDA requirements remain enabled.
- Retried `requirements/cuda.txt`: installed successfully without changing
  Torch 2.10.0+cu128.
- `requirements/windows.txt`: installed successfully.

## FlashAttention-V100 build

- Command: `python -m pip install . --no-build-isolation -vvv` with
  `TORCH_CUDA_ARCH_LIST=7.0`.
- First hard error: Torch's compiler probe reported
  `Error checking compiler version for cl: [WinError 2]`, then Ninja failed
  to launch `cl` with `CreateProcess failed: The system cannot find the file
  specified`.
- Root-cause check: `vcvarsall.bat x64` configured
  `VCToolsInstallDir=...\MSVC\14.51.36231\`, but `where cl` found nothing and
  `%VCToolsInstallDir%bin\Hostx64\x64\cl.exe` does not exist.
- Status: blocked before compilation by a missing MSVC x64 compiler binary.
  No source-level compile fix is justified until the Visual Studio C++ build
  tools are installed. The vLLM extension build was not started because it
  requires the same missing compiler.

## Final smoke

```text
SMOKE IMPORTS FAIL: ModuleNotFoundError("No module named 'vllm'")
```

Neither `flash-attn-v100` nor `1cat-vllm` is installed because the mandatory
first extension build is blocked on `cl.exe`.

## Resume attempt: bootstrap expansion failure

- `vcvarsall.bat x64` completed, and the compiler now exists at
  `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe`.
- The mandatory `where cl` gate nevertheless failed after the supplied
  one-line `cmd /c` chain. The resulting environment had CUDA 13 at the front
  of `PATH` and no MSVC compiler directory; `CUDA_HOME` was also CUDA 13.
- Cause: `%CUDA_PATH%` and `%PATH%` were expanded by `cmd.exe` before
  `vcvarsall.bat` and preceding `set` commands executed, so the later `set
  PATH=...` replaced the Visual Studio environment with the parent process
  path. Unquoted `set NAME=value &&` assignments also retained trailing spaces.
- Per the mandatory stop rule, no compilation was attempted. Resume with
  deferred expansion and quoted assignments, then repeat `where cl`, `where
  nvcc`, and preflight before building.

## 2026-09-27 — flash_attn_v100 build fail (C1083)

- Env: vcvarsall OK, cl.exe OK, preflight OK (Torch 2.10.0+cu128, V100).
- Error: `fatal error C1083: cannot open include file: cuda_runtime.h`
- Root cause: CUDA Toolkit **v12.8 install incomplete** — `include\cuda_runtime.h` / `cuda.h` missing (v12.4 and v13.0 have them).
- Action: repair/reinstall CUDA 12.8 Development/Runtime headers, or temporarily point `CUDA_HOME` to v12.4.


## 2026-09-27 — next errors after header repair

1. C1189 unsupported MSVC (VS 2026 / `_MSC_VER >= 1950`) vs CUDA 12.8 — fixed with `-allow-unsupported-compiler`.
2. `ushort` undefined in `fused_mha_backward.cu` under MSVC — typedef `unsigned short`.

## 2026-09-27 — cudafe++ ACCESS_VIOLATION
CUDA 12.8 + VS 2026 (MSVC 14.51): `nvcc error: 'cudafe++' died with status 0xC0000005`. Need MSVC v143 (VS 2022) via `vcvarsall ... -vcvars_ver=14.3x`.


## 2026-09-27 — paged_kv PYBIND
Moved `PYBIND11_MODULE` out of `paged_to_contiguous.cu` into `paged_kv_utils_api.cpp` (MSVC needs torch/extension.h only on host).


## 2026-09-27 — setuptools-scm / git PATH
FA-V100 installed OK. 1cat-vllm metadata failed because `run_build_mvp.cmd` stripped Git from PATH. Added `Git\cmd` to PATH; skip FA rebuild when already importable.


## 2026-09-27 — git PATH for CMake cutlass
CMake failed: `could not find git for clone of cutlass-populate`. Git is at `%LOCALAPPDATA%\hermes\git\cmd`; `run_build_mvp.cmd` now probes hermes/Git installs.


## 2026-09-27 — patch.exe for FA SM70
CMake: `Could not find PATCH_EXECUTABLE` (`patch`). Added `%GIT_HOME%\usr\bin` (hermes Git) to `run_build_mvp.cmd` PATH.


## 2026-09-27 — torch_python link
Ninja failed looking for Linux `libtorch_python.so`. `CMakeLists.txt` now `find_library(torch_python)` and links `TORCH_PYTHON_LIB` (Windows: `torch_python.lib`).


## 2026-09-27 — quote CUDA -I paths
MSVC C1083: `-IC:/Program Files/NVIDIA...` split on spaces. Quote `-I` like vllm-windows.


## 2026-09-27 — MSVC host types
- `cumem_allocator.cpp`: `ssize_t` > `ptrdiff_t` (vllm-windows).
- `spinloop.cpp`: `clock_gettime` via QueryPerformanceCounter.
- `activation_kernels.cu`: `__int128_t`/`__int64_t` typedefs for MSVC.


## 2026-09-27 — MAYBE_HOST_DEVICE on MSVC
`utils.cuh`: drop `C10_HOST_DEVICE` on constexpr var templates under `_MSC_VER` (same idea as ROCm empty define).


## 2026-09-27 — MSVC template data_ptr
`nvfp4_qpn4_sm70.cu`: use `.template data_ptr<T>()` inside templates; `MAYBE_HOST_DEVICE` = `__device__` on MSVC.


## 2026-09-27 — quant_type_max_v
CUDA+MSVC rejects device variable templates. Replaced `quant_type_max_v<T>` with `quant_type_max<T>::val()`.


## 2026-09-27 — bulk .template data_ptr
MSVC dependent-name fix: `.template data_ptr<T>()` inside template functions across SM70/marlin/moe CUDA sources.


## 2026-09-27 — next_pow_2 without __builtin_clz
`csrc/core/math.hpp`: portable bit-smear `next_pow_2` (MSVC has no `__builtin_clz`).


## 2026-09-27 — M_SQRT2
`libtorch_stable/activation_kernels.cu`: `#define _USE_MATH_DEFINES` before `cmath` (MSVC).


## 2026-09-27 — uint / torch_utils include
- `fused_qknorm_rope_kernel.cu`: `uint` > `unsigned int` (MSVC).
- `attention/dtype_fp8.cuh`: include `../torch_utils.h`.


## 2026-09-27 — bulk uint>unsigned int
Standalone `uint` is not a type on MSVC; replaced across `csrc` (`uint2`/`uint8_t` untouched).


## 2026-09-27 — device isinf
`merge_attn_states.cu`: replace `std::isinf`/`::isinf` with `fabsf(x)==INFINITY` (MSVC host template).


## 2026-09-27 — mamba/topk MSVC
- `selective_scan_fwd.cu`: `M_LOG2E`; hoist `#ifdef` out of lambda via `set_max_dynamic_shared_memory`.
- `persistent_topk.cuh`: `__forceinline` instead of GCC `always_inline`.


## 2026-09-27 — asm volatile
`awq/gemm_kernels.cu`: `__asm__` > `asm` for MSVC+nvcc.


## 2026-09-27 — moe MSVC
- softplus: hoist launch `#ifdef` out of lambda.
- grouped_topk: `__attribute((aligned))` > `alignas`.


## 2026-09-27 — cublasLt link
`_C.pyd` LNK2019 missing `cublasLt*`; link `CUDA::cublas`/`CUDA::cublasLt` on WIN32.


## 2026-09-27 — cublas for stable/moe
Link `cublas`/`cublasLt` into `_C_stable_libtorch` and `_moe_C` on WIN32 (LNK2019 `cublasHgemm`).


## 2026-09-27 — gdn C2872
Split `gdn_api.cpp` + `torch_cuda_compat.h` for FlashQLA SM70 (same MSVC/CCCL `std` clash as FA-V100).


## 2026-09-27 — BUILD MVP OK
`scripts/windows/run_build_mvp.cmd` completed: `Successfully installed 1cat-vllm-1.5.1.dev0+windows`, `SMOKE IMPORTS OK` (Torch 2.10.0+cu128, flash_attn_v100 1.2.0, V100).

