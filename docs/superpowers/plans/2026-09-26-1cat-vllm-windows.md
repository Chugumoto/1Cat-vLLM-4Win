# 1Cat-vLLM Windows (V100) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Windows-native build of 1Cat-vLLM that compiles under MSVC + CUDA 12.8 + Torch 2.10 and passes single-V100 smoke (imports + short `vllm serve`), then publish the fork.

**Architecture:** Keep `1CatAI/1Cat-vLLM` as the base tree. Surgically port Windows build/runtime patches from `SystemPanic/vllm-windows` branch `vllm-for-windows` (setup/CMake/MSVC/requirements). Never adopt CUDA 13 or newer Torch from that fork. FlashAttention-V100 stays a first-class SM70 extension with MSVC-compatible compile flags.

**Tech Stack:** Python 3.12, CUDA Toolkit 12.8, PyTorch 2.10 (+ cu128 wheels), MSVC (VS 2019+), CMake, Ninja, `flash_attn_v100` (in-tree under `flash-attention-v100/`).

## Global Constraints

- Base tree = `1CatAI/1Cat-vLLM` `main`; Windows patches from `SystemPanic/vllm-windows` `vllm-for-windows` only as reference.
- CUDA **12.8** only — CUDA 13 dropped Volta/SM70; do not install CUDA 13 toolkit or CUDA 13 wheels.
- PyTorch **2.10.0** (CUDA 12.x ABI), matching 1Cat.
- `TORCH_CUDA_ARCH_LIST=7.0` / `CMAKE_CUDA_ARCHITECTURES=70`.
- MVP = **1× Tesla V100**, native Windows (not WSL); no TP/NCCL requirement.
- Do not wholesale-merge the two repos; cherry-pick/port minimal diffs.
- Do not copy `fix_cuda_13_align.py` (CUDA 13-only).
- Preserve `docs/superpowers/**` when replacing the workspace with the 1Cat clone.
- Commits require a configured git `user.name` / `user.email` (do not invent global git config; ask the user if missing).

---

## File map (create / modify)

| Path | Responsibility |
|---|---|
| `docs/superpowers/specs/2026-09-26-1cat-vllm-windows-design.md` | Approved design (already present; keep) |
| `docs/windows/INVENTORY.md` | Catalog of Windows deltas taken from `vllm-for-windows` |
| `requirements/windows.txt` | Windows-only Python deps |
| `requirements/build/cuda.txt` | Drop/gate Linux-only `patchelf`; keep Torch 2.10 |
| `pyproject.toml` | Mirror build-requires (no `patchelf` on win32) |
| `setup.py` | `IS_WINDOWS`, allow Windows platform, disable FA3 on Windows, path fixes, pull `windows.txt` |
| `CMakeLists.txt` | Port `WIN32` MSVC CUDA flags + `fix_cutlass_msvc.py` hook |
| `cmake/external_projects/vllm_flash_attn.cmake` | Port `WIN32` `/Zc:preprocessor` + cutlass fix hook |
| `fix_cutlass_msvc.py` | Copied from SystemPanic (MSVC cutlass header patch) |
| `flash-attention-v100/setup.py` | MSVC-safe `cxx` flags (`/O2`, `/std:c++17`) instead of GCC `-O3` |
| `scripts/windows/env_build.cmd` | `vcvarsall` + env pins for build |
| `scripts/windows/preflight_env.py` | Fail if wrong CUDA/Torch/GPU |
| `scripts/windows/smoke_imports.py` | Import gate after build |
| `scripts/windows/smoke_serve.ps1` | Single-GPU serve + one HTTP completion |
| `tests/windows/test_setup_windows_guards.py` | Unit tests for Windows platform guards / requirements selection |
| `README.windows.md` | Windows install, pins, smoke, limits, attribution |

Reference clone (read-only): keep or re-clone `SystemPanic/vllm-windows` at `C:\Users\Chugumoto\Projects\_tmp_fork_probe\win` (branch `vllm-for-windows`) while porting.

Smoke model default: `facebook/opt-125m` (tiny, fits 1× V100 16GB with low `--max-model-len`). Override with env `VLLM_SMOKE_MODEL` if the user already has a local path.

---

### Task 1: Bootstrap workspace from 1Cat (preserve design docs)

**Files:**
- Create: full 1Cat tree under `C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win`
- Keep: `docs/superpowers/specs/2026-09-26-1cat-vllm-windows-design.md`, `docs/superpowers/plans/2026-09-26-1cat-vllm-windows.md`

**Interfaces:**
- Consumes: approved design + this plan on disk
- Produces: git repo rooted at 1Cat `main`, remotes `upstream` + `windows-ref`, design/plan still present

- [ ] **Step 1: Stash design docs outside the workspace**

```powershell
$root = "C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win"
$stash = "C:\Users\Chugumoto\Projects\_1cat_win_docs_stash"
New-Item -ItemType Directory -Force -Path $stash | Out-Null
Copy-Item -Recurse -Force "$root\docs" "$stash\docs"
```

- [ ] **Step 2: Replace workspace with 1Cat clone**

```powershell
$root = "C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win"
Remove-Item -Recurse -Force $root
git clone https://github.com/1CatAI/1Cat-vLLM.git $root
Set-Location $root
git remote rename origin upstream
git remote add windows-ref https://github.com/SystemPanic/vllm-windows.git
git fetch windows-ref vllm-for-windows
```

Expected: `git log -1 --oneline` shows a 1Cat commit; `git remote -v` lists `upstream` and `windows-ref`.

- [ ] **Step 3: Restore design/plan docs**

```powershell
$root = "C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win"
$stash = "C:\Users\Chugumoto\Projects\_1cat_win_docs_stash"
New-Item -ItemType Directory -Force -Path "$root\docs\superpowers" | Out-Null
Copy-Item -Recurse -Force "$stash\docs\superpowers\*" "$root\docs\superpowers\"
```

- [ ] **Step 4: Verify docs exist**

```powershell
Test-Path "$root\docs\superpowers\specs\2026-09-26-1cat-vllm-windows-design.md"
Test-Path "$root\docs\superpowers\plans\2026-09-26-1cat-vllm-windows.md"
```

Expected: both `True`.

- [ ] **Step 5: Commit restored docs on top of 1Cat**

```powershell
Set-Location C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win
git add docs/superpowers
git commit -m "docs: add Windows V100 fork design and implementation plan"
```

If commit fails with missing identity, stop and ask the user to set local `user.name` / `user.email` (do not run `git config --global`).

---

### Task 2: Preflight + failing smoke scaffolds (TDD gate)

**Files:**
- Create: `scripts/windows/preflight_env.py`
- Create: `scripts/windows/smoke_imports.py`
- Create: `tests/windows/test_preflight_contract.py`
- Create: `scripts/windows/env_build.cmd`

**Interfaces:**
- Consumes: system Python/CUDA/Torch/GPU
- Produces: `preflight_env.main() -> int` (0 ok); `smoke_imports.main() -> int` (0 ok)

- [ ] **Step 1: Write failing contract test**

Create `tests/windows/test_preflight_contract.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_preflight_script_exists():
    assert (ROOT / "scripts/windows/preflight_env.py").is_file()


def test_smoke_imports_script_exists():
    assert (ROOT / "scripts/windows/smoke_imports.py").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
Set-Location C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win
python -m pytest tests/windows/test_preflight_contract.py -v
```

Expected: FAIL — file not found / import path missing.

- [ ] **Step 3: Add preflight + smoke import scripts**

Create `scripts/windows/preflight_env.py`:

```python
"""Fail closed unless Windows V100 MVP stack is present."""
from __future__ import annotations

import platform
import sys


def main() -> int:
    errors: list[str] = []
    if platform.system() != "Windows":
        errors.append(f"OS must be Windows, got {platform.system()}")

    if not (sys.version_info.major == 3 and sys.version_info.minor == 12):
        errors.append(f"Python 3.12 required, got {sys.version.split()[0]}")

    try:
        import torch
    except ImportError:
        errors.append("torch is not installed")
        _print(errors)
        return 1

    ver = torch.__version__
    if not ver.startswith("2.10"):
        errors.append(f"torch 2.10.x required, got {ver}")

    cuda = torch.version.cuda
    if cuda is None or not str(cuda).startswith("12."):
        errors.append(f"torch must be CUDA 12.x build, got cuda={cuda}")

    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is False")
    else:
        name = torch.cuda.get_device_name(0)
        if "V100" not in name.upper() and "TESLA V100" not in name.upper():
            # Allow substring V100 in common names like "Tesla V100-SXM2-16GB"
            if "V100" not in name:
                errors.append(f"GPU0 must be Tesla V100 for MVP smoke, got {name!r}")

        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) != (7, 0):
            errors.append(f"expect SM 7.0, got {major}.{minor}")

    _print(errors)
    return 1 if errors else 0


def _print(errors: list[str]) -> None:
    if errors:
        print("PREFLIGHT FAIL:")
        for e in errors:
            print(f"  - {e}")
    else:
        import torch

        print("PREFLIGHT OK")
        print("  Python", sys.version.split()[0])
        print("  Torch", torch.__version__, "CUDA", torch.version.cuda)
        print("  GPU", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `scripts/windows/smoke_imports.py`:

```python
"""Import gate after Windows build. Fails until extensions install."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
        import vllm
        import flash_attn_v100
        from flash_attn_v100 import flash_attn_v100_cuda, paged_kv_utils
    except Exception as exc:  # noqa: BLE001 - smoke must show any failure
        print("SMOKE IMPORTS FAIL:", repr(exc))
        return 1

    print("SMOKE IMPORTS OK")
    print("  Torch", torch.__version__)
    print("  vLLM", getattr(vllm, "__version__", "?"))
    print("  flash_attn_v100", getattr(flash_attn_v100, "__version__", "?"))
    print("  modules", flash_attn_v100_cuda, paged_kv_utils)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `scripts/windows/env_build.cmd`:

```bat
@echo off
REM Call from "x64 Native Tools" OR after vcvarsall.
REM Edit VS path if needed.
if not defined VSCMD_ARG_TGT_ARCH (
  echo ERROR: run vcvarsall.bat x64 first
  exit /b 1
)

set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
set TORCH_CUDA_ARCH_LIST=7.0
set CMAKE_CUDA_ARCHITECTURES=70
set MAX_JOBS=8
set VLLM_DISABLE_FA3_BUILD=1

echo Build env ready for SM70 / CUDA 12.8
```

- [ ] **Step 4: Re-run contract test**

```powershell
python -m pytest tests/windows/test_preflight_contract.py -v
```

Expected: PASS.

- [ ] **Step 5: Run preflight (may fail until Torch/CUDA installed — that is OK)**

```powershell
python scripts\windows\preflight_env.py
```

Expected: either `PREFLIGHT OK` or a clear missing-dep list. Do not proceed to long compiles until OK.

- [ ] **Step 6: Confirm smoke_imports fails before build**

```powershell
python scripts\windows\smoke_imports.py
```

Expected: `SMOKE IMPORTS FAIL` (modules not installed yet).

- [ ] **Step 7: Commit**

```powershell
git add scripts/windows tests/windows
git commit -m "test: add Windows preflight and import smoke scaffolds"
```

---

### Task 3: Inventory Windows deltas from `vllm-for-windows`

**Files:**
- Create: `docs/windows/INVENTORY.md`

**Interfaces:**
- Consumes: `windows-ref/vllm-for-windows` vs local 1Cat tree
- Produces: checklist of files to port (used by Tasks 4–7)

- [ ] **Step 1: Generate a raw hit list**

```powershell
Set-Location C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win
git grep -n -i "WIN32\|IS_WINDOWS\|windows.txt\|fix_cutlass_msvc\|VLLM_FORCE_FA3" windows-ref/vllm-for-windows -- setup.py CMakeLists.txt cmake requirements vllm/envs.py 2>$null
# Fallback if git grep remote path unsupported: use local probe clone
rg -n -i "WIN32|IS_WINDOWS|windows.txt|fix_cutlass_msvc|VLLM_FORCE_FA3" C:\Users\Chugumoto\Projects\_tmp_fork_probe\win\setup.py C:\Users\Chugumoto\Projects\_tmp_fork_probe\win\CMakeLists.txt C:\Users\Chugumoto\Projects\_tmp_fork_probe\win\cmake C:\Users\Chugumoto\Projects\_tmp_fork_probe\win\requirements
```

- [ ] **Step 2: Write `docs/windows/INVENTORY.md` with at least these known entries**

```markdown
# Windows port inventory (from SystemPanic/vllm-windows @ vllm-for-windows)

## Do port
| Source file | Why |
|---|---|
| `setup.py` (`IS_WINDOWS`, FA3 disable, path joins, `windows.txt`) | Build entry on Win32 |
| `CMakeLists.txt` (`if (WIN32)` MSVC flags + cutlass fix) | Compile CUDA/C++ with MSVC |
| `cmake/external_projects/vllm_flash_attn.cmake` (`WIN32` preprocessor flags) | Flash-attn subproject |
| `fix_cutlass_msvc.py` | Patch cutlass headers for MSVC |
| `requirements/windows.txt` | winloop / triton-windows / portalocker / xformers |

## Do NOT port
| Source file | Why |
|---|---|
| `fix_cuda_13_align.py` | CUDA 13 only; V100 requires CUDA 12.8 |
| Torch/CUDA 13 pins in their `requirements/build/cuda.txt` | Breaks SM70 |
| NCCL multi-GPU bits | Out of MVP scope |

## 1Cat-specific (no upstream Windows analog)
| File | Windows work |
|---|---|
| `flash-attention-v100/setup.py` | Replace GCC `-O3`/`-std=c++17` with MSVC `/O2` `/std:c++17` when `os.name == "nt"` |
| `pyproject.toml` `patchelf` build-require | Make Windows-conditional / omit on win32 |
| `requirements/cuda.txt` packages tagged `[cu13]` | Gate or skip on Windows MVP if they pull CUDA 13 |
```

- [ ] **Step 3: Commit**

```powershell
git add docs/windows/INVENTORY.md
git commit -m "docs: inventory Windows deltas for V100 fork"
```

---

### Task 4: Requirements + build metadata for Windows / no patchelf

**Files:**
- Create: `requirements/windows.txt`
- Modify: `requirements/build/cuda.txt`
- Modify: `pyproject.toml` (build-system requires)
- Test: `tests/windows/test_requirements_windows.py`

**Interfaces:**
- Consumes: inventory
- Produces: installable Windows requirement set without Linux-only `patchelf`

- [ ] **Step 1: Write failing test**

Create `tests/windows/test_requirements_windows.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_windows_requirements_exist_and_list_core_deps():
    text = (ROOT / "requirements/windows.txt").read_text(encoding="utf-8")
    assert "winloop" in text
    assert "triton-windows" in text
    assert "portalocker" in text


def test_build_cuda_txt_omits_unconditional_patchelf():
    text = (ROOT / "requirements/build/cuda.txt").read_text(encoding="utf-8")
    # Either removed, or clearly environment-marked; bare unconditional pin is forbidden
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        assert not s.startswith("patchelf"), "patchelf must not be unconditional on Windows builds"
```

- [ ] **Step 2: Run to verify fail**

```powershell
python -m pytest tests/windows/test_requirements_windows.py -v
```

Expected: FAIL (missing `windows.txt` / patchelf still unconditional).

- [ ] **Step 3: Add `requirements/windows.txt`**

```text
winloop
triton-windows
portalocker
# xformers: install a Torch-2.10-compatible Windows wheel if needed for the chosen attention path.
# Do not blindly pin SystemPanic's 0.0.35 if it requires newer torch.
```

Start without a hard `xformers==…` pin; add a concrete Windows wheel pin only after confirming it installs on Torch 2.10. Prefer leaving xformers out of MVP if FlashAttention-V100 path does not need it for smoke.

- [ ] **Step 4: Gate `patchelf` in `requirements/build/cuda.txt` and `pyproject.toml`**

In `requirements/build/cuda.txt`, change:

```text
patchelf>=0.19.1.0
```

to:

```text
patchelf>=0.19.1.0; sys_platform == "linux"
```

In `pyproject.toml` `[build-system].requires`, replace the bare `"patchelf>=0.19.1.0"` entry with the same environment marker form if setuptools/PEP 508 markers are accepted there; if the build backend rejects markers in `requires`, remove `patchelf` from `pyproject.toml` and keep it only in `requirements/build/cuda.txt` (Linux CI / Linux wheels).

Also keep `torch == 2.10.0` — do **not** switch to SystemPanic’s `2.11+cu130`.

- [ ] **Step 5: Run tests**

```powershell
python -m pytest tests/windows/test_requirements_windows.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add requirements/windows.txt requirements/build/cuda.txt pyproject.toml tests/windows/test_requirements_windows.py
git commit -m "build: add Windows requirements and gate patchelf"
```

---

### Task 5: Port `setup.py` Windows guards from SystemPanic

**Files:**
- Modify: `setup.py`
- Test: `tests/windows/test_setup_windows_guards.py`

**Interfaces:**
- Consumes: SystemPanic `setup.py` patterns (`IS_WINDOWS`, FA3 disable, `windows.txt` include)
- Produces: 1Cat `setup.py` that accepts Windows as a first-class platform

- [ ] **Step 1: Write failing unit test for helper behavior**

Create `tests/windows/test_setup_windows_guards.py`:

```python
"""Lightweight checks that Windows platform constants exist after port."""
from pathlib import Path
import re

SETUP = Path(__file__).resolve().parents[2] / "setup.py"


def test_setup_defines_is_windows():
    text = SETUP.read_text(encoding="utf-8")
    assert "IS_WINDOWS" in text
    assert 'platform.system() == "Windows"' in text


def test_setup_disables_fa3_on_windows():
    text = SETUP.read_text(encoding="utf-8")
    assert "VLLM_DISABLE_FA3_BUILD" in text
    assert "IS_WINDOWS" in text


def test_setup_includes_windows_requirements():
    text = SETUP.read_text(encoding="utf-8")
    assert 'windows.txt' in text
```

- [ ] **Step 2: Run — expect FAIL**

```powershell
python -m pytest tests/windows/test_setup_windows_guards.py -v
```

- [ ] **Step 3: Port the minimal Windows blocks into 1Cat `setup.py`**

Near the top platform detection (today 1Cat only allows linux/darwin), mirror SystemPanic:

```python
IS_WINDOWS = platform.system() == "Windows"

# In the platform gate, allow Windows:
elif not (sys.platform.startswith("linux") or sys.platform.startswith("darwin")
          or IS_WINDOWS):
    ...

IS_WSL = ("microsoft-standard-WSL2" in platform.uname().release
          or "-Microsoft" in platform.uname().release)

if ((IS_WINDOWS or IS_WSL)
        and os.environ.get("VLLM_FORCE_FA3_WINDOWS_BUILD", "0") != "1"):
    os.environ["VLLM_DISABLE_FA3_BUILD"] = "1"
```

In the CMake/extension path-building helpers, port SystemPanic’s Windows path normalization (`\` → `/` for CMake) wherever 1Cat currently joins `sys.path` / compiler launchers.

In the requirements aggregation function (SystemPanic ~line where `_read_requirements("windows.txt")` is extended when `IS_WINDOWS`), add the same for 1Cat.

Skip Linux-only `patchelf` rpath rewriting when `IS_WINDOWS` (1Cat currently calls `patchelf` around the SM70 wheel portability helper near the `which("patchelf")` site).

- [ ] **Step 4: Run tests — expect PASS**

```powershell
python -m pytest tests/windows/test_setup_windows_guards.py -v
```

- [ ] **Step 5: Commit**

```powershell
git add setup.py tests/windows/test_setup_windows_guards.py
git commit -m "build: teach setup.py Windows platform and FA3 disable"
```

---

### Task 6: Port CMake / cutlass MSVC fixes

**Files:**
- Create: `fix_cutlass_msvc.py` (copy from SystemPanic)
- Modify: `CMakeLists.txt` (add `if (WIN32)` blocks analogous to SystemPanic)
- Modify: `cmake/external_projects/vllm_flash_attn.cmake` (WIN32 preprocessor + cutlass fix)

**Interfaces:**
- Consumes: SystemPanic WIN32 flag lists
- Produces: MSVC-capable CMake for CUDA 12.8 / SM70

- [ ] **Step 1: Copy `fix_cutlass_msvc.py` unchanged from SystemPanic**

```powershell
Copy-Item C:\Users\Chugumoto\Projects\_tmp_fork_probe\win\fix_cutlass_msvc.py `
  C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win\fix_cutlass_msvc.py
```

Do **not** copy `fix_cuda_13_align.py`.

- [ ] **Step 2: Insert WIN32 compile flags into 1Cat `CMakeLists.txt`**

Locate the CUTLASS / GPU flags section in 1Cat’s `CMakeLists.txt` (same structural area as SystemPanic ~line 488). Add:

```cmake
if (WIN32)
  set(CUTLASS_ENABLE_CUBLAS ON CACHE BOOL "cuBLAS enabled for Cutlass")
  set(CUBLAS_ENABLED ON CACHE BOOL "cuBLAS enabled")
  list(APPEND VLLM_GPU_FLAGS "--disable-warnings")
  list(APPEND VLLM_GPU_FLAGS "-O2")
  list(APPEND VLLM_GPU_FLAGS "-FS")
  list(APPEND VLLM_GPU_FLAGS "-Xptxas=-O2")
  list(APPEND VLLM_GPU_FLAGS "-Xcompiler=/O2")
  list(APPEND VLLM_GPU_FLAGS "-Xcompiler=/FS")
  list(APPEND VLLM_GPU_FLAGS "-Xcompiler=/Z7")
  list(APPEND VLLM_GPU_FLAGS "-Xcompiler=/Zc:__cplusplus")
  list(APPEND VLLM_GPU_FLAGS "-Xcompiler=/Zc:preprocessor")
  list(APPEND VLLM_GPU_FLAGS "-Xcompiler=/DWIN32_LEAN_AND_MEAN")
  list(APPEND VLLM_GPU_FLAGS "-Xcompiler=/DUSE_CUDA")
  string(REPLACE "/Zi" "/Z7" VLLM_GPU_FLAGS "${VLLM_GPU_FLAGS}")
  string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
  string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
  string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus /Zc:preprocessor /DWIN32_LEAN_AND_MEAN")
  set(CMAKE_CUDA_FLAGS_DEBUG "")
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/csrc")
endif()
```

After `FetchContent_MakeAvailable(cutlass)` (or equivalent in 1Cat), add:

```cmake
if (WIN32)
  find_package(PythonInterp)
  find_package(Python)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fix_cutlass_msvc.py ${FETCHCONTENT_BASE_DIR}/cutlass-src)
endif()
```

Adapt `${FETCHCONTENT_BASE_DIR}/cutlass-src` if 1Cat’s FetchContent directory name differs — verify after first configure.

- [ ] **Step 3: Port WIN32 block into `cmake/external_projects/vllm_flash_attn.cmake`**

```cmake
if (WIN32)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor")
endif()
```

After `FetchContent_MakeAvailable(vllm-flash-attn)`:

```cmake
if (WIN32)
  foreach(_fa_tgt _vllm_fa2_C _vllm_fa3_C)
    if(TARGET ${_fa_tgt})
      target_compile_options(${_fa_tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
      )
    endif()
  endforeach()
  execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fix_cutlass_msvc.py ${vllm-flash-attn_SOURCE_DIR}/csrc/cutlass)
endif()
```

Note: FA3 targets may be disabled via `VLLM_DISABLE_FA3_BUILD`; the loop is still safe with `if(TARGET ...)`.

- [ ] **Step 4: Static check**

```powershell
rg -n "fix_cutlass_msvc|WIN32_LEAN_AND_MEAN|Zc:preprocessor" CMakeLists.txt cmake\external_projects\vllm_flash_attn.cmake fix_cutlass_msvc.py
```

Expected: hits in all three.

- [ ] **Step 5: Commit**

```powershell
git add fix_cutlass_msvc.py CMakeLists.txt cmake/external_projects/vllm_flash_attn.cmake
git commit -m "build: port MSVC/CMake Windows flags and cutlass fix"
```

---

### Task 7: Make `flash-attention-v100` MSVC-compilable

**Files:**
- Modify: `flash-attention-v100/setup.py`
- Test: `tests/windows/test_flash_attn_v100_msvc_flags.py`

**Interfaces:**
- Consumes: existing CUDAExtension definitions
- Produces: `cxx` args that work on MSVC (`/O2`, `/std:c++17`) while keeping GCC flags on Linux

- [ ] **Step 1: Failing test**

```python
from pathlib import Path

SETUP = Path(__file__).resolve().parents[2] / "flash-attention-v100" / "setup.py"


def test_msvc_cxx_flags_present():
    text = SETUP.read_text(encoding="utf-8")
    assert "/O2" in text or 'os.name == "nt"' in text
    assert "std=c++17" in text or "/std:c++17" in text
```

- [ ] **Step 2: Run — expect FAIL** (currently only `-O3`)

- [ ] **Step 3: Patch `get_ext_modules()` compile args**

```python
import os

def _cxx_args():
    if os.name == "nt":
        return ["/O2", "/std:c++17"]
    return ["-O3", "-std=c++17"]

# In each CUDAExtension extra_compile_args:
extra_compile_args={
    "cxx": _cxx_args(),
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-gencode",
        "arch=compute_70,code=sm_70",
        # ... keep existing -U__CUDA_NO_HALF* and expt flags ...
    ],
},
```

Keep `arch=compute_70,code=sm_70`. Do not add SM80+ arches.

- [ ] **Step 4: Run test — PASS**

- [ ] **Step 5: Commit**

```powershell
git add flash-attention-v100/setup.py tests/windows/test_flash_attn_v100_msvc_flags.py
git commit -m "build: MSVC cxx flags for flash_attn_v100"
```

---

### Task 8: First Windows build loop (extensions + vLLM)

**Files:**
- Modify: only as compile errors demand (prefer tiny fixes; log each in commit messages)
- Create: `docs/windows/BUILD_LOG.md` (append error → fix entries)

**Interfaces:**
- Consumes: Tasks 4–7
- Produces: `pip install` success for `flash_attn_v100` and `1cat-vllm`

- [ ] **Step 1: Install pinned Torch (CUDA 12.8 index)**

From an elevated-enough developer cmd **after** `vcvarsall.bat x64`:

```bat
call "%VSINSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win
call scripts\windows\env_build.cmd
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python scripts\windows\preflight_env.py
```

Expected: `PREFLIGHT OK`. If the cu128 index URL/tag differs on the day of build, use the official PyTorch install selector for **Torch 2.10 + CUDA 12.8** — still never CUDA 13.

- [ ] **Step 2: Install build + runtime requirements**

```bat
python -m pip install -r requirements\build\cuda.txt
python -m pip install -r requirements\cuda.txt
python -m pip install -r requirements\windows.txt
```

If any package in `requirements/cuda.txt` hard-requires CUDA 13 extras (`nvidia-cutlass-dsl[cu13]`, `humming-kernels[cu13]`, etc.) and fails on Windows/12.8: comment those lines behind a short note in `requirements/cuda.txt` and record in `docs/windows/BUILD_LOG.md`. Do not upgrade the whole stack to CUDA 13.

- [ ] **Step 3: Build FlashAttention-V100 first**

```bat
cd flash-attention-v100
set TORCH_CUDA_ARCH_LIST=7.0
python -m pip install . --no-build-isolation -vvv
cd ..
```

On failure: capture the first hard error, apply the minimal MSVC/CUDA fix, append to `BUILD_LOG.md`, rebuild. Repeat until install succeeds.

- [ ] **Step 4: Build 1Cat-vLLM**

```bat
python -m pip install . --no-build-isolation -vvv
```

Same error→minimal-fix loop. Prefer porting additional SystemPanic snippets over inventing new abstractions.

- [ ] **Step 5: Import smoke must pass**

```bat
python scripts\windows\smoke_imports.py
```

Expected: `SMOKE IMPORTS OK`.

- [ ] **Step 6: Commit build fixes + log**

```powershell
git add docs/windows/BUILD_LOG.md
git add -u
git commit -m "build: Windows MSVC compile fixes for SM70 MVP"
```

---

### Task 9: Single-GPU serve smoke

**Files:**
- Create: `scripts/windows/smoke_serve.ps1`
- Modify: `docs/windows/BUILD_LOG.md` (record model used)

**Interfaces:**
- Consumes: installed package + V100
- Produces: HTTP 200 from OpenAI-compatible `/v1/completions` (or `/v1/chat/completions`)

- [ ] **Step 1: Write serve smoke script**

Create `scripts/windows/smoke_serve.ps1`:

```powershell
$ErrorActionPreference = "Stop"
$model = $env:VLLM_SMOKE_MODEL
if (-not $model) { $model = "facebook/opt-125m" }

$env:VLLM_TARGET_DEVICE = "cuda"
$env:TORCH_CUDA_ARCH_LIST = "7.0"

$argsList = @(
  "serve", $model,
  "--host", "127.0.0.1",
  "--port", "8000",
  "--max-model-len", "512",
  "--gpu-memory-utilization", "0.7",
  "--trust-remote-code"
)

$proc = Start-Process -FilePath "vllm" -ArgumentList $argsList -PassThru -NoNewWindow
try {
  $ok = $false
  foreach ($i in 1..60) {
    Start-Sleep -Seconds 5
    try {
      $r = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/completions" -ContentType "application/json" -Body (@{
        model = $model
        prompt = "Hello"
        max_tokens = 8
      } | ConvertTo-Json)
      Write-Host "SMOKE SERVE OK:" ($r | ConvertTo-Json -Compress)
      $ok = $true
      break
    } catch {
      Write-Host "waiting for server... ($i)"
    }
  }
  if (-not $ok) { throw "smoke serve timed out" }
}
finally {
  if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
```

- [ ] **Step 2: Run smoke**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\smoke_serve.ps1
```

Expected: `SMOKE SERVE OK` with a short completion. If `opt-125m` fails due to model-class quirks, set `VLLM_SMOKE_MODEL` to another tiny HF causal LM or a local folder the user provides — still single GPU, low max len.

- [ ] **Step 3: On runtime blockers only (FlashInfer / asyncio loop / etc.)**

Port the **smallest** SystemPanic Win32 runtime stub that unblocks single-GPU serve. Document the file+reason in `BUILD_LOG.md`. Do not enable TP/NCCL.

- [ ] **Step 4: Commit**

```powershell
git add scripts/windows/smoke_serve.ps1 docs/windows/BUILD_LOG.md
git add -u
git commit -m "test: Windows single-V100 serve smoke"
```

---

### Task 10: Windows README + publish prep

**Files:**
- Create: `README.windows.md`
- Modify: top of `README.md` with a short pointer to `README.windows.md` (do not delete Linux/V100 content)

**Interfaces:**
- Consumes: green smoke
- Produces: publishable docs; repo ready for `origin` push when user provides GitHub destination

- [ ] **Step 1: Write `README.windows.md`**

Must include:

1. Title: 1Cat-vLLM for Windows (Tesla V100 / SM70)
2. Hard pin: Python 3.12, CUDA **12.8**, Torch **2.10** — CUDA 13 unsupported (Volta removed)
3. Attribution: based on 1CatAI/1Cat-vLLM; Windows build adaptation informed by SystemPanic/vllm-windows
4. Build steps: VS `vcvarsall`, `scripts/windows/env_build.cmd`, pip installs, build flash-attn-v100 then vLLM
5. Smoke: `preflight_env.py`, `smoke_imports.py`, `smoke_serve.ps1`
6. Limits: no TP/NCCL in MVP; not a drop-in for non-V100 Windows users (use SystemPanic for Ampere+)

- [ ] **Step 2: Add a 5-line pointer at the top of `README.md`**

```markdown
> **Windows + Tesla V100:** see [README.windows.md](README.windows.md).
> Requires CUDA 12.8 (CUDA 13 dropped Volta/SM70).
```

- [ ] **Step 3: Update design status line**

In the design doc header, set status to: `Implemented per plan 2026-09-26-1cat-vllm-windows.md (MVP smoke green)` only after Task 9 passes.

- [ ] **Step 4: Commit**

```powershell
git add README.windows.md README.md docs/superpowers/specs/2026-09-26-1cat-vllm-windows-design.md
git commit -m "docs: Windows V100 build and smoke guide"
```

- [ ] **Step 5: Publish (manual gate — ask user for GitHub owner/name)**

```powershell
# After user provides OWNER/REPO:
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin HEAD
```

Do not force-push. Wheel upload is optional.

---

## Self-review vs spec

| Spec requirement | Task |
|---|---|
| Base = 1Cat; port Windows patches | 1, 3–7 |
| CUDA 12.8 / Torch 2.10 / no CUDA 13 | Global Constraints; Tasks 2, 4, 8, 10 |
| SM70 / FlashAttention-V100 | Tasks 7–8 |
| MVP smoke imports + single-GPU serve | Tasks 2, 9 |
| Publish after smoke | Task 10 |
| No TP/NCCL in MVP | Global Constraints; Task 9 Step 3 |
| Preserve design docs across bootstrap | Task 1 |

Placeholder scan: no TBD/TODO left for required steps; iterative compile unknowns are constrained to “minimal fix + BUILD_LOG entry.”

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-26-1cat-vllm-windows.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with checkpoints  

Which approach?
