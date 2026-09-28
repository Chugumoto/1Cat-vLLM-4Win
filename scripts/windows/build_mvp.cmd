@echo off
setlocal
cd /d "%~dp0..\.."

call "%~dp0dev_env.cmd"
if errorlevel 1 exit /b 1

echo === PREFLIGHT ===
python scripts\windows\preflight_env.py
if errorlevel 1 exit /b 1

echo === BUILD flash-attention-v100 ===
python -c "import flash_attn_v100" >nul 2>&1
if errorlevel 1 (
  pushd flash-attention-v100
  set TORCH_CUDA_ARCH_LIST=7.0
  python -m pip install . --no-build-isolation -vvv
  if errorlevel 1 (
    popd
    echo ERROR: flash-attention-v100 build failed
    exit /b 1
  )
  popd
) else (
  echo flash_attn_v100 already installed — skip rebuild
)

REM setuptools-scm + CMake FetchContent/patch need git and patch on PATH.
where git >nul 2>&1
if errorlevel 1 (
  echo ERROR: git.exe not on PATH ^(needed for FetchContent and version^)
  exit /b 1
)
where patch >nul 2>&1
if errorlevel 1 (
  echo ERROR: patch.exe not on PATH ^(needed for vllm_flash_attn SM70 patches^)
  exit /b 1
)
if not defined VLLM_VERSION_OVERRIDE set "VLLM_VERSION_OVERRIDE=1.5.1.dev0+windows"

echo === BUILD 1cat-vllm ===
python -m pip install . --no-build-isolation -vvv
if errorlevel 1 (
  echo ERROR: 1cat-vllm build failed
  exit /b 1
)

echo === SMOKE IMPORTS ===
python scripts\windows\smoke_imports.py
if errorlevel 1 exit /b 1

echo === BUILD MVP OK ===
exit /b 0
