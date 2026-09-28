@echo off
REM Activate MSVC 14.44 (VS2022 toolset) + CUDA 12.8 + V100 device 0.
REM Usage: call scripts\windows\dev_env.cmd
REM Prefer a FRESH cmd.exe — repeated vcvarsall can blow PATH ("input line is too long").

set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
if not exist "%VCVARS%" (
  echo ERROR: vcvarsall.bat not found: %VCVARS%
  exit /b 1
)

REM Prefer CUDA 12.8 MSVC-compatible toolset (14.44), not VS2026 default 14.51.
call "%VCVARS%" x64 -vcvars_ver=14.44
if errorlevel 1 (
  echo ERROR: vcvarsall failed for -vcvars_ver=14.44
  echo If you see "input line is too long", open a NEW cmd.exe and retry.
  exit /b 1
)

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=%CUDA_PATH%"
set "CUDA_PATH_V12_8=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
set "INCLUDE=%CUDA_PATH%\include;%INCLUDE%"
set "LIB=%CUDA_PATH%\lib\x64;%LIB%"

set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0

set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
set TORCH_CUDA_ARCH_LIST=7.0
set CMAKE_CUDA_ARCHITECTURES=70
set MAX_JOBS=8
set VLLM_DISABLE_FA3_BUILD=1
set NVCC_PREPEND_FLAGS=-allow-unsupported-compiler
set TORCH_NVCC_FLAGS=-allow-unsupported-compiler

where cl >nul 2>&1
if errorlevel 1 (
  echo ERROR: cl.exe not on PATH after vcvarsall
  exit /b 1
)
where nvcc >nul 2>&1
if errorlevel 1 (
  echo ERROR: nvcc not on PATH after CUDA 12.8 prepend
  exit /b 1
)

echo === DEV ENV OK ===
where cl
cl 2>&1 | findstr /i "Version"
where nvcc
nvcc --version | findstr /i "release"
echo CUDA_PATH=%CUDA_PATH%
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
exit /b 0
