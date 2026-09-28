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
REM Without PCI_BUS_ID, CUDA may pick CMP/RTX as device 0 instead of the V100.
set CUDA_DEVICE_ORDER=PCI_BUS_ID
if not defined CUDA_VISIBLE_DEVICES set CUDA_VISIBLE_DEVICES=0

echo Build env ready for SM70 / CUDA 12.8
