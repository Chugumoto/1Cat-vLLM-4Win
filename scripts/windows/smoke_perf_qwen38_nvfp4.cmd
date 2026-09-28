@echo off
setlocal
cd /d "%~dp0..\.."

if not defined CUDA_DEVICE_ORDER set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=0"
if not defined PYTHONUTF8 set "PYTHONUTF8=1"
if not defined PYTHONIOENCODING set "PYTHONIOENCODING=utf-8"
if not defined VLLM_TARGET_DEVICE set "VLLM_TARGET_DEVICE=cuda"
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=7.0"
if not defined HF_HUB_DISABLE_XET set "HF_HUB_DISABLE_XET=1"

set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" set "PS=%SystemRoot%\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" (
  echo ERROR: powershell.exe not found
  exit /b 1
)

"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0smoke_perf_qwen38_nvfp4.ps1"
exit /b %ERRORLEVEL%
