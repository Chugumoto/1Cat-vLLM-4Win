@echo off
setlocal
REM nvidia/...-NVFP4 uses Xet storage; do not set HF_HUB_DISABLE_XET here.
set HF_HUB_DISABLE_XET=
cd /d "%~dp0..\.."
set "LOG=docs\windows\perf_results\hf_download_qwen36_35b_a3b_nvfp4.log"
echo LOG=%LOG%
C:\Python312\python.exe -u scripts\windows\_prefetch_35b_nvfp4.py > "%LOG%" 2>&1
set EC=%ERRORLEVEL%
echo exit=%EC%
type "%LOG%"
exit /b %EC%
