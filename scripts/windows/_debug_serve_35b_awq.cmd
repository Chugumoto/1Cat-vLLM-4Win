@echo off
setlocal
set HF_HUB_DISABLE_XET=1
set VLLM_TARGET_DEVICE=cuda
set TORCH_CUDA_ARCH_LIST=7.0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set "LOG=%~dp0..\..\docs\windows\perf_results\serve_debug_35b_awq_cmd.log"
cd /d "%TEMP%"
echo LOG=%LOG%
C:\Python312\Scripts\vllm.exe serve QuantTrio/Qwen3.6-35B-A3B-AWQ --host 127.0.0.1 --port 8004 --dtype float16 --kv-cache-dtype fp8_e5m2 --max-model-len 8192 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 --max-num-seqs 1 --trust-remote-code --limit-mm-per-prompt "{\"image\":0,\"video\":0}" > "%LOG%" 2>&1
echo exit=%ERRORLEVEL%
type "%LOG%"
exit /b %ERRORLEVEL%
