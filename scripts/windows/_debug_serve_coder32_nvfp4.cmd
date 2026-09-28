@echo off
setlocal
set HF_HUB_DISABLE_XET=1
set VLLM_TARGET_DEVICE=cuda
set TORCH_CUDA_ARCH_LIST=7.0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "MODEL=%USERPROFILE%\.cache\huggingface\hub\models--drawais--Qwen2.5-Coder-32B-Instruct-NVFP4\manual"
set "LOG=%~dp0..\..\docs\windows\perf_results\serve_debug_coder32_nvfp4.log"
cd /d "%TEMP%"
echo LOG=%LOG%
C:\Python312\python.exe -m vllm.entrypoints.cli.main serve "%MODEL%" --host 127.0.0.1 --port 8006 --dtype float16 --kv-cache-dtype fp8_e5m2 --max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 1 --attention-backend FLASH_ATTN_V100 --enforce-eager --trust-remote-code --limit-mm-per-prompt "{\"image\":0,\"video\":0}" > "%LOG%" 2>&1
echo exit=%ERRORLEVEL%
type "%LOG%"
exit /b %ERRORLEVEL%
