@echo off
REM Interactive OpenAI-compatible server: nvidia/Qwen3.6-35B-A3B-NVFP4 on 1x V100.
REM Prefetch weights first (Hub/Xet often stalls):
REM   scripts\windows\_prefetch_35b_nvfp4_curl.cmd
REM Client model id must match --served-model-name below: qwen36-35b-nvfp4
REM Start from %%TEMP%% so Windows can load vllm\_C.pyd (repo cwd can lock the DLL).

cd /d %TEMP%
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set HF_HUB_DISABLE_XET=1
set VLLM_SM70_GDN_DECODE_FLASHQLA=0
C:\Python312\python.exe -m vllm.entrypoints.cli.main serve ^
  "%USERPROFILE%\.cache\huggingface\hub\models--nvidia--Qwen3.6-35B-A3B-NVFP4\manual" ^
  --served-model-name qwen36-35b-nvfp4 ^
  --host 127.0.0.1 --port 8005 ^
  --dtype float16 --kv-cache-dtype fp8_e5m2 ^
  --max-model-len 262144 ^
  --max-num-batched-tokens 8192 ^
  --gpu-memory-utilization 0.95 ^
  --max-num-seqs 1 ^
  --attention-backend FLASH_ATTN_V100 ^
  --gdn-prefill-backend triton ^
  --trust-remote-code --enable-auto-tool-choice --tool-call-parser qwen3_xml ^
  --limit-mm-per-prompt "{\"image\":0,\"video\":0}"

pause
