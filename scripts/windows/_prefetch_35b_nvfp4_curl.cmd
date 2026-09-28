@echo off
setlocal EnableDelayedExpansion
REM Download nvidia/Qwen3.6-35B-A3B-NVFP4 safetensor shards via curl (Xet hub API hangs).
set "DEST=C:\Users\Chugumoto\.cache\huggingface\hub\models--nvidia--Qwen3.6-35B-A3B-NVFP4\manual"
mkdir "%DEST%" 2>nul
set "BASE=https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/resolve/main"
set "LOG=C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win\docs\windows\perf_results\hf_download_qwen36_35b_a3b_nvfp4_curl.log"

echo DEST=%DEST% > "%LOG%"
for %%F in (model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors) do (
  echo ==== %%F ==== >> "%LOG%"
  echo Downloading %%F ...
  curl.exe --http1.1 -L --retry 5 --retry-delay 3 -C - -o "%DEST%\%%F" "%BASE%/%%F?download=true" >> "%LOG%" 2>&1
  if errorlevel 1 (
    echo FAIL %%F >> "%LOG%"
    echo FAIL %%F
    exit /b 1
  )
  echo OK %%F >> "%LOG%"
)

REM copy small metadata from existing hub snapshot if present
set "SNAP=C:\Users\Chugumoto\.cache\huggingface\hub\models--nvidia--Qwen3.6-35B-A3B-NVFP4\snapshots\1355db6a052410cfd62085d94b58866fd0f2c3c5"
if exist "%SNAP%\config.json" (
  for %%F in (config.json configuration.json generation_config.json hf_quant_config.json model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json chat_template.jinja preprocessor_config.json video_preprocessor_config.json README.md .gitattributes .quant_summary.txt) do (
    if exist "%SNAP%\%%F" copy /Y "%SNAP%\%%F" "%DEST%\%%F" >nul
  )
)

echo DONE path=%DEST% >> "%LOG%"
echo DONE %DEST%
exit /b 0
