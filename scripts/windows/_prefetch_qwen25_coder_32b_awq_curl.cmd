@echo off
setlocal EnableDelayedExpansion
REM Official AWQ (~19GB). Prefer curl if HF Xet stalls.
set "DEST=C:\Users\Chugumoto\.cache\huggingface\hub\models--Qwen--Qwen2.5-Coder-32B-Instruct-AWQ\manual"
mkdir "%DEST%" 2>nul
set "BASE=https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-AWQ/resolve/main"
set "LOG=C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win\docs\windows\perf_results\hf_download_qwen25_coder_32b_awq_curl.log"

echo DEST=%DEST% > "%LOG%"

for %%F in (
  .gitattributes LICENSE README.md config.json generation_config.json merges.txt
  model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json
) do (
  echo meta %%F >> "%LOG%"
  curl.exe --http1.1 -L --retry 5 --retry-delay 2 -C - -o "%DEST%\%%F" "%BASE%/%%F?download=true" >> "%LOG%" 2>&1
)

for %%F in (
  model-00001-of-00005.safetensors
  model-00002-of-00005.safetensors
  model-00003-of-00005.safetensors
  model-00004-of-00005.safetensors
  model-00005-of-00005.safetensors
) do (
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

echo DONE path=%DEST% >> "%LOG%"
echo DONE %DEST%
exit /b 0
