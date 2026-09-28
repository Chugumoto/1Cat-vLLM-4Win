@echo off
setlocal EnableDelayedExpansion
REM Community compressed-tensors NVFP4 (~20.7GB). Official nvidia/ Coder-32B NVFP4 not found.
set "DEST=%USERPROFILE%\.cache\huggingface\hub\models--drawais--Qwen2.5-Coder-32B-Instruct-NVFP4\manual"
mkdir "%DEST%" 2>nul
set "BASE=https://huggingface.co/drawais/Qwen2.5-Coder-32B-Instruct-NVFP4/resolve/main"
set "LOG=%~dp0..\..\docs\windows\perf_results\hf_download_qwen25_coder_32b_nvfp4_curl.log"

echo DEST=%DEST% > "%LOG%"

for %%F in (
  .gitattributes LICENSE NOTICE README.md chat_template.jinja config.json generation_config.json
  recipe.yaml tokenizer.json tokenizer_config.json
) do (
  echo meta %%F >> "%LOG%"
  curl.exe --http1.1 -L --retry 5 --retry-delay 2 -C - -o "%DEST%\%%F" "%BASE%/%%F?download=true" >> "%LOG%" 2>&1
)

echo ==== model.safetensors ==== >> "%LOG%"
echo Downloading model.safetensors (~20.7GB) ...
curl.exe --http1.1 -L --retry 5 --retry-delay 3 -C - -o "%DEST%\model.safetensors" "%BASE%/model.safetensors?download=true" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo FAIL model.safetensors >> "%LOG%"
  echo FAIL model.safetensors
  exit /b 1
)

echo DONE path=%DEST% >> "%LOG%"
echo DONE %DEST%
exit /b 0
