$ErrorActionPreference = "Stop"

$model = $env:VLLM_PERF_MODEL
if (-not $model) { $model = "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4" }

$port = 8003
if ($env:VLLM_PERF_PORT) { $port = [int]$env:VLLM_PERF_PORT }

$util = $env:VLLM_PERF_GPU_MEM_UTIL
if (-not $util) { $util = "0.90" }

$maxLen = $env:VLLM_PERF_MAX_MODEL_LEN
if (-not $maxLen) { $maxLen = "2048" }

$kv = $env:VLLM_PERF_KV_CACHE_DTYPE
if (-not $kv) { $kv = "fp8_e5m2" }

$outTokens = 256
if ($env:VLLM_PERF_MAX_TOKENS) { $outTokens = [int]$env:VLLM_PERF_MAX_TOKENS }

$dflash2Enabled = $false
$dflashToggle = $env:VLLM_PERF_ENABLE_DFLASH2
if ($dflashToggle -and @("1", "true", "yes") -contains $dflashToggle.ToLowerInvariant()) {
  $dflash2Enabled = $true
}

$floor = 21.0
if ($dflash2Enabled) { $floor = 0.0 }
if ($null -ne $env:VLLM_PERF_TOK_S_FLOOR) {
  if ($env:VLLM_PERF_TOK_S_FLOOR -eq "" -or $env:VLLM_PERF_TOK_S_FLOOR -eq "0") {
    $floor = 0.0
  } else {
    $floor = [double]$env:VLLM_PERF_TOK_S_FLOOR
  }
}

$prompt = "Write a short technical paragraph explaining what NVIDIA Tesla V100 is used for in machine learning inference."
if ($env:VLLM_PERF_PROMPT) { $prompt = $env:VLLM_PERF_PROMPT }

$env:VLLM_TARGET_DEVICE = "cuda"
$env:TORCH_CUDA_ARCH_LIST = "7.0"
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:HF_HUB_DISABLE_XET) { $env:HF_HUB_DISABLE_XET = "1" }

$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not (Test-Path (Join-Path $repoRoot "vllm"))) {
  $repoRoot = (Get-Location).Path
}
$resultsDir = Join-Path $repoRoot "docs\windows\perf_results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
  $freeLine = & nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
  if ($freeLine) {
    $freeMiB = [int](($freeLine | Select-Object -First 1).ToString().Trim())
    if ($freeMiB -lt 12000) {
      throw "V100 free memory ${freeMiB} MiB < 12000 MiB; kill leftover GPU processes (nvidia-smi) and retry"
    }
  }
}

$speculativeConfig = $null
if ($dflash2Enabled) {
  $speculativeConfig = '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","revision":"dedf8df68adfb1afeaf7b7480c0a0243108177b4","kv_cache_dtype":"auto"}'
}

$argsList = @(
  "serve", $model,
  "--host", "127.0.0.1",
  "--port", "$port",
  "--dtype", "float16",
  "--kv-cache-dtype", $kv,
  "--max-model-len", $maxLen,
  "--gpu-memory-utilization", $util,
  "--max-num-seqs", "1",
  "--attention-backend", "FLASH_ATTN_V100",
  "--trust-remote-code",
  "--limit-mm-per-prompt", '{"image":0,"video":0}'
)
if ($dflash2Enabled) {
  $argsList += "--speculative-config", $speculativeConfig
}

$vllm = Get-Command vllm -ErrorAction SilentlyContinue
if (-not $vllm) {
  $vllmCmd = "C:\Python312\Scripts\vllm.exe"
  if (-not (Test-Path $vllmCmd)) { throw "vllm executable not found" }
} else {
  $vllmCmd = $vllm.Source
}

function Format-ProcessArgument([string]$Value) {
  if ($null -eq $Value) { return '""' }
  if ($Value -notmatch '\s|"') { return $Value }
  '"' + ($Value.Replace('\', '\\').Replace('"', '\"')) + '"'
}

$workDir = $env:TEMP
$uri = "http://127.0.0.1:$port/v1/completions"
$argString = ($argsList | ForEach-Object { Format-ProcessArgument $_ }) -join ' '
Write-Host "Starting: $vllmCmd $argString"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $vllmCmd
$psi.WorkingDirectory = $workDir
$psi.Arguments = $argString
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
if (-not $proc.Start()) { throw "Failed to start vllm serve" }

function Invoke-Completion([int]$maxTokens, [switch]$IgnoreEos) {
  $body = @{
    model       = $model
    prompt      = $prompt
    max_tokens  = $maxTokens
    temperature = 0
  }
  if ($IgnoreEos) {
    $body.ignore_eos = $true
  }
  $json = $body | ConvertTo-Json
  return Invoke-RestMethod -Method Post -Uri $uri -ContentType "application/json" -Body $json -TimeoutSec 600
}

try {
  $ready = $false
  foreach ($i in 1..90) {
    Start-Sleep -Seconds 10
    if ($proc.HasExited) {
      throw "vllm serve exited early with code $($proc.ExitCode)"
    }
    try {
      $null = Invoke-Completion -maxTokens 8
      $ready = $true
      Write-Host "server ready ($i)"
      break
    } catch {
      Write-Host "waiting for server... ($i)"
    }
  }
  if (-not $ready) { throw "smoke perf timed out waiting for server" }

  Write-Host "warmup..."
  $null = Invoke-Completion -maxTokens 32

  Write-Host "timed decode max_tokens=$outTokens ..."
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $r = Invoke-Completion -maxTokens $outTokens -IgnoreEos
  $sw.Stop()
  $seconds = [Math]::Max($sw.Elapsed.TotalSeconds, 0.001)
  $completionTokens = 0
  if ($r.usage -and $r.usage.completion_tokens) {
    $completionTokens = [int]$r.usage.completion_tokens
  } else {
    $completionTokens = $outTokens
  }
  $tokPerSec = $completionTokens / $seconds

  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $outPath = Join-Path $resultsDir "perf_qwen38_nvfp4_$stamp.json"
  $payload = [ordered]@{
    timestamp              = (Get-Date).ToString("o")
    model                  = $model
    host                   = "127.0.0.1"
    port                   = $port
    dtype                  = "float16"
    kv_cache_dtype         = $kv
    max_model_len          = [int]$maxLen
    gpu_memory_utilization = [double]$util
    attention_backend      = "FLASH_ATTN_V100"
    dflash2_enabled        = $dflash2Enabled
    speculative_config     = $speculativeConfig
    max_tokens             = $outTokens
    completion_tokens      = $completionTokens
    wall_seconds           = [Math]::Round($seconds, 4)
    e2e_output_tok_s       = [Math]::Round($tokPerSec, 3)
    metric                 = "e2e_output_tok_s"
    floor_applied          = $floor
    sample_text            = $(if ($r.choices) { $r.choices[0].text } else { $null })
  }
  ($payload | ConvertTo-Json -Depth 5) | Set-Content -Path $outPath -Encoding utf8
  Write-Host "wrote $outPath"

  if ($floor -gt 0 -and $tokPerSec -lt $floor) {
    throw ("SMOKE PERF FAIL: e2e_output_tok_s={0:N3} < floor={1}" -f $tokPerSec, $floor)
  }

  Write-Host ("SMOKE PERF OK: e2e_output_tok_s={0:N3} completion_tokens={1} wall_s={2:N2}" -f $tokPerSec, $completionTokens, $seconds)
}
finally {
  if ($proc -and -not $proc.HasExited) {
    & taskkill.exe /F /T /PID $proc.Id 2>$null | Out-Null
    Start-Sleep -Seconds 2
  }
}
