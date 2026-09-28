$ErrorActionPreference = "Stop"

$defaultModels = @(
  "QuantTrio/Qwen3.5-9B-AWQ",
  "QuantTrio/Qwen3.6-27B-AWQ"
)
$nvfp4Model = "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
$model35b = "QuantTrio/Qwen3.6-35B-A3B-AWQ"
$model35bNvfp4 = "nvidia/Qwen3.6-35B-A3B-NVFP4"

if ($env:VLLM_QUALITY_MODELS) {
  $models = @(
    $env:VLLM_QUALITY_MODELS.Split(",") |
      ForEach-Object { $_.Trim() } |
      Where-Object { $_ }
  )
} else {
  $models = $defaultModels
}

if ($env:VLLM_QUALITY_INCLUDE_NVFP4 -eq "1" -and $models -notcontains $nvfp4Model) {
  $models += $nvfp4Model
}
if ($env:VLLM_QUALITY_INCLUDE_35B -eq "1" -and $models -notcontains $model35b) {
  $models += $model35b
}
if ($env:VLLM_QUALITY_INCLUDE_35B_NVFP4 -eq "1" -and $models -notcontains $model35bNvfp4) {
  $models += $model35bNvfp4
}
if ($models.Count -eq 0) {
  throw "VLLM_QUALITY_MODELS did not contain any model ids"
}

$prompts = @(
  "Say hello in one short sentence.",
  "What is 2+2? Answer with a single number.",
  "Назови столицу Франции одним словом."
)

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

$vllm = Get-Command vllm -ErrorAction SilentlyContinue
if (-not $vllm) {
  $vllmCmd = "C:\Python312\Scripts\vllm.exe"
  if (-not (Test-Path $vllmCmd)) { throw "vllm executable not found" }
} else {
  $vllmCmd = $vllm.Source
}

function Test-Enabled([string]$Value) {
  return $Value -and @("1", "true", "yes") -contains $Value.ToLowerInvariant()
}

function Get-ModelProfile([string]$Model) {
  switch -Exact ($Model) {
    "QuantTrio/Qwen3.5-9B-AWQ" {
      return [ordered]@{ Port = 8001; MaxModelLen = 8192; MaxNumBatchedTokens = $null; AttentionBackend = $null }
    }
    "QuantTrio/Qwen3.6-27B-AWQ" {
      return [ordered]@{ Port = 8002; MaxModelLen = 4096; MaxNumBatchedTokens = $null; AttentionBackend = $null }
    }
    $nvfp4Model {
      return [ordered]@{ Port = 8003; MaxModelLen = 2048; MaxNumBatchedTokens = $null; AttentionBackend = "FLASH_ATTN_V100" }
    }
    $model35b {
      return [ordered]@{ Port = 8004; MaxModelLen = 8192; MaxNumBatchedTokens = 8192; AttentionBackend = $null }
    }
    $model35bNvfp4 {
      return [ordered]@{ Port = 8005; MaxModelLen = 4096; MaxNumBatchedTokens = 4096; AttentionBackend = "FLASH_ATTN_V100" }
    }
    default {
      throw "No quality serve profile for model '$Model'"
    }
  }
}

function Format-ProcessArgument([string]$Value) {
  if ($null -eq $Value) { return '""' }
  if ($Value -notmatch '\s|"') { return $Value }
  return '"' + ($Value.Replace('\', '\\').Replace('"', '\"')) + '"'
}

function Test-RepeatedCharacter([string]$Text) {
  $compact = $Text -replace '\s', ''
  if ($compact.Length -lt 2) { return $false }

  $counts = @{}
  foreach ($character in $compact.ToCharArray()) {
    $key = [string]$character
    if ($counts.ContainsKey($key)) {
      $counts[$key] += 1
    } else {
      $counts[$key] = 1
    }
  }
  $maxCount = ($counts.Values | Measure-Object -Maximum).Maximum
  return ($maxCount / $compact.Length) -ge 0.8
}

function Invoke-QualityCompletion(
  [string]$Uri,
  [string]$Model,
  [string]$Prompt,
  [int]$MaxTokens
) {
  $body = @{
    model       = $Model
    prompt      = $Prompt
    max_tokens  = $MaxTokens
    temperature = 0
  } | ConvertTo-Json
  # PS 5.1 Invoke-RestMethod defaults to wrong charset for non-ASCII prompts (HTTP 400).
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
  return Invoke-RestMethod `
    -Method Post `
    -Uri $Uri `
    -ContentType "application/json; charset=utf-8" `
    -Body $bytes `
    -TimeoutSec 600
}

$allPassed = $true
$modelResults = @()

foreach ($model in $models) {
  $profile = $null
  $proc = $null
  $promptResults = @()
  $modelPassed = $false
  $modelError = $null

  try {
    $profile = Get-ModelProfile $model
    $argsList = @(
      "serve", $model,
      "--host", "127.0.0.1",
      "--port", "$($profile.Port)",
      "--dtype", "float16",
      "--kv-cache-dtype", "fp8_e5m2",
      "--max-model-len", "$($profile.MaxModelLen)",
      "--gpu-memory-utilization", "0.90",
      "--max-num-seqs", "1",
      "--trust-remote-code",
      "--limit-mm-per-prompt", '{"image":0,"video":0}'
    )
    if ($profile.MaxNumBatchedTokens) {
      $argsList += "--max-num-batched-tokens", "$($profile.MaxNumBatchedTokens)"
    }
    if ($profile.AttentionBackend) {
      $argsList += "--attention-backend", $profile.AttentionBackend
    }
    if ($model -eq $nvfp4Model -and (Test-Enabled $env:VLLM_PERF_ENABLE_DFLASH2)) {
      $speculativeConfig = '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","revision":"dedf8df68adfb1afeaf7b7480c0a0243108177b4","kv_cache_dtype":"auto"}'
      $argsList += "--speculative-config", $speculativeConfig
    }

    $argString = ($argsList | ForEach-Object { Format-ProcessArgument $_ }) -join " "
    Write-Host "Starting quality serve: $vllmCmd $argString"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $vllmCmd
    $psi.WorkingDirectory = $env:TEMP
    $psi.Arguments = $argString
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    if (-not $proc.Start()) { throw "Failed to start vllm serve" }

    $uri = "http://127.0.0.1:$($profile.Port)/v1/completions"
    $ready = $false
    foreach ($i in 1..90) {
      Start-Sleep -Seconds 10
      if ($proc.HasExited) {
        throw "vllm serve exited early with code $($proc.ExitCode)"
      }
      try {
        $null = Invoke-QualityCompletion $uri $model $prompts[0] 1
        $ready = $true
        Write-Host "server ready ($i)"
        break
      } catch {
        Write-Host "waiting for server... ($i)"
      }
    }
    if (-not $ready) { throw "quality smoke timed out waiting for server" }

    $modelPassed = $true
    foreach ($prompt in $prompts) {
      $response = Invoke-QualityCompletion $uri $model $prompt 64
      $text = if ($response.choices) { [string]$response.choices[0].text } else { "" }
      $failure = $null
      if ([string]::IsNullOrWhiteSpace($text)) {
        $failure = "empty completion text"
      } elseif (Test-RepeatedCharacter $text) {
        $failure = "one character occupies at least 80 percent of non-whitespace output"
      }
      $passed = $null -eq $failure
      if (-not $passed) { $modelPassed = $false }
      $promptResults += [pscustomobject][ordered]@{
        prompt = $prompt
        text   = $text
        passed = $passed
        failure = $failure
      }
    }
  } catch {
    $modelError = $_.Exception.Message
    $modelPassed = $false
  } finally {
    if ($proc -and -not $proc.HasExited) {
      & taskkill.exe /F /T /PID $proc.Id 2>$null | Out-Null
      Start-Sleep -Seconds 2
    }
  }

  if (-not $modelPassed) { $allPassed = $false }
  $modelResults += [pscustomobject][ordered]@{
    model         = $model
    port          = $(if ($profile) { $profile.Port } else { $null })
    max_model_len = $(if ($profile) { $profile.MaxModelLen } else { $null })
    passed        = $modelPassed
    error         = $modelError
    prompts       = $promptResults
  }
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outPath = Join-Path $resultsDir "quality_matrix_$stamp.json"
$payload = [ordered]@{
  timestamp  = (Get-Date).ToString("o")
  all_passed = $allPassed
  models     = $modelResults
}
($payload | ConvertTo-Json -Depth 7) | Set-Content -Path $outPath -Encoding utf8
Write-Host "wrote $outPath"

if (-not $allPassed) {
  throw "SMOKE QUALITY FAIL: one or more models failed the quality matrix"
}
Write-Host "SMOKE QUALITY OK"
