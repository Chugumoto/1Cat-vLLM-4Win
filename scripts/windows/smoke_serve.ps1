$ErrorActionPreference = "Stop"
$model = $env:VLLM_SMOKE_MODEL
if (-not $model) { $model = "facebook/opt-125m" }

$env:VLLM_TARGET_DEVICE = "cuda"
$env:TORCH_CUDA_ARCH_LIST = "7.0"
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
if (-not $env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES = "0" }
# Avoid cp125x UnicodeEncodeError on the vLLM banner / logs.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$util = $env:VLLM_SMOKE_GPU_MEM_UTIL
if (-not $util) { $util = "0.3" }

$argsList = @(
  "serve", $model,
  "--host", "127.0.0.1",
  "--port", "8000",
  "--max-model-len", "512",
  "--gpu-memory-utilization", $util,
  "--trust-remote-code"
)

$vllm = Get-Command vllm -ErrorAction SilentlyContinue
if (-not $vllm) {
  $vllmCmd = "C:\Python312\Scripts\vllm.exe"
  if (-not (Test-Path $vllmCmd)) {
    throw "vllm executable not found on PATH or at $vllmCmd"
  }
} else {
  $vllmCmd = $vllm.Source
}

# Avoid project-root cwd shadowing site-packages\vllm (source tree has no .pyd).
$workDir = $env:TEMP
Write-Host "Starting: $vllmCmd $($argsList -join ' ')"
$proc = Start-Process -FilePath $vllmCmd -ArgumentList $argsList -WorkingDirectory $workDir -PassThru -NoNewWindow
try {
  $ok = $false
  foreach ($i in 1..60) {
    Start-Sleep -Seconds 5
    if ($proc.HasExited) {
      throw "vllm serve exited early with code $($proc.ExitCode)"
    }
    try {
      $r = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/completions" -ContentType "application/json" -Body (@{
        model = $model
        prompt = "Hello"
        max_tokens = 8
      } | ConvertTo-Json)
      Write-Host "SMOKE SERVE OK:" ($r | ConvertTo-Json -Compress)
      $ok = $true
      break
    } catch {
      Write-Host "waiting for server... ($i)"
    }
  }
  if (-not $ok) { throw "smoke serve timed out" }
}
finally {
  if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
