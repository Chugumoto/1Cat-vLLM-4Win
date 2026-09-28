$ErrorActionPreference = "Stop"

$model = "QuantTrio/Qwen3.6-35B-A3B-AWQ"
$port = 8004
$maxLen = "8192"
if ($env:VLLM_PERF_MAX_MODEL_LEN) { $maxLen = $env:VLLM_PERF_MAX_MODEL_LEN }

$env:HF_HUB_DISABLE_XET = "1"
$env:VLLM_TARGET_DEVICE = "cuda"
$env:TORCH_CUDA_ARCH_LIST = "7.0"
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$repoRoot = "C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $repoRoot "docs\windows\perf_results\serve_debug_35b_awq_psi_$stamp.log"

$mmLimit = '{"image":0,"video":0}'
$argsList = @(
  "serve", $model,
  "--host", "127.0.0.1",
  "--port", "$port",
  "--dtype", "float16",
  "--kv-cache-dtype", "fp8_e5m2",
  "--max-model-len", $maxLen,
  "--gpu-memory-utilization", "0.90",
  "--max-num-seqs", "1",
  "--trust-remote-code",
  "--limit-mm-per-prompt", $mmLimit
)

function Format-ProcessArgument([string]$Value) {
  if ($null -eq $Value) { return '""' }
  if ($Value -notmatch '\s|"') { return $Value }
  '"' + ($Value.Replace('\', '\\').Replace('"', '\"')) + '"'
}

$argString = ($argsList | ForEach-Object { Format-ProcessArgument $_ }) -join ' '
$vllmCmd = "C:\Python312\Scripts\vllm.exe"
Write-Host "ARGS: $argString"
Write-Host "LOG: $log"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $vllmCmd
$psi.WorkingDirectory = $env:TEMP
$psi.Arguments = $argString
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$handler = {
  if ($EventArgs.Data) {
    Add-Content -Path $Event.MessageData -Value $EventArgs.Data
  }
}
$outEvent = Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived -Action $handler -MessageData $log
$errEvent = Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived -Action $handler -MessageData $log

if (-not $proc.Start()) { throw "Failed to start vllm" }
$proc.BeginOutputReadLine()
$proc.BeginErrorReadLine()
Write-Host "started pid=$($proc.Id)"

while (-not $proc.HasExited) {
  Start-Sleep -Seconds 5
  $secs = [int]((Get-Date) - $proc.StartTime).TotalSeconds
  Write-Host "alive ${secs}s"
  if (Test-Path $log) {
    Get-Content $log -Tail 3 | ForEach-Object { Write-Host "  $_" }
  }
}

Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
Unregister-Event -SourceIdentifier $errEvent.Name -ErrorAction SilentlyContinue
Write-Host "exit=$($proc.ExitCode)"
if (Test-Path $log) {
  Write-Host "==== last 100 log lines ===="
  Get-Content $log -Tail 100
}
exit $proc.ExitCode
