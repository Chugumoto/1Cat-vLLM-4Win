# Windows Perf Gate 1 (Qwen3.5-9B-AWQ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repeatable Windows smoke-perf gate that serves `QuantTrio/Qwen3.5-9B-AWQ` on 1× Tesla V100-32GB, measures steady decode tok/s, writes a JSON artifact, and documents a soft floor **N** after the first green run.

**Architecture:** Mirror MVP `scripts/windows/smoke_serve.ps1` / `.cmd`: start `vllm.exe serve` from `%TEMP%` (avoid source-tree shadowing), wait for HTTP readiness on port 8001, warmup, timed completion, tear down. Docs under `docs/windows/PERF_GATE.md`; results under `docs/windows/perf_results/`.

**Tech Stack:** PowerShell 5+, Python 3.12, existing Windows MVP install (`C:\Python312`), CUDA 12.8 / Torch 2.10+cu128, Hugging Face model `QuantTrio/Qwen3.5-9B-AWQ`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-27-windows-perf-gate-qwen35-awq-design.md`
- 1× V100 only: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`
- No TP / NCCL / track A in this plan
- No CUDA 13; keep Torch 2.10 + cu128
- Port **8001** (MVP smoke uses 8000)
- Do not invent git `user.name` / `user.email`; only commit when the user explicitly asks
- Kill leftover GPU processes before long runs; fail early if V100 free memory is too low

---

## File map

| Path | Responsibility |
|---|---|
| `tests/windows/test_smoke_perf_qwen35_contract.py` | String/existence contract for scripts + PERF_GATE |
| `scripts/windows/smoke_perf_qwen35_awq.ps1` | Serve + warmup + timed decode + JSON + OK/FAIL |
| `scripts/windows/smoke_perf_qwen35_awq.cmd` | CMD wrapper with full PowerShell path |
| `docs/windows/PERF_GATE.md` | How to run, contract, **N** (filled after first green) |
| `docs/windows/perf_results/.gitkeep` | Results directory (JSON ignored or committed selectively) |
| `README.windows.md` | One-paragraph pointer to PERF_GATE |
| `docs/superpowers/specs/2026-09-27-windows-perf-gate-qwen35-awq-design.md` | Status + **N** after first green |

---

### Task 1: Contract tests for perf-gate scaffolding

**Files:**
- Create: `tests/windows/test_smoke_perf_qwen35_contract.py`

**Interfaces:**
- Consumes: nothing
- Produces: pytest file that fails until scripts/docs exist with required markers

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_smoke_perf_ps1_exists_and_pins_model_port():
    text = (ROOT / "scripts/windows/smoke_perf_qwen35_awq.ps1").read_text(encoding="utf-8")
    assert "QuantTrio/Qwen3.5-9B-AWQ" in text
    assert "8001" in text
    assert "SMOKE PERF OK" in text
    assert "CUDA_DEVICE_ORDER" in text
    assert "perf_results" in text or "PERF_RESULT" in text


def test_smoke_perf_cmd_wrapper_uses_system_powershell():
    text = (ROOT / "scripts/windows/smoke_perf_qwen35_awq.cmd").read_text(encoding="utf-8")
    assert "WindowsPowerShell" in text
    assert "smoke_perf_qwen35_awq.ps1" in text


def test_perf_gate_doc_exists():
    text = (ROOT / "docs/windows/PERF_GATE.md").read_text(encoding="utf-8")
    assert "Qwen3.5-9B-AWQ" in text
    assert "V100" in text
    assert "tok/s" in text or "tok/s" in text.lower() or "throughput" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bat
C:\Python312\python.exe -m pytest tests\windows\test_smoke_perf_qwen35_contract.py -v
```

Expected: FAIL with file-not-found / assertion errors for missing scripts.

- [ ] **Step 3: Commit** (only if user asked to commit)

```bat
git add tests\windows\test_smoke_perf_qwen35_contract.py
git commit -m "test: add Windows Qwen3.5-AWQ perf-gate contract"
```

---

### Task 2: CMD wrapper

**Files:**
- Create: `scripts/windows/smoke_perf_qwen35_awq.cmd`

**Interfaces:**
- Consumes: `smoke_perf_qwen35_awq.ps1` (created in Task 3; wrapper may land first)
- Produces: double-click / cmd entry that finds System32 PowerShell

- [ ] **Step 1: Write the wrapper**

```bat
@echo off
setlocal
cd /d "%~dp0..\.."

if not defined CUDA_DEVICE_ORDER set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=0"
if not defined PYTHONUTF8 set "PYTHONUTF8=1"
if not defined PYTHONIOENCODING set "PYTHONIOENCODING=utf-8"
if not defined VLLM_TARGET_DEVICE set "VLLM_TARGET_DEVICE=cuda"
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=7.0"

set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" set "PS=%SystemRoot%\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" (
  echo ERROR: powershell.exe not found
  exit /b 1
)

"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0smoke_perf_qwen35_awq.ps1"
exit /b %ERRORLEVEL%
```

- [ ] **Step 2: Sanity-check the file exists**

Run: `dir scripts\windows\smoke_perf_qwen35_awq.cmd`  
Expected: file listed.

---

### Task 3: PowerShell smoke-perf script

**Files:**
- Create: `scripts/windows/smoke_perf_qwen35_awq.ps1`
- Create: `docs/windows/perf_results/.gitkeep`

**Interfaces:**
- Consumes: installed `vllm.exe` (MVP), HF model id / override env
- Produces: console `SMOKE PERF OK` / throw; JSON under `docs/windows/perf_results/`

- [ ] **Step 1: Ensure results directory**

Create `docs/windows/perf_results/.gitkeep` (empty file).

- [ ] **Step 2: Implement `smoke_perf_qwen35_awq.ps1`**

Full script (implement as one file):

```powershell
$ErrorActionPreference = "Stop"

$model = $env:VLLM_PERF_MODEL
if (-not $model) { $model = "QuantTrio/Qwen3.5-9B-AWQ" }

$port = 8001
if ($env:VLLM_PERF_PORT) { $port = [int]$env:VLLM_PERF_PORT }

$util = $env:VLLM_PERF_GPU_MEM_UTIL
if (-not $util) { $util = "0.90" }

$maxLen = $env:VLLM_PERF_MAX_MODEL_LEN
if (-not $maxLen) { $maxLen = "8192" }

$kv = $env:VLLM_PERF_KV_CACHE_DTYPE
if (-not $kv) { $kv = "fp8_e5m2" }

$outTokens = 256
if ($env:VLLM_PERF_MAX_TOKENS) { $outTokens = [int]$env:VLLM_PERF_MAX_TOKENS }

$floor = 0.0
if ($env:VLLM_PERF_TOK_S_FLOOR) { $floor = [double]$env:VLLM_PERF_TOK_S_FLOOR }

$prompt = "Write a short technical paragraph explaining what NVIDIA Tesla V100 is used for in machine learning inference."
if ($env:VLLM_PERF_PROMPT) { $prompt = $env:VLLM_PERF_PROMPT }

$env:VLLM_TARGET_DEVICE = "cuda"
$env:TORCH_CUDA_ARCH_LIST = "7.0"
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
if (-not $env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES = "0" }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not (Test-Path (Join-Path $repoRoot "vllm"))) {
  $repoRoot = (Get-Location).Path
}
$resultsDir = Join-Path $repoRoot "docs\windows\perf_results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

# Fail early if V100 looks occupied (optional soft check via nvidia-smi)
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

$argsList = @(
  "serve", $model,
  "--host", "127.0.0.1",
  "--port", "$port",
  "--dtype", "float16",
  "--kv-cache-dtype", $kv,
  "--max-model-len", $maxLen,
  "--gpu-memory-utilization", $util,
  "--max-num-seqs", "1",
  "--trust-remote-code",
  "--limit-mm-per-prompt", '{"image":0,"video":0}'
)

$vllm = Get-Command vllm -ErrorAction SilentlyContinue
if (-not $vllm) {
  $vllmCmd = "C:\Python312\Scripts\vllm.exe"
  if (-not (Test-Path $vllmCmd)) { throw "vllm executable not found" }
} else {
  $vllmCmd = $vllm.Source
}

$workDir = $env:TEMP
$uri = "http://127.0.0.1:$port/v1/completions"
Write-Host "Starting: $vllmCmd $($argsList -join ' ')"
$proc = Start-Process -FilePath $vllmCmd -ArgumentList $argsList -WorkingDirectory $workDir -PassThru -NoNewWindow

function Invoke-Completion([int]$maxTokens) {
  $body = @{
    model       = $model
    prompt      = $prompt
    max_tokens  = $maxTokens
    temperature = 0
  } | ConvertTo-Json
  return Invoke-RestMethod -Method Post -Uri $uri -ContentType "application/json" -Body $body -TimeoutSec 600
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
  $r = Invoke-Completion -maxTokens $outTokens
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
  $outPath = Join-Path $resultsDir "perf_qwen35_awq_$stamp.json"
  $payload = [ordered]@{
    timestamp           = (Get-Date).ToString("o")
    model               = $model
    host                = "127.0.0.1"
    port                = $port
    dtype               = "float16"
    kv_cache_dtype      = $kv
    max_model_len       = [int]$maxLen
    gpu_memory_utilization = [double]$util
    max_tokens          = $outTokens
    completion_tokens   = $completionTokens
    wall_seconds        = [Math]::Round($seconds, 4)
    e2e_output_tok_s    = [Math]::Round($tokPerSec, 3)
    metric              = "e2e_output_tok_s"
    floor_applied       = $floor
    sample_text         = $(if ($r.choices) { $r.choices[0].text } else { $null })
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
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
  }
}
```

- [ ] **Step 3: Re-run contract tests**

```bat
C:\Python312\python.exe -m pytest tests\windows\test_smoke_perf_qwen35_contract.py -v
```

Expected: still FAIL on `PERF_GATE.md` until Task 4; ps1/cmd assertions PASS.

---

### Task 4: PERF_GATE.md + README pointer

**Files:**
- Create: `docs/windows/PERF_GATE.md`
- Modify: `README.windows.md` (append short section)

**Interfaces:**
- Consumes: script paths from Tasks 2–3
- Produces: operator docs; **N** placeholder until Task 5

- [ ] **Step 1: Write `docs/windows/PERF_GATE.md`**

```markdown
# Windows performance gate (1× V100)

## Gate 1 — Qwen3.5-9B-AWQ

- Model: `QuantTrio/Qwen3.5-9B-AWQ`
- Hardware: 1× Tesla V100 via `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`
- Script: `scripts/windows/smoke_perf_qwen35_awq.cmd`
- Metric: `e2e_output_tok_s` (completion_tokens / wall seconds after warmup)
- Soft floor **N**: _TBD after first green run_ (set `VLLM_PERF_TOK_S_FLOOR`)

### Run

```bat
call scripts\windows\env_build.cmd
scripts\windows\smoke_perf_qwen35_awq.cmd
```

Optional env overrides: `VLLM_PERF_MODEL`, `VLLM_PERF_PORT` (default 8001), `VLLM_PERF_MAX_MODEL_LEN`, `VLLM_PERF_KV_CACHE_DTYPE`, `VLLM_PERF_GPU_MEM_UTIL`, `VLLM_PERF_MAX_TOKENS`, `VLLM_PERF_TOK_S_FLOOR`.

Results JSON: `docs/windows/perf_results/perf_qwen35_awq_*.json`.

### Before you run

1. `nvidia-smi` — free V100 memory (script fails if free &lt; 12 GiB).
2. Do not run from a cmd session whose PATH was wiped by `run_build_mvp` without PowerShell (wrapper uses System32 path).
3. First download of the HF model can take a long time.

### Out of scope

TP/NCCL, Linux 4× NVFP4/DFlash2 parity, Habr 2×16GB agent setup.
```

- [ ] **Step 2: Add pointer to `README.windows.md`**

Append:

```markdown
## Performance gate (1× V100)

See [docs/windows/PERF_GATE.md](docs/windows/PERF_GATE.md) for the Qwen3.5-9B-AWQ smoke-perf gate (`scripts/windows/smoke_perf_qwen35_awq.cmd`).
```

- [ ] **Step 3: Run contract tests — expect PASS**

```bat
C:\Python312\python.exe -m pytest tests\windows\test_smoke_perf_qwen35_contract.py -v
```

Expected: all PASS.

---

### Task 5: First green run on the V100 machine + set N

**Files:**
- Modify: `docs/windows/PERF_GATE.md` (replace TBD **N**)
- Modify: `docs/superpowers/specs/2026-09-27-windows-perf-gate-qwen35-awq-design.md` (status + **N**)
- Create: one JSON under `docs/windows/perf_results/` (from the run)

**Interfaces:**
- Consumes: working MVP install + scripts from Tasks 2–4
- Produces: measured `e2e_output_tok_s`; **N = max(1, floor(0.7 × measured))**

- [ ] **Step 1: Free V100 if needed**

```bat
nvidia-smi
REM taskkill /F /PID <pid> if a stale python holds tens of GiB
```

- [ ] **Step 2: Run gate with floor disabled (default 0)**

```bat
cd C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win
call scripts\windows\env_build.cmd
scripts\windows\smoke_perf_qwen35_awq.cmd
```

Expected: `SMOKE PERF OK: e2e_output_tok_s=...` and a new JSON file.  
If OOM: rerun with `set VLLM_PERF_MAX_MODEL_LEN=4096` and record the override in PERF_GATE.  
If `fp8_e5m2` fails: try `set VLLM_PERF_KV_CACHE_DTYPE=auto` (or omit via script edit) and document.

- [ ] **Step 3: Compute and write N**

Example: measured 42.5 → N = floor(0.7×42.5) = 29.

Update `PERF_GATE.md`:

```markdown
- Soft floor **N**: **29** tok/s (`VLLM_PERF_TOK_S_FLOOR=29`)
```

Update design spec Status line to: `Gate 1 implemented; N=<value> from first green run <timestamp>`.

- [ ] **Step 4: Re-run with floor**

```bat
set VLLM_PERF_TOK_S_FLOOR=29
scripts\windows\smoke_perf_qwen35_awq.cmd
```

Expected: `SMOKE PERF OK` again.

- [ ] **Step 5: Commit** (only if user asked)

```bat
git add scripts\windows\smoke_perf_qwen35_awq.ps1 scripts\windows\smoke_perf_qwen35_awq.cmd tests\windows\test_smoke_perf_qwen35_contract.py docs\windows\PERF_GATE.md docs\windows\perf_results README.windows.md docs\superpowers\specs\2026-09-27-windows-perf-gate-qwen35-awq-design.md
git commit -m "feat(windows): Qwen3.5-9B-AWQ perf gate on 1x V100"
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| Serve `QuantTrio/Qwen3.5-9B-AWQ` on 1× V100 | 3, 5 |
| Scripted warmup + decode + JSON | 3 |
| Soft floor N after first green | 5 |
| `PERF_GATE.md` | 4, 5 |
| Port 8001 / PCI_BUS_ID / TEMP cwd | 2, 3 |
| README pointer | 4 |
| Contract tests | 1, 4 |
| Non-goals TP/NVFP4/Habr | documented in PERF_GATE + Global Constraints |

Placeholder scan: no TBD left in executable steps; **N** is intentionally filled only in Task 5 after measurement.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-27-windows-perf-gate-qwen35-awq.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with checkpoints  

Which approach?
