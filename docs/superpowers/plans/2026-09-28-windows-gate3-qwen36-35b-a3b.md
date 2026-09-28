# Windows Gate 3 (Qwen3.6-35B-A3B-AWQ + MTP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Windows Gate 3 smoke path for `QuantTrio/Qwen3.6-35B-A3B-AWQ` **and** `nvidia/Qwen3.6-35B-A3B-NVFP4` on 1× V100: baseline tok/s + soft floors (**N35** / **N35nv**), then native MTP with `num_speculative_tokens=1` and (if green) `=2`, quality canary, and PERF_GATE summary rows.

**Architecture:** Mirror Gate 2 `smoke_perf_qwen38_nvfp4.*` (env toggle for speculative). Dedicated `.cmd`/`.ps1` per quant; ProcessStartInfo + `Format-ProcessArgument`; `taskkill /F /T` teardown; JSON under `docs/windows/perf_results/`. Quality matrix: AWQ port **8004**, NVFP4 port **8005**.

**Tech Stack:** PowerShell 5+, Python 3.12, CUDA 12.8 / Torch 2.10+cu128, installed Windows `vllm.exe`, HF Hub (`HF_HUB_DISABLE_XET=1`).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-28-windows-gate3-qwen36-35b-a3b-design.md`
- 1× V100 only: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`
- No TP / NCCL / track A; no DFlash2/external draft (native MTP only)
- Ports: **8004** (AWQ), **8005** (NVFP4 ModelOpt)
- Soft floors: **N35** / **N35nv** / MTP floors = `max(1, floor(0.7 × first green e2e_output_tok_s))`
- Commits only when the user explicitly asks
- Kill leftover GPU processes; fail if free VRAM &lt; 12000 MiB
- Prefer `HF_HUB_DISABLE_XET=1`; no `vcvars` for runtime
- Use `nvidia/Qwen3.6-35B-A3B-NVFP4` (safetensors ModelOpt); **not** GGUF `simhadrig/...-NVFP4-Q8_0`

---

## File map

| Path | Responsibility |
|---|---|
| `tests/windows/test_smoke_perf_gate3_contract.py` | Marker contracts |
| `scripts/windows/smoke_perf_qwen36_35b_a3b_awq.ps1` | AWQ baseline + MTP env |
| `scripts/windows/smoke_perf_qwen36_35b_a3b_awq.cmd` | AWQ CMD wrapper |
| `scripts/windows/smoke_perf_qwen36_35b_a3b_nvfp4.ps1` | NVFP4 baseline + MTP + FLASH_ATTN_V100 |
| `scripts/windows/smoke_perf_qwen36_35b_a3b_nvfp4.cmd` | NVFP4 CMD wrapper |
| `scripts/windows/smoke_quality_matrix.ps1` | 35B AWQ (8004) + 35B NVFP4 (8005) profiles |
| `docs/windows/PERF_GATE.md` | Gate 3 runbook + summary |
| Spec file | Status + floors after greens |

---

### Task 1: Contract tests (RED)

**Files:**
- Create: `tests/windows/test_smoke_perf_gate3_contract.py`

**Interfaces:**
- Consumes: nothing
- Produces: failing pytest until scripts/docs exist

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_smoke_perf_35b_ps1_pins_model_port_mtp():
    text = (ROOT / "scripts/windows/smoke_perf_qwen36_35b_a3b_awq.ps1").read_text(encoding="utf-8")
    assert "QuantTrio/Qwen3.6-35B-A3B-AWQ" in text
    assert "8004" in text
    assert "VLLM_PERF_ENABLE_MTP" in text
    assert '"method":"mtp"' in text.replace(" ", "") or "'method':'mtp'" in text.replace(" ", "") or 'method":"mtp"' in text
    assert "SMOKE PERF OK" in text
    assert "perf_qwen36_35b_a3b_awq" in text


def test_smoke_perf_35b_cmd_uses_system_powershell():
    text = (ROOT / "scripts/windows/smoke_perf_qwen36_35b_a3b_awq.cmd").read_text(encoding="utf-8")
    assert "WindowsPowerShell" in text
    assert "smoke_perf_qwen36_35b_a3b_awq.ps1" in text


def test_quality_matrix_has_35b_profile():
    text = (ROOT / "scripts/windows/smoke_quality_matrix.ps1").read_text(encoding="utf-8")
    assert "Qwen3.6-35B-A3B-AWQ" in text
    assert "8004" in text


def test_perf_gate_has_gate3_section():
    text = (ROOT / "docs/windows/PERF_GATE.md").read_text(encoding="utf-8")
    assert "Gate 3" in text
    assert "35B-A3B" in text
    assert "MTP" in text
```

- [ ] **Step 2: Run ? expect RED**

```bat
C:\Python312\python.exe -m pytest tests\windows\test_smoke_perf_gate3_contract.py -v --confcutdir=tests/windows
```

- [ ] **Step 3: Commit** (only if user asked)

---

### Task 2: CMD wrapper

**Files:**
- Create: `scripts/windows/smoke_perf_qwen36_35b_a3b_awq.cmd`

**Interfaces:**
- Same skeleton as `smoke_perf_qwen36_27b_awq.cmd` with `-File` ? `smoke_perf_qwen36_35b_a3b_awq.ps1` and `HF_HUB_DISABLE_XET=1` default

- [ ] **Step 1: Write wrapper** (verbatim Gate 2 CMD pattern; only script name differs)

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
if not defined HF_HUB_DISABLE_XET set "HF_HUB_DISABLE_XET=1"

set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" set "PS=%SystemRoot%\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" (
  echo ERROR: powershell.exe not found
  exit /b 1
)

"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0smoke_perf_qwen36_35b_a3b_awq.ps1"
exit /b %ERRORLEVEL%
```

- [ ] **Step 2: `dir` sanity-check**

---

### Task 3: PowerShell smoke script (baseline + MTP)

**Files:**
- Create: `scripts/windows/smoke_perf_qwen36_35b_a3b_awq.ps1`

**Interfaces:**
- Env: `VLLM_PERF_ENABLE_MTP` (`1`/`true`/`yes`), `VLLM_PERF_MTP_NUM_TOKENS` (default `1` when MTP on)
- Produces: JSON `perf_qwen36_35b_a3b_awq_*.json` with `mtp_enabled`, `mtp_num_tokens`, `speculative_config`

- [ ] **Step 1: Implement by copying `smoke_perf_qwen38_nvfp4.ps1` and adapting**

Defaults:

| Setting | Value |
|---|---|
| Model | `QuantTrio/Qwen3.6-35B-A3B-AWQ` |
| Port | `8004` |
| max-model-len | `8192` |
| kv | `fp8_e5m2` |
| util | `0.90` |
| Floor default | `0` until Task 5 sets **N35** (empty/`0` disables) |
| MTP | if enabled: `--speculative-config` JSON `{"method":"mtp","num_speculative_tokens":N}` where N from env (default 1) |
| Teardown | `taskkill /F /T /PID` |
| Timed | `ignore_eos` |
| Do **not** force `--attention-backend FLASH_ATTN_V100` unless baseline fails without it (MoE path may differ from NVFP4); start without it, add only if needed and document |

Keep ProcessStartInfo + Format-ProcessArgument + VRAM preflight + CUDA pin + HF_HUB_DISABLE_XET.

- [ ] **Step 2: Partial pytest** for ps1+cmd contract tests ? expect PASS for those two

---

### Task 4: Quality profile + PERF_GATE Gate 3 docs

**Files:**
- Modify: `scripts/windows/smoke_quality_matrix.ps1` (add profile for `QuantTrio/Qwen3.6-35B-A3B-AWQ`: port 8004, max_model_len 8192, same float16/fp8/util flags as smoke)
- Modify: `docs/windows/PERF_GATE.md` (append Gate 3)
- Modify: `README.windows.md` (one line mentioning Gate 3)

- [ ] **Step 1: Extend quality matrix `Get-ModelProfile`** so the 35B id maps to port 8004 / len 8192; allow `VLLM_QUALITY_INCLUDE_35B=1` or include via `VLLM_QUALITY_MODELS`.

- [ ] **Step 2: Append to PERF_GATE.md**

```markdown
## Gate 3 ? Qwen3.6-35B-A3B-AWQ (+ MTP)

- Model: `QuantTrio/Qwen3.6-35B-A3B-AWQ`
- Script: `scripts/windows/smoke_perf_qwen36_35b_a3b_awq.cmd`
- Port: 8004
- Soft floor **N35**: _TBD after baseline green_
- MTP-1 / MTP-2 floors: _TBD_

### Baseline

```bat
set HF_HUB_DISABLE_XET=1
set VLLM_PERF_TOK_S_FLOOR=0
scripts\windows\smoke_perf_qwen36_35b_a3b_awq.cmd
```

### MTP-1

```bat
set VLLM_PERF_ENABLE_MTP=1
set VLLM_PERF_MTP_NUM_TOKENS=1
set VLLM_PERF_TOK_S_FLOOR=0
scripts\windows\smoke_perf_qwen36_35b_a3b_awq.cmd
```

### MTP-2 (only if MTP-1 green)

```bat
set VLLM_PERF_ENABLE_MTP=1
set VLLM_PERF_MTP_NUM_TOKENS=2
scripts\windows\smoke_perf_qwen36_35b_a3b_awq.cmd
```

Add summary rows for baseline / MTP-1 / MTP-2 (TBD until measured).
```

- [ ] **Step 3: Full Gate 3 contracts ? expect all PASS**

```bat
C:\Python312\python.exe -m pytest tests\windows\test_smoke_perf_gate3_contract.py -v --confcutdir=tests/windows
```

---

### Task 5: First green baseline + set N35

**Files:**
- Modify: `PERF_GATE.md`, design spec status
- Create: JSON artifact

- [ ] **Step 1: Free V100; prefetch**

```bat
set HF_HUB_DISABLE_XET=1
C:\Python312\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('QuantTrio/Qwen3.6-35B-A3B-AWQ'))"
```

- [ ] **Step 2: Baseline run**

```bat
set VLLM_PERF_ENABLE_MTP=
set VLLM_PERF_TOK_S_FLOOR=0
scripts\windows\smoke_perf_qwen36_35b_a3b_awq.cmd
```

On OOM: `set VLLM_PERF_MAX_MODEL_LEN=4096` and retry; document.

- [ ] **Step 3: N35 = max(1, floor(0.7×measured)); update docs; set script default floor to N35; re-run with floor**

```bat
set VLLM_PERF_TOK_S_FLOOR=<N35>
scripts\windows\smoke_perf_qwen36_35b_a3b_awq.cmd
```

---

### Task 6: MTP-1 (then optional MTP-2)

**Files:**
- Modify: `PERF_GATE.md` summary + floors
- Modify: design spec

- [ ] **Step 1: MTP-1**

```bat
set VLLM_PERF_ENABLE_MTP=1
set VLLM_PERF_MTP_NUM_TOKENS=1
set VLLM_PERF_TOK_S_FLOOR=0
scripts\windows\smoke_perf_qwen36_35b_a3b_awq.cmd
```

If OK: **Nmtp1**; update docs; optional floor re-run.  
If FAIL: document reason; **skip MTP-2**.

- [ ] **Step 2: MTP-2 only if Step 1 green**

```bat
set VLLM_PERF_ENABLE_MTP=1
set VLLM_PERF_MTP_NUM_TOKENS=2
set VLLM_PERF_TOK_S_FLOOR=0
scripts\windows\smoke_perf_qwen36_35b_a3b_awq.cmd
```

If OK: **Nmtp2**. If FAIL: document FAIL row.

---

### Task 7: Quality canary + finalize

**Files:**
- Modify: `PERF_GATE.md` Quality column
- Create: quality JSON if run produces one

- [ ] **Step 1: Run quality including 35B**

```bat
set VLLM_QUALITY_MODELS=QuantTrio/Qwen3.6-35B-A3B-AWQ
set HF_HUB_DISABLE_XET=1
scripts\windows\smoke_quality_matrix.cmd
```

(Or include alongside prior models if time allows.)

- [ ] **Step 2: Finalize summary table; update design Status to implemented with floors/FAIL notes**

- [ ] **Step 3: Re-run Gate 3 contracts ? PASS**

- [ ] **Step 4: Commit** (only if user asked)

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| Baseline + N35 | 3, 5 |
| MTP-1 then MTP-2 if green | 3, 6 |
| Quality canary | 4, 7 |
| PERF_GATE + contracts | 1, 4, 7 |
| Port 8004, 1× V100, no TP/DFlash2 | Global + 2?3 |
| Structured FAIL OK for MTP | 6 |

Placeholder scan: floors TBD until Tasks 5?6; no unimplemented ?later? steps without commands.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-28-windows-gate3-qwen36-35b-a3b.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** ? fresh subagent per task  
2. **Inline Execution** ? this session with checkpoints  

Which approach?

### Task 8: MTP probe on prior Gate models (27B-AWQ, optional 9B)

**Files:**
- Modify: `scripts/windows/smoke_perf_qwen36_27b_awq.ps1` (add same MTP env as 35B: `VLLM_PERF_ENABLE_MTP`, `VLLM_PERF_MTP_NUM_TOKENS`)
- Optionally modify: `scripts/windows/smoke_perf_qwen35_awq.ps1` (same toggle)
- Modify: `docs/windows/PERF_GATE.md` (MTP rows for 27B / 9B)

**Why:** Dense 27B no-MTP measured ~29 tok/s; Reddit ~40?55 tok/s on V100 is typically with MTP.

- [ ] **Step 1: Add MTP toggle to 27B script** (copy pattern from 35B Task 3: speculative-config `{"method":"mtp","num_speculative_tokens":N}`)
- [ ] **Step 2: Run 27B MTP-1 then MTP-2 if green; record tok/s vs baseline 29.328**
- [ ] **Step 3 (optional): same for 9B-AWQ vs baseline 62**
- [ ] **Step 4: Update PERF_GATE summary; do not invent floors unless green**
- [ ] **Step 5: Commit only if user asked**

