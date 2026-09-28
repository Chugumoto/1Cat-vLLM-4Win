# Windows Gate 2 (27B-AWQ + NVFP4 ± DFlash2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Windows Gate 2 smoke scripts and docs: bring up `QuantTrio/Qwen3.6-27B-AWQ`, attempt `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` (target-only then optional DFlash2), record soft floors, and publish a speed/quality summary for models that ran (including Gate 1 9B).

**Architecture:** Mirror Gate 1 `smoke_perf_qwen35_awq.*`: CMD → System32 PowerShell → `vllm serve` from `%TEMP%`, HTTP warmup + timed decode (`ignore_eos`), JSON under `docs/windows/perf_results/`. Separate scripts per phase; NVFP4 script toggles DFlash2 via env. Document structured fails (OOM / missing path) in `PERF_GATE.md` without inventing TP.

**Tech Stack:** PowerShell 5+, Python 3.12, CUDA 12.8 / Torch 2.10+cu128, installed Windows MVP `vllm.exe`, HF Hub (`HF_HUB_DISABLE_XET=1`).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-27-windows-gate2-qwen36-27b-nvfp4-design.md`
- 1× V100 only: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0` (no TP across CMP/RTX)
- No TP / NCCL / track A
- No CUDA 13; Torch 2.10 + cu128
- Ports: **8002** (27B-AWQ), **8003** (NVFP4 ± DFlash2); Gate 1 keeps 8001
- Do not invent git `user.name` / `user.email`; only commit when the user explicitly asks
- Kill leftover GPU processes before long runs; fail early if free VRAM &lt; 12000 MiB
- Prefer `HF_HUB_DISABLE_XET=1`; do not require `env_build.cmd` / vcvars for runtime smokes
- Soft floors: **N27** / **Nnv** / **Ndflash** = `max(1, floor(0.7 × first green e2e_output_tok_s))` after each green phase
- Linux 4× TP4 ~260 tok/s and Habr 2×16GB are reference only, not pass bars

---

## File map

| Path | Responsibility |
|---|---|
| `tests/windows/test_smoke_perf_gate2_contract.py` | String/existence contracts for Gate 2 scripts + PERF_GATE sections |
| `scripts/windows/smoke_perf_qwen36_27b_awq.ps1` | Phase 1 serve + measure |
| `scripts/windows/smoke_perf_qwen36_27b_awq.cmd` | CMD wrapper |
| `scripts/windows/smoke_perf_qwen38_nvfp4.ps1` | Phase 2a/2b NVFP4 (± DFlash2 via env) |
| `scripts/windows/smoke_perf_qwen38_nvfp4.cmd` | CMD wrapper |
| `scripts/windows/smoke_quality_matrix.ps1` | Phase 3 canaries against already-known models (or re-serve sequentially) |
| `scripts/windows/smoke_quality_matrix.cmd` | CMD wrapper |
| `docs/windows/PERF_GATE.md` | Gate 2 runbooks + speed summary table + floors |
| `docs/superpowers/specs/2026-09-27-windows-gate2-qwen36-27b-nvfp4-design.md` | Status + floors after greens |
| `README.windows.md` | One-line pointer that Gate 2 lives in PERF_GATE |

---

### Task 1: Contract tests for Gate 2 scaffolding

**Files:**
- Create: `tests/windows/test_smoke_perf_gate2_contract.py`

**Interfaces:**
- Consumes: nothing
- Produces: pytest file that fails until scripts/docs contain required markers

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_smoke_perf_27b_awq_ps1_pins_model_port():
    text = (ROOT / "scripts/windows/smoke_perf_qwen36_27b_awq.ps1").read_text(encoding="utf-8")
    assert "QuantTrio/Qwen3.6-27B-AWQ" in text
    assert "8002" in text
    assert "SMOKE PERF OK" in text
    assert "CUDA_DEVICE_ORDER" in text
    assert "perf_qwen36_27b_awq" in text or "perf_results" in text


def test_smoke_perf_27b_cmd_uses_system_powershell():
    text = (ROOT / "scripts/windows/smoke_perf_qwen36_27b_awq.cmd").read_text(encoding="utf-8")
    assert "WindowsPowerShell" in text
    assert "smoke_perf_qwen36_27b_awq.ps1" in text


def test_smoke_perf_nvfp4_ps1_pins_models_and_dflash_toggle():
    text = (ROOT / "scripts/windows/smoke_perf_qwen38_nvfp4.ps1").read_text(encoding="utf-8")
    assert "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4" in text
    assert "8003" in text
    assert "VLLM_PERF_ENABLE_DFLASH2" in text
    assert "incoai/Qwen3.8-27B-DFlash2" in text
    assert "SMOKE PERF OK" in text


def test_smoke_perf_nvfp4_cmd_uses_system_powershell():
    text = (ROOT / "scripts/windows/smoke_perf_qwen38_nvfp4.cmd").read_text(encoding="utf-8")
    assert "WindowsPowerShell" in text
    assert "smoke_perf_qwen38_nvfp4.ps1" in text


def test_quality_matrix_script_exists():
    text = (ROOT / "scripts/windows/smoke_quality_matrix.ps1").read_text(encoding="utf-8")
    assert "SMOKE QUALITY" in text
    assert "Qwen3.5-9B-AWQ" in text or "QuantTrio/Qwen3.5-9B-AWQ" in text


def test_perf_gate_has_gate2_section():
    text = (ROOT / "docs/windows/PERF_GATE.md").read_text(encoding="utf-8")
    assert "Gate 2" in text
    assert "Qwen3.6-27B-AWQ" in text
    assert "QUASAR" in text or "NVFP4" in text
    assert "Speed summary" in text or "speed summary" in text.lower()
```

- [ ] **Step 2: Run test — expect RED**

```bat
C:\Python312\python.exe -m pytest tests\windows\test_smoke_perf_gate2_contract.py -v --confcutdir=tests/windows
```

Expected: FAIL (missing files / markers).

- [ ] **Step 3: Commit** (only if user asked)

```bat
git add tests\windows\test_smoke_perf_gate2_contract.py
git commit -m "test: add Windows Gate 2 perf contract"
```

---

### Task 2: CMD wrappers (27B-AWQ + NVFP4)

**Files:**
- Create: `scripts/windows/smoke_perf_qwen36_27b_awq.cmd`
- Create: `scripts/windows/smoke_perf_qwen38_nvfp4.cmd`
- Create: `scripts/windows/smoke_quality_matrix.cmd`

**Interfaces:**
- Consumes: sibling `.ps1` files (may land in later tasks)
- Produces: System32 PowerShell launchers matching Gate 1 `.cmd` pattern

- [ ] **Step 1: Write all three wrappers** (same skeleton; only `-File` name differs)

`smoke_perf_qwen36_27b_awq.cmd`:

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

"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0smoke_perf_qwen36_27b_awq.ps1"
exit /b %ERRORLEVEL%
```

`smoke_perf_qwen38_nvfp4.cmd`: identical except `-File "%~dp0smoke_perf_qwen38_nvfp4.ps1"`.

`smoke_quality_matrix.cmd`: identical except `-File "%~dp0smoke_quality_matrix.ps1"`.

- [ ] **Step 2: Sanity-check**

```bat
dir scripts\windows\smoke_perf_qwen36_27b_awq.cmd scripts\windows\smoke_perf_qwen38_nvfp4.cmd scripts\windows\smoke_quality_matrix.cmd
```

Expected: three files listed.

---

### Task 3: PowerShell Phase 1 — Qwen3.6-27B-AWQ

**Files:**
- Create: `scripts/windows/smoke_perf_qwen36_27b_awq.ps1`

**Interfaces:**
- Consumes: installed `vllm.exe`, HF model
- Produces: `SMOKE PERF OK/FAIL`, JSON `docs/windows/perf_results/perf_qwen36_27b_awq_*.json`

- [ ] **Step 1: Implement `.ps1` by copying Gate 1 script and changing defaults**

Start from `scripts/windows/smoke_perf_qwen35_awq.ps1` and apply these defaults (keep ProcessStartInfo + Format-ProcessArgument + ignore_eos timed path):

| Setting | Value |
|---|---|
| Default model | `QuantTrio/Qwen3.6-27B-AWQ` |
| Default port | `8002` |
| Default max-model-len | `4096` |
| Default kv-cache-dtype | `fp8_e5m2` |
| Default floor | `0` until Task 6 sets **N27** (env `VLLM_PERF_TOK_S_FLOOR`; empty/`0` disables) |
| JSON prefix | `perf_qwen36_27b_awq_` |
| Success line | `SMOKE PERF OK: ...` |

Also set `$env:HF_HUB_DISABLE_XET = "1"` if unset. Keep free-VRAM &lt; 12000 throw. Keep mm limit JSON quoting via Format-ProcessArgument.

- [ ] **Step 2: Partial contract check**

```bat
C:\Python312\python.exe -m pytest tests\windows\test_smoke_perf_gate2_contract.py::test_smoke_perf_27b_awq_ps1_pins_model_port tests\windows\test_smoke_perf_gate2_contract.py::test_smoke_perf_27b_cmd_uses_system_powershell -v --confcutdir=tests/windows
```

Expected: those two PASS; others still FAIL.

---

### Task 4: PowerShell Phase 2 — NVFP4 (± DFlash2)

**Files:**
- Create: `scripts/windows/smoke_perf_qwen38_nvfp4.ps1`

**Interfaces:**
- Consumes: `vllm.exe`; env `VLLM_PERF_ENABLE_DFLASH2=1` enables draft
- Produces: JSON `perf_qwen38_nvfp4_*.json` with `dflash2_enabled` bool; OK or throw with clear reason

- [ ] **Step 1: Implement script**

Defaults:

| Setting | Value |
|---|---|
| Model | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| Port | `8003` |
| max-model-len | `2048` |
| gpu-memory-utilization | `0.90` |
| attention-backend | `FLASH_ATTN_V100` (include flag if this fork accepts it on Windows; if serve rejects, remove and log in PERF_GATE) |
| kv-cache-dtype | `fp8_e5m2` |
| TP | **omit** (implicit 1) |
| DFlash2 | if `$env:VLLM_PERF_ENABLE_DFLASH2` in `1`,`true`,`yes`: add `--speculative-config` JSON string: `{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","revision":"dedf8df68adfb1afeaf7b7480c0a0243108177b4","kv_cache_dtype":"auto"}` (same shape as README example, without TP4) |

Reuse Gate 1 ProcessStartInfo / Format-ProcessArgument / warmup / timed ignore_eos / floor logic. JSON must include:

```text
dflash2_enabled = <bool>
speculative_config = <string or null>
```

Default floor `0` until greens set **Nnv** / **Ndflash**.

On early serve exit, throw including exit code (structured fail is OK for Gate 2).

- [ ] **Step 2: Contract check for NVFP4 tests**

```bat
C:\Python312\python.exe -m pytest tests\windows\test_smoke_perf_gate2_contract.py::test_smoke_perf_nvfp4_ps1_pins_models_and_dflash_toggle tests\windows\test_smoke_perf_gate2_contract.py::test_smoke_perf_nvfp4_cmd_uses_system_powershell -v --confcutdir=tests/windows
```

Expected: PASS.

---

### Task 5: Quality matrix script + PERF_GATE Gate 2 docs

**Files:**
- Create: `scripts/windows/smoke_quality_matrix.ps1`
- Modify: `docs/windows/PERF_GATE.md`
- Modify: `README.windows.md` (one sentence under Performance gate)

**Interfaces:**
- Consumes: models list env `VLLM_QUALITY_MODELS` (comma-separated HF ids) OR defaults to Gate1 9B + 27B-AWQ (NVFP4 included only if `VLLM_QUALITY_INCLUDE_NVFP4=1`)
- Produces: console `SMOKE QUALITY OK` / fail; JSON `docs/windows/perf_results/quality_matrix_*.json`

- [ ] **Step 1: Implement `smoke_quality_matrix.ps1`**

Behavior (keep YAGNI):

1. Parse model list (default: `QuantTrio/Qwen3.5-9B-AWQ,QuantTrio/Qwen3.6-27B-AWQ`).
2. For each model, start serve with the **same flag profile as that model's smoke script** (9B → port 8001 / len 8192; 27B-AWQ → 8002 / 4096; NVFP4 → 8003 / 2048; no DFlash2 in quality loop unless env says so).
3. For each of 3 prompts, POST `/v1/completions` with `temperature=0`, `max_tokens=64`:
   - `Say hello in one short sentence.`
   - `What is 2+2? Answer with a single number.`
   - `Назови столицу Франции одним словом.`
4. Canary fail if empty text OR (after stripping whitespace) a single character repeated for ≥80% of length.
5. Tear down serve between models.
6. Write aggregate JSON; print `SMOKE QUALITY OK` if all canaries passed.

- [ ] **Step 2: Append Gate 2 sections to `PERF_GATE.md`**

Add after Gate 1:

```markdown
## Gate 2 — 27B-AWQ + NVFP4 (± DFlash2)

### Phase 1 — Qwen3.6-27B-AWQ

- Model: `QuantTrio/Qwen3.6-27B-AWQ`
- Script: `scripts/windows/smoke_perf_qwen36_27b_awq.cmd`
- Port: 8002
- Soft floor **N27**: _TBD after first green_
- Start: `max-model-len=4096` (fallback 2048 on OOM)

```bat
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set HF_HUB_DISABLE_XET=1
scripts\windows\smoke_perf_qwen36_27b_awq.cmd
```

### Phase 2a — Qwen3.8-27B-NVFP4 (target only)

- Model: `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`
- Script: `scripts/windows/smoke_perf_qwen38_nvfp4.cmd`
- Port: 8003
- Soft floor **Nnv**: _TBD after first green_ (or document FAIL)
- Note: Linux headlines use 4× TP4; this gate tries **1×32GB** only.

```bat
set HF_HUB_DISABLE_XET=1
scripts\windows\smoke_perf_qwen38_nvfp4.cmd
```

### Phase 2b — + DFlash2 (only if 2a green)

```bat
set VLLM_PERF_ENABLE_DFLASH2=1
scripts\windows\smoke_perf_qwen38_nvfp4.cmd
```

Draft: `incoai/Qwen3.8-27B-DFlash2`. Soft floor **Ndflash**: _TBD_.

### Quality matrix

```bat
scripts\windows\smoke_quality_matrix.cmd
```

### Speed summary

| Model | tok/s | Floor | Quality | Notes |
|---|---|---|---|---|
| Qwen3.5-9B-AWQ | 62.059 | N=43 | _TBD_ | Gate 1 |
| Qwen3.6-27B-AWQ | _TBD_ | N27=_TBD_ | _TBD_ | |
| Qwen3.8-27B-NVFP4 | _TBD_ | Nnv=_TBD_ | _TBD_ | 1×32GB attempt |
| Qwen3.8-27B-NVFP4+DFlash2 | _TBD_ | Ndflash=_TBD_ | _TBD_ | only if 2a green |
```
```

- [ ] **Step 3: README.windows.md** — extend Performance gate paragraph to mention Gate 2 in `PERF_GATE.md`.

- [ ] **Step 4: Full contract tests**

```bat
C:\Python312\python.exe -m pytest tests\windows\test_smoke_perf_gate2_contract.py -v --confcutdir=tests/windows
```

Expected: all PASS.

---

### Task 6: First green — Qwen3.6-27B-AWQ + set N27

**Files:**
- Modify: `docs/windows/PERF_GATE.md` (N27 + summary row)
- Modify: `docs/superpowers/specs/2026-09-27-windows-gate2-qwen36-27b-nvfp4-design.md` (status)
- Create: JSON under `docs/windows/perf_results/`

**Interfaces:**
- Consumes: Task 3 script + free V100
- Produces: measured tok/s; **N27**

- [ ] **Step 1: Free V100**

```bat
nvidia-smi
REM taskkill leftover python/vllm on GPU0 if needed
```

- [ ] **Step 2: Prefetch if needed**

```bat
set HF_HUB_DISABLE_XET=1
C:\Python312\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('QuantTrio/Qwen3.6-27B-AWQ'))"
```

Log optionally to `docs/windows/perf_results/hf_download_qwen36_27b_*.log`.

- [ ] **Step 3: Run Phase 1 (floor disabled)**

```bat
cd C:\Users\Chugumoto\Projects\1Cat-vLLM-4Win
set HF_HUB_DISABLE_XET=1
set VLLM_PERF_TOK_S_FLOOR=0
scripts\windows\smoke_perf_qwen36_27b_awq.cmd
```

Expected: `SMOKE PERF OK` or OOM → retry `set VLLM_PERF_MAX_MODEL_LEN=2048` and document.

- [ ] **Step 4: Set N27 = max(1, floor(0.7 × measured)); update PERF_GATE + spec status; re-run with floor**

```bat
set VLLM_PERF_TOK_S_FLOOR=<N27>
scripts\windows\smoke_perf_qwen36_27b_awq.cmd
```

Expected: `SMOKE PERF OK` again.

---

### Task 7: NVFP4 target-only, then optional DFlash2

**Files:**
- Modify: `docs/windows/PERF_GATE.md` summary rows
- Modify: design spec status
- Create: JSON artifact(s)

- [ ] **Step 1: Prefetch QUASAR NVFP4 (large)**

```bat
set HF_HUB_DISABLE_XET=1
C:\Python312\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4'))"
```

- [ ] **Step 2: Phase 2a (no DFlash2)**

```bat
set VLLM_PERF_ENABLE_DFLASH2=
set VLLM_PERF_TOK_S_FLOOR=0
scripts\windows\smoke_perf_qwen38_nvfp4.cmd
```

If green: set **Nnv**, update docs, optional floor re-run.  
If fail: write FAIL row in Speed summary with reason (OOM / missing kernel / exit code); **skip Task 7 Step 3**.

- [ ] **Step 3: Phase 2b (only if 2a green)**

```bat
set HF_HUB_DISABLE_XET=1
C:\Python312\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('incoai/Qwen3.8-27B-DFlash2'))"
set VLLM_PERF_ENABLE_DFLASH2=1
set VLLM_PERF_TOK_S_FLOOR=0
scripts\windows\smoke_perf_qwen38_nvfp4.cmd
```

If green: set **Ndflash**. If fail: document FAIL for DFlash2 row; keep 2a numbers.

---

### Task 8: Quality matrix run + finalize summary

**Files:**
- Modify: `docs/windows/PERF_GATE.md` (fill Quality column + any remaining TBD in summary)
- Create: `docs/windows/perf_results/quality_matrix_*.json`

- [ ] **Step 1: Run quality matrix for models that served**

```bat
REM Example after 27B-AWQ green; add NVFP4 only if 2a green:
set VLLM_QUALITY_MODELS=QuantTrio/Qwen3.5-9B-AWQ,QuantTrio/Qwen3.6-27B-AWQ
REM if NVFP4 green: set VLLM_QUALITY_INCLUDE_NVFP4=1  (script must honor this)
scripts\windows\smoke_quality_matrix.cmd
```

Expected: `SMOKE QUALITY OK` or listed canary failures (fix heuristics only if false positives are clear).

- [ ] **Step 2: Update PERF_GATE speed/quality table with real numbers; set design spec Status to implemented with floors/fail notes**

- [ ] **Step 3: Re-run Gate 2 contract tests — expect PASS**

```bat
C:\Python312\python.exe -m pytest tests\windows\test_smoke_perf_gate2_contract.py -v --confcutdir=tests/windows
```

- [ ] **Step 4: Commit** (only if user asked)

```bat
git add scripts\windows\smoke_perf_qwen36_27b_awq.* scripts\windows\smoke_perf_qwen38_nvfp4.* scripts\windows\smoke_quality_matrix.* tests\windows\test_smoke_perf_gate2_contract.py docs\windows\PERF_GATE.md docs\windows\perf_results README.windows.md docs\superpowers\specs\2026-09-27-windows-gate2-qwen36-27b-nvfp4-design.md
git commit -m "feat(windows): Gate 2 27B-AWQ and NVFP4 smoke path"
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| Phase 1 27B-AWQ smoke + N27 | 3, 6 |
| Phase 2a NVFP4 target-only | 4, 7 |
| Phase 2b DFlash2 if 2a green | 4, 7 |
| Speed summary three(+ ) models | 5, 6, 7, 8 |
| Minimal quality canary | 5, 8 |
| Ports 8002/8003, 1× V100, no TP | Global + 2–4 |
| HF_HUB_DISABLE_XET / no vcvars runtime | Global + 2, 6 |
| Contract tests | 1, 5, 8 |
| Structured fail OK for NVFP4 | 4, 7 |

Placeholder scan: floor TBD cells are intentionally filled only after measurement tasks; no “implement later” steps without commands.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-27-windows-gate2-qwen36-27b-nvfp4.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute in this session with checkpoints  

Which approach?
