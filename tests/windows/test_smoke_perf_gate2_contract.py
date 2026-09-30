from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_smoke_perf_27b_awq_ps1_pins_model_port():
    text = _read("scripts/windows/smoke_perf_qwen36_27b_awq.ps1")
    assert "QuantTrio/Qwen3.6-27B-AWQ" in text
    assert "8002" in text
    assert "SMOKE PERF OK" in text
    assert "CUDA_DEVICE_ORDER" in text
    assert "perf_qwen36_27b_awq" in text or "perf_results" in text


def test_smoke_perf_27b_cmd_uses_system_powershell():
    text = _read("scripts/windows/smoke_perf_qwen36_27b_awq.cmd")
    assert "WindowsPowerShell" in text
    assert "smoke_perf_qwen36_27b_awq.ps1" in text


def test_smoke_perf_nvfp4_ps1_pins_models_and_dflash_toggle():
    text = _read("scripts/windows/smoke_perf_qwen38_nvfp4.ps1")
    assert "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4" in text
    assert "8003" in text
    assert "VLLM_PERF_ENABLE_DFLASH2" in text
    assert "incoai/Qwen3.8-27B-DFlash2" in text
    assert "SMOKE PERF OK" in text


def test_smoke_perf_nvfp4_cmd_uses_system_powershell():
    text = _read("scripts/windows/smoke_perf_qwen38_nvfp4.cmd")
    assert "WindowsPowerShell" in text
    assert "smoke_perf_qwen38_nvfp4.ps1" in text


def test_quality_matrix_script_exists():
    text = _read("scripts/windows/smoke_quality_matrix.ps1")
    assert "SMOKE QUALITY" in text
    assert "Qwen3.5-9B-AWQ" in text or "QuantTrio/Qwen3.5-9B-AWQ" in text


def test_perf_gate_has_gate2_section():
    text = _read("docs/windows/PERF_GATE.md")
    assert "Gate 2" in text
    assert "Qwen3.6-27B-AWQ" in text
    assert "QUASAR" in text or "NVFP4" in text
    assert "Speed summary" in text or "speed summary" in text.lower()
