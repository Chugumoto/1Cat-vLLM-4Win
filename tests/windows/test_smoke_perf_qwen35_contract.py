from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_smoke_perf_ps1_exists_and_pins_model_port():
    text = _read("scripts/windows/smoke_perf_qwen35_awq.ps1")
    assert "QuantTrio/Qwen3.5-9B-AWQ" in text
    assert "8001" in text
    assert "SMOKE PERF OK" in text
    assert "CUDA_DEVICE_ORDER" in text
    assert "perf_results" in text or "PERF_RESULT" in text


def test_smoke_perf_cmd_wrapper_uses_system_powershell():
    text = _read("scripts/windows/smoke_perf_qwen35_awq.cmd")
    assert "WindowsPowerShell" in text
    assert "smoke_perf_qwen35_awq.ps1" in text


def test_perf_gate_doc_exists():
    text = _read("docs/windows/PERF_GATE.md")
    assert "Qwen3.5-9B-AWQ" in text
    assert "V100" in text
    assert "tok/s" in text or "throughput" in text.lower()
    assert "VLLM_PERF_TOK_S_FLOOR=43" in text or "**43**" in text
