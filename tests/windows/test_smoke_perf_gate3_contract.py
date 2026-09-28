from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_smoke_perf_35b_ps1_pins_model_port_mtp():
    text = (ROOT / "scripts/windows/smoke_perf_qwen36_35b_a3b_awq.ps1").read_text(encoding="utf-8")
    assert "QuantTrio/Qwen3.6-35B-A3B-AWQ" in text
    assert "8004" in text
    assert "max-num-batched-tokens" in text
    assert "VLLM_PERF_ENABLE_MTP" in text
    assert "vllm.entrypoints.cli.main" in text
    assert '"method":"mtp"' in text.replace(" ", "") or "'method':'mtp'" in text.replace(" ", "") or 'method":"mtp"' in text
    assert "SMOKE PERF OK" in text
    assert "perf_qwen36_35b_a3b_awq" in text


def test_smoke_perf_35b_cmd_uses_system_powershell():
    text = (ROOT / "scripts/windows/smoke_perf_qwen36_35b_a3b_awq.cmd").read_text(encoding="utf-8")
    assert "WindowsPowerShell" in text
    assert "smoke_perf_qwen36_35b_a3b_awq.ps1" in text


def test_smoke_perf_35b_nvfp4_ps1_pins_model_port_mtp():
    text = (ROOT / "scripts/windows/smoke_perf_qwen36_35b_a3b_nvfp4.ps1").read_text(
        encoding="utf-8"
    )
    assert "nvidia/Qwen3.6-35B-A3B-NVFP4" in text
    assert "8005" in text
    assert "FLASH_ATTN_V100" in text
    assert "max-num-batched-tokens" in text
    assert "VLLM_PERF_ENABLE_MTP" in text
    assert "perf_qwen36_35b_a3b_nvfp4" in text


def test_smoke_perf_35b_nvfp4_cmd_uses_system_powershell():
    text = (ROOT / "scripts/windows/smoke_perf_qwen36_35b_a3b_nvfp4.cmd").read_text(
        encoding="utf-8"
    )
    assert "WindowsPowerShell" in text
    assert "smoke_perf_qwen36_35b_a3b_nvfp4.ps1" in text


def test_quality_matrix_has_35b_profile():
    text = (ROOT / "scripts/windows/smoke_quality_matrix.ps1").read_text(encoding="utf-8")
    assert "Qwen3.6-35B-A3B-AWQ" in text
    assert "8004" in text
    assert "nvidia/Qwen3.6-35B-A3B-NVFP4" in text
    assert "8005" in text
    assert "VLLM_QUALITY_INCLUDE_35B_NVFP4" in text


def test_perf_gate_has_gate3_section():
    text = (ROOT / "docs/windows/PERF_GATE.md").read_text(encoding="utf-8")
    assert "Gate 3" in text
    assert "35B-A3B" in text
    assert "MTP" in text
    assert "nvidia/Qwen3.6-35B-A3B-NVFP4" in text
