"""Lightweight checks that Windows platform constants exist after port."""

from pathlib import Path


SETUP = Path(__file__).resolve().parents[2] / "setup.py"


def test_setup_defines_is_windows():
    text = SETUP.read_text(encoding="utf-8")
    assert "IS_WINDOWS" in text
    assert 'platform.system() == "Windows"' in text


def test_setup_disables_fa3_on_windows():
    text = SETUP.read_text(encoding="utf-8")
    assert "VLLM_DISABLE_FA3_BUILD" in text
    assert "IS_WINDOWS" in text


def test_setup_includes_windows_requirements():
    text = SETUP.read_text(encoding="utf-8")
    assert "windows.txt" in text
