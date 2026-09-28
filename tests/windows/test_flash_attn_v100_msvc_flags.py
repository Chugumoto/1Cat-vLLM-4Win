from pathlib import Path

SETUP = Path(__file__).resolve().parents[2] / "flash-attention-v100" / "setup.py"


def test_msvc_cxx_flags_present():
    text = SETUP.read_text(encoding="utf-8")
    assert "/O2" in text or 'os.name == "nt"' in text
    assert "std=c++17" in text or "/std:c++17" in text
