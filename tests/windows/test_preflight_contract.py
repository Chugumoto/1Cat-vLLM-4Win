from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_preflight_script_exists():
    assert (ROOT / "scripts/windows/preflight_env.py").is_file()


def test_smoke_imports_script_exists():
    assert (ROOT / "scripts/windows/smoke_imports.py").is_file()
