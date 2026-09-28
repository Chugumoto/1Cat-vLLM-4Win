from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_windows_requirements_exist_and_list_core_deps():
    text = (ROOT / "requirements/windows.txt").read_text(encoding="utf-8")
    assert "winloop" in text
    assert "triton-windows" in text
    assert "portalocker" in text
    # common.txt uses x86_64 markers; Windows is AMD64 and needs explicit pins.
    assert "llguidance" in text
    assert "xgrammar" in text


def test_build_cuda_txt_omits_unconditional_patchelf():
    text = (ROOT / "requirements/build/cuda.txt").read_text(encoding="utf-8")
    # Either removed, or clearly environment-marked; bare unconditional pin is forbidden
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        if s.startswith("patchelf"):
            assert "sys_platform" in s, (
                "patchelf must not be unconditional on Windows builds"
            )


def test_pyproject_build_requires_gates_patchelf():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "patchelf>=0.19.1.0" in text
    assert '"patchelf>=0.19.1.0",' not in text
    assert 'sys_platform == \\"linux\\"' in text
