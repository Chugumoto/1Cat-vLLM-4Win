"""Fail closed unless Windows V100 MVP stack is present."""
from __future__ import annotations

import platform
import sys


def main() -> int:
    errors: list[str] = []
    if platform.system() != "Windows":
        errors.append(f"OS must be Windows, got {platform.system()}")

    if not (sys.version_info.major == 3 and sys.version_info.minor == 12):
        errors.append(f"Python 3.12 required, got {sys.version.split()[0]}")

    try:
        import torch
    except ImportError:
        errors.append("torch is not installed")
        _print(errors)
        return 1

    ver = torch.__version__
    if not ver.startswith("2.10"):
        errors.append(f"torch 2.10.x required, got {ver}")

    cuda = torch.version.cuda
    if cuda is None or not str(cuda).startswith("12."):
        errors.append(f"torch must be CUDA 12.x build, got cuda={cuda}")

    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is False")
    else:
        name = torch.cuda.get_device_name(0)
        if "V100" not in name.upper() and "TESLA V100" not in name.upper():
            # Allow substring V100 in common names like "Tesla V100-SXM2-16GB"
            if "V100" not in name:
                errors.append(f"GPU0 must be Tesla V100 for MVP smoke, got {name!r}")

        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) != (7, 0):
            errors.append(f"expect SM 7.0, got {major}.{minor}")

    _print(errors)
    return 1 if errors else 0


def _print(errors: list[str]) -> None:
    if errors:
        print("PREFLIGHT FAIL:")
        for e in errors:
            print(f"  - {e}")
    else:
        import torch

        print("PREFLIGHT OK")
        print("  Python", sys.version.split()[0])
        print("  Torch", torch.__version__, "CUDA", torch.version.cuda)
        print("  GPU", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    raise SystemExit(main())
