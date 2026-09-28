"""Import gate after Windows build. Fails until extensions install."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
        import vllm
        import flash_attn_v100
        from flash_attn_v100 import flash_attn_v100_cuda, paged_kv_utils
    except Exception as exc:  # noqa: BLE001 - smoke must show any failure
        print("SMOKE IMPORTS FAIL:", repr(exc))
        return 1

    print("SMOKE IMPORTS OK")
    print("  Torch", torch.__version__)
    print("  vLLM", getattr(vllm, "__version__", "?"))
    print("  flash_attn_v100", getattr(flash_attn_v100, "__version__", "?"))
    print("  modules", flash_attn_v100_cuda, paged_kv_utils)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
