# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""uvloop on Unix, winloop on Windows (SystemPanic/vllm-windows pattern)."""

from __future__ import annotations

import os
import platform

if platform.system() == "Windows":
    import winloop as uvloop_impl

    # Windows does not support fork
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # Disable libuv on Windows by default
    os.environ["USE_LIBUV"] = os.environ.get("USE_LIBUV", "0")
else:
    import uvloop as uvloop_impl

__all__ = ["uvloop_impl"]
