# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Policy mirror for SM70 QPN4 M=1 split_k selection.

Keep in sync with ``pick_qpn4_*_split_k`` in
``csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu``. The CUDA dispatcher used to
hard-code dense ``split_k=17`` (tuned for Qwen3.8-27B ``intermediate=17408``);
widths such as Coder-32B ``intermediate=27648`` then failed with
``invalid split_k for K``.
"""

from __future__ import annotations

import pytest


def pick_qpn4_dense_split_k(k: int) -> int:
    assert k > 0 and k % 16 == 0
    groups = k // 16
    for split_k in (17, 16, 10, 8, 4):
        if groups % split_k == 0:
            return split_k
    raise AssertionError(f"no supported dense split_k for K={k}")


def pick_qpn4_gated_split_k(k: int) -> int:
    assert k > 0 and k % 16 == 0
    groups = k // 16
    for split_k in (8, 16, 10, 4):
        if groups % split_k == 0:
            return split_k
    raise AssertionError(f"no supported gated split_k for K={k}")


@pytest.mark.parametrize(
    "k,expected",
    [
        # Qwen3.8-27B / Quasar down_proj: keep measured dense 17 when legal.
        (17408, 17),
        # Qwen2.5-Coder-32B down_proj: 17 does not divide → fall back to 16.
        (27648, 16),
        # Shared hidden width (dense non-gated path if used).
        (5120, 16),
    ],
)
def test_dense_split_k_for_known_model_widths(k: int, expected: int) -> None:
    assert pick_qpn4_dense_split_k(k) == expected


@pytest.mark.parametrize(
    "k,expected",
    [
        (5120, 8),
        (17408, 8),
        (27648, 8),
    ],
)
def test_gated_split_k_prefers_measured_eight(k: int, expected: int) -> None:
    assert pick_qpn4_gated_split_k(k) == expected


def test_any_k_multiple_of_128_has_dense_fallback() -> None:
    # Alignment contract in nvfp4_qpn4_gemm_* requires k % 128 == 0, which
    # guarantees groups % 8 == 0 so {8,4} always admit a candidate.
    for k in range(128, 128 * 400, 128):
        split_k = pick_qpn4_dense_split_k(k)
        assert (k // 16) % split_k == 0
        assert split_k in (17, 16, 10, 8, 4)
