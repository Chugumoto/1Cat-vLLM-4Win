# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, D.Skryabin

import os
from pathlib import Path

from packaging.version import parse
from setuptools import setup

this_dir = Path(__file__).parent.resolve()


def _cxx_args():
    if os.name == "nt":
        return [
            "/O2",
            "/std:c++17",
            "/Zc:__cplusplus",
            "/Zc:preprocessor",
            "/DNOMINMAX",
            "/DWIN32_LEAN_AND_MEAN",
        ]
    return ["-O3", "-std=c++17"]


def _force_cuda_home() -> Path:
    """Prefer CUDA_HOME/CUDA_PATH from the environment (CUDA 12.8 for V100).

    On machines with multiple toolkits, Torch may otherwise resolve CUDA 13.
    """
    for key in ("CUDA_HOME", "CUDA_PATH"):
        raw = os.environ.get(key)
        if not raw:
            continue
        home = Path(raw)
        if (home / "include" / "cuda_runtime.h").is_file():
            os.environ["CUDA_HOME"] = str(home)
            os.environ["CUDA_PATH"] = str(home)
            return home
    raise RuntimeError(
        "CUDA 12.x toolkit with include/cuda_runtime.h not found. "
        "Set CUDA_HOME to the CUDA 12.8 root before building."
    )


def _include_dirs(*extra: Path) -> list:
    cuda_home = _force_cuda_home()
    dirs = [this_dir / "include", this_dir / "kernel", cuda_home / "include", *extra]
    return [str(p) for p in dirs]


def _nvcc_args():
    args = [
        "-O3",
        "-std=c++17",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]
    if os.name == "nt":
        # CUDA 12.8 rejects MSVC >= 1950; keep override for mixed VS installs.
        args.append("-allow-unsupported-compiler")
        # Mitigate C2872 std ambiguity (CCCL/cuda::std vs MSVC std) with Torch headers.
        args.extend(
            [
                "-Xcompiler",
                "/Zc:__cplusplus",
                "-Xcompiler",
                "/Zc:preprocessor",
                "-Xcompiler",
                "/DNOMINMAX",
                "-Xcompiler",
                "/DWIN32_LEAN_AND_MEAN",
            ]
        )
    return args


def get_ext_modules():
    # Force CUDA_HOME before Torch probes the toolkit.
    _force_cuda_home()
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError as e:
        raise RuntimeError(
            "torch is required to build flash_attn_v100. "
            "Please install torch >= 2.5 first (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu118`)."
        ) from e

    return [
        CUDAExtension(
            name="flash_attn_v100_cuda",
            sources=[
                "kernel/fused_mha_api.cpp",
                "kernel/fused_mha_forward.cu",
                "kernel/fused_mha_forward_paged.cu",
                "kernel/fp8_kv_bridge.cu",
                "kernel/flash_decode_paged.cu",
                "kernel/flash_decode_turboquant.cu",
                "kernel/fused_mha_backward.cu",
            ],
            include_dirs=_include_dirs(),
            extra_compile_args={
                "cxx": _cxx_args(),
                "nvcc": _nvcc_args(),
            },
        ),
        CUDAExtension(
            name="paged_kv_utils",
            sources=[
                "kernel/paged_to_contiguous.cu",
                "kernel/paged_kv_utils_api.cpp",
            ],
            include_dirs=_include_dirs(),
            extra_compile_args={
                "cxx": _cxx_args(),
                "nvcc": _nvcc_args(),
            },
        ),
    ]


def get_cmdclass():
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ImportError as e:
        raise RuntimeError(
            "torch is required to build flash_attn_v100. "
            "Please install torch >= 2.5 first."
        ) from e

    class CustomBuildExtension(BuildExtension):
        def build_extensions(self):
            import torch
            from torch.utils.cpp_extension import CUDA_HOME

            home = _force_cuda_home()
            if CUDA_HOME is None and home is None:
                raise RuntimeError("CUDA toolkit is required but CUDA_HOME is not set.")
            if torch.version.cuda is None:
                raise RuntimeError("A CUDA-enabled PyTorch build is required.")
            if parse(torch.version.cuda) < parse("11.6"):
                raise RuntimeError(
                    f"CUDA version {torch.version.cuda} < 11.6 is not supported. "
                    "Please use CUDA ≥ 11.6 (e.g., PyTorch built with CUDA 11.8/12.x)."
                )
            if parse(torch.version.cuda).major >= 13:
                raise RuntimeError(
                    f"Refusing Torch CUDA {torch.version.cuda}: "
                    "Volta/SM70 needs CUDA 12.x."
                )
            super().build_extensions()

    return {"build_ext": CustomBuildExtension}


try:
    with open(this_dir / "README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Flash Attention implementation for Tesla V100"

setup(
    name="flash_attn_v100",
    version="1.2.0",
    packages=["flash_attn_v100"],
    ext_modules=get_ext_modules(),
    cmdclass=get_cmdclass(),
    python_requires=">=3.10",
    install_requires=["torch>=2.5", "einops", "packaging"],
    zip_safe=False,
    description="Flash Attention implementation under unsupported Tesla V100",
    license="BSD-3-Clause",
    author="D.Skryabin",
    author_email="tg @ai_bond007",
    url="https://github.com/ai-bond/flash-attention-v100",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
)
