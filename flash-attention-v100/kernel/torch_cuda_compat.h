#pragma once
// SPDX-License-Identifier: Apache-2.0
//
// MSVC + CUDA 12.x: <torch/extension.h> pulls dynamo/compiled_autograd.h which
// does `namespace std { ... }` and clashes with CCCL (error C2872: ambiguous std).
// .cu kernels only need Tensor/CUDA helpers — keep pybind in fused_mha_api.cpp.

#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <optional>
#include <string>
#include <vector>
