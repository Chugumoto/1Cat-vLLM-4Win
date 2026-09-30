import os
import sys

cutlass_base_dir = sys.argv[1]

platform_h_file = os.path.join(
    cutlass_base_dir, "include", "cutlass", "platform", "platform.h"
)

if os.path.exists(platform_h_file):
    with open(platform_h_file, mode="r", encoding="utf-8") as file:
        header_content = "".join(file.readlines())

    old = "#if (201703L <=__cplusplus)"
    new = "#if defined(_MSC_VER) || (201703L <=__cplusplus)"
    if f"\n{old}\n" in header_content:
        header_content = header_content.replace(old, new)
        with open(platform_h_file, mode="w", encoding="utf-8") as file:
            file.write(header_content)

cuda_host_adapter_file = os.path.join(
    cutlass_base_dir, "include", "cutlass", "cuda_host_adapter.hpp"
)

if os.path.exists(cuda_host_adapter_file):
    with open(cuda_host_adapter_file, mode="r", encoding="utf-8") as file:
        header_content = "".join(file.readlines())

    old = "CUTLASS_HOST_DEVICE\n  Status memsetDevice"
    new = "CUTLASS_HOST\n  Status memsetDevice"
    if old in header_content:
        header_content = header_content.replace(old, new)
        with open(cuda_host_adapter_file, mode="w", encoding="utf-8") as file:
            file.write(header_content)
