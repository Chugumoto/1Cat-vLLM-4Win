#include <torch/extension.h>

#include <vector>

torch::Tensor paged_to_contiguous(torch::Tensor paged_cache,
                                  torch::Tensor block_table,
                                  torch::Tensor seq_lens);

std::vector<torch::Tensor> paged_kv_to_contiguous(
    torch::Tensor paged_key_cache, torch::Tensor paged_value_cache,
    torch::Tensor block_table, torch::Tensor seq_lens);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_to_contiguous", &paged_to_contiguous,
        "Paged KV Cache to Contiguous (Stride-Aware)");
  m.def("paged_kv_to_contiguous", &paged_kv_to_contiguous,
        "Paged KV Cache to Contiguous K/V Pair (Stride-Aware)");
}
