#include <torch/extension.h>

#include <c10/util/Optional.h>

#include <vector>

std::vector<torch::Tensor> gdn_forward(torch::Tensor q, torch::Tensor k,
                                       torch::Tensor v, torch::Tensor gate,
                                       torch::Tensor beta,
                                       c10::optional<torch::Tensor> initial_state,
                                       double scale, bool output_final_state,
                                       bool gate_is_exp);

std::vector<torch::Tensor> gdn_forward_vlk_varlen(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor gate,
    torch::Tensor beta, c10::optional<torch::Tensor> initial_state,
    torch::Tensor cu_seqlens, double scale, bool output_final_state,
    bool validate_cu_seqlens, bool gate_is_exp,
    c10::optional<torch::Tensor> output_arg);

void gdn_decode_mixed_qkv_global_state(torch::Tensor mixed_qkv, torch::Tensor a,
                                       torch::Tensor b, torch::Tensor A_log,
                                       torch::Tensor dt_bias,
                                       torch::Tensor state,
                                       torch::Tensor state_indices,
                                       torch::Tensor output, double scale,
                                       bool use_qk_l2norm);

void gdn_decode_mixed_qkv_ddtree_state(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state,
    torch::Tensor state_indices, torch::Tensor parent_ids,
    torch::Tensor num_accepted_tokens, torch::Tensor cu_seqlens,
    torch::Tensor output, double scale, bool use_qk_l2norm);

int resolve_column_groups_per_block(int tokens, int q_heads, int v_heads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_forward", &gdn_forward, "SM70/SM75 FlashQLA GDN forward");
  m.def("gdn_forward_vlk_varlen", &gdn_forward_vlk_varlen,
        "SM70/SM75 FlashQLA GDN forward for vLLM [N,Hv,V,K] state");
  m.def("gdn_decode_mixed_qkv_global_state",
        &gdn_decode_mixed_qkv_global_state,
        "SM70/SM75 FlashQLA fused mixed-QKV decode for vLLM global state");
  m.def("gdn_decode_mixed_qkv_ddtree_state",
        &gdn_decode_mixed_qkv_ddtree_state,
        "SM70/SM75 FlashQLA mixed-QKV DDTree decode for vLLM global state");
  m.def("resolve_column_groups_per_block", &resolve_column_groups_per_block,
        "Resolve SM70/SM75 FlashQLA GDN column groups per block");
}
