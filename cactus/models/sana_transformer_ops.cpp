#include "sana_transformer_ops.h"
#include "model.h"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace cactus {
namespace engine {

namespace {

constexpr float kSanaQkNormEps = 1e-5f;

bool weight_file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

size_t maybe_apply_qk_rms_norm(
    CactusGraph* gb,
    size_t input_2d,
    size_t seq_len,
    size_t heads,
    size_t dim_head,
    const std::string& weight_path) {
    if (!weight_file_exists(weight_path)) {
        return input_2d;
    }

    size_t norm_w = gb->mmap_weights(weight_path);
    const auto& weight_shape = gb->get_output_buffer(norm_w).shape;
    if (weight_shape.empty()) {
        throw std::runtime_error("Invalid qk norm weight shape for " + weight_path);
    }

    size_t norm_dim = weight_shape.back();
    if (norm_dim == dim_head) {
        // qk_norm across heads: normalize each head chunk independently.
        size_t x = gb->reshape(input_2d, {seq_len, heads, dim_head});
        x = gb->reshape(x, {seq_len * heads, dim_head});
        x = gb->rms_norm(x, norm_w, kSanaQkNormEps);
        x = gb->reshape(x, {seq_len, heads, dim_head});
        return gb->reshape(x, {seq_len, heads * dim_head});
    }

    if (norm_dim == heads * dim_head) {
        // Full hidden-dim RMSNorm.
        return gb->rms_norm(input_2d, norm_w, kSanaQkNormEps);
    }

    throw std::runtime_error(
        "Unsupported qk norm weight dim (" + std::to_string(norm_dim) + ") for " + weight_path);
}

size_t build_glumbconv(CactusGraph* gb, size_t input, const std::string& prefix, const std::string& model_folder, size_t dim) {
    size_t x = input; // assumed N, C, H, W
    size_t hidden_channels = static_cast<size_t>(2.5f * dim);

    // conv_inverted, then SiLU on all channels (HF: act before conv_depth)
    size_t w_inv = gb->mmap_weights(model_folder + "/" + prefix + ".conv_inverted.weight.weights");
    size_t b_inv = gb->mmap_weights(model_folder + "/" + prefix + ".conv_inverted.bias.weights");
    x = gb->conv2d(x, w_inv, b_inv, 1, 1, 0, 0);
    x = gb->silu(x);

    // conv_depth (groups = hidden_channels * 2)
    size_t w_dep = gb->mmap_weights(model_folder + "/" + prefix + ".conv_depth.weight.weights");
    size_t b_dep = gb->mmap_weights(model_folder + "/" + prefix + ".conv_depth.bias.weights");
    x = gb->conv2d(x, w_dep, b_dep, 1, 1, 1, 1, 1, 1, hidden_channels * 2);

    // chunk into hidden and gate
    size_t hidden = gb->slice(x, 1, 0, hidden_channels);
    size_t gate = gb->slice(x, 1, hidden_channels, hidden_channels);
    gate = gb->silu(gate);
    x = gb->multiply(hidden, gate); // GLU

    // conv_point
    size_t w_pt = gb->mmap_weights(model_folder + "/" + prefix + ".conv_point.weight.weights");
    x = gb->conv2d(x, w_pt, 0, 1, 1, 0, 0); // no bias

    return x;
}

size_t build_sana_linear_attn(CactusGraph* gb, size_t hidden, const std::string& prefix, const std::string& model_folder, size_t heads, size_t dim_head) {
    // hidden is 2D: {L, dim}
    size_t q = gb->matmul(hidden, gb->mmap_weights(model_folder + "/" + prefix + ".to_q.weight.weights"), true);
    size_t k = gb->matmul(hidden, gb->mmap_weights(model_folder + "/" + prefix + ".to_k.weight.weights"), true);
    size_t v = gb->matmul(hidden, gb->mmap_weights(model_folder + "/" + prefix + ".to_v.weight.weights"), true);
    const std::string q_bias = model_folder + "/" + prefix + ".to_q.bias.weights";
    const std::string k_bias = model_folder + "/" + prefix + ".to_k.bias.weights";
    const std::string v_bias = model_folder + "/" + prefix + ".to_v.bias.weights";
    if (weight_file_exists(q_bias)) {
        q = gb->add(q, gb->mmap_weights(q_bias));
    }
    if (weight_file_exists(k_bias)) {
        k = gb->add(k, gb->mmap_weights(k_bias));
    }
    if (weight_file_exists(v_bias)) {
        v = gb->add(v, gb->mmap_weights(v_bias));
    }

    auto shape = gb->get_output_buffer(hidden).shape;
    size_t L = shape[0]; // hidden is 2D: {L, dim}

    q = maybe_apply_qk_rms_norm(
        gb, q, L, heads, dim_head, model_folder + "/" + prefix + ".norm_q.weight.weights");
    k = maybe_apply_qk_rms_norm(
        gb, k, L, heads, dim_head, model_folder + "/" + prefix + ".norm_k.weight.weights");

    gb->register_debug_node(0, prefix + ".q", q);
    gb->register_debug_node(0, prefix + ".k", k);
    gb->register_debug_node(0, prefix + ".v", v);

    // Reshape for multi-head attention: {L, dim} -> {1, L, heads, dim_head} -> {1, heads, L, dim_head}
    q = gb->reshape(q, {1, L, heads, dim_head});
    q = gb->transposeN(q, {0, 2, 1, 3}); // {1, heads, L, dim_head}

    k = gb->reshape(k, {1, L, heads, dim_head});
    k = gb->transposeN(k, {0, 2, 1, 3}); // {1, heads, L, dim_head}

    v = gb->reshape(v, {1, L, heads, dim_head});
    v = gb->transposeN(v, {0, 2, 1, 3}); // {1, heads, L, dim_head}

    // True linear attention API (ReLU + K^T * V)
    float scale = 1.0f / std::sqrt((float)dim_head);
    size_t out = gb->linear_attention(q, k, v, scale, ComputeBackend::CPU);

    // Reshape back: {1, heads, L, dim_head} -> {1, L, heads, dim_head} -> {L, heads*dim_head}
    out = gb->transposeN(out, {0, 2, 1, 3}); // {1, L, heads, dim_head}
    out = gb->reshape(out, {L, heads * dim_head}); // back to 2D
    gb->register_debug_node(0, prefix + ".core_out", out);

    size_t w_out0 = gb->mmap_weights(model_folder + "/" + prefix + ".to_out.0.weight.weights");
    size_t b_out0 = gb->mmap_weights(model_folder + "/" + prefix + ".to_out.0.bias.weights");
    out = gb->matmul(out, w_out0, true);
    out = gb->add(out, b_out0);

    return out;
}

size_t build_cross_attn(CactusGraph* gb, size_t hidden, size_t enc_hidden, const std::string& prefix, const std::string& model_folder, size_t heads, size_t dim_head, size_t kv_heads, size_t encoder_mask_node = 0) {
    // hidden is 2D: {L, dim}, enc_hidden is 2D: {S, enc_dim}
    size_t q = gb->matmul(hidden, gb->mmap_weights(model_folder + "/" + prefix + ".to_q.weight.weights"), true);
    size_t k = gb->matmul(enc_hidden, gb->mmap_weights(model_folder + "/" + prefix + ".to_k.weight.weights"), true);
    size_t v = gb->matmul(enc_hidden, gb->mmap_weights(model_folder + "/" + prefix + ".to_v.weight.weights"), true);
    const std::string q_bias = model_folder + "/" + prefix + ".to_q.bias.weights";
    const std::string k_bias = model_folder + "/" + prefix + ".to_k.bias.weights";
    const std::string v_bias = model_folder + "/" + prefix + ".to_v.bias.weights";
    if (weight_file_exists(q_bias)) {
        q = gb->add(q, gb->mmap_weights(q_bias));
    }
    if (weight_file_exists(k_bias)) {
        k = gb->add(k, gb->mmap_weights(k_bias));
    }
    if (weight_file_exists(v_bias)) {
        v = gb->add(v, gb->mmap_weights(v_bias));
    }

    auto shape = gb->get_output_buffer(hidden).shape;
    size_t L = shape[0];

    auto shape_enc = gb->get_output_buffer(enc_hidden).shape;
    size_t S = shape_enc[0];

    q = maybe_apply_qk_rms_norm(
        gb, q, L, heads, dim_head, model_folder + "/" + prefix + ".norm_q.weight.weights");
    k = maybe_apply_qk_rms_norm(
        gb, k, S, kv_heads, dim_head, model_folder + "/" + prefix + ".norm_k.weight.weights");

    // attention() expects [B, Seq, Heads, HeadDim]
    q = gb->reshape(q, {1, L, heads, dim_head});
    k = gb->reshape(k, {1, S, kv_heads, dim_head});
    v = gb->reshape(v, {1, S, kv_heads, dim_head});

    size_t attn = (encoder_mask_node != 0)
        ? gb->attention_with_mask(q, k, v, 1.0f / std::sqrt((float)dim_head), encoder_mask_node)
        : gb->attention(q, k, v, 1.0f / std::sqrt((float)dim_head), false);
    size_t out = gb->reshape(attn, {L, heads * dim_head}); // back to 2D

    size_t w_out0 = gb->mmap_weights(model_folder + "/" + prefix + ".to_out.0.weight.weights");
    size_t b_out0 = gb->mmap_weights(model_folder + "/" + prefix + ".to_out.0.bias.weights");
    out = gb->matmul(out, w_out0, true);
    out = gb->add(out, b_out0);

    return out;
}

} // namespace

size_t build_sana_transformer_block(CactusGraph* gb, size_t hidden, size_t enc_hidden, size_t timestep_embedded, const std::string& prefix, const std::string& model_folder, size_t dim, size_t height, size_t width, size_t num_heads, size_t dim_head, size_t num_cross_heads, size_t cross_dim_head, size_t encoder_mask_node) {
    // hidden is 2D: {L, dim}, enc_hidden is 2D: {S, enc_dim}
    auto shape = gb->get_output_buffer(hidden).shape;
    size_t L = shape[0];

    size_t scale_shift_table = gb->mmap_weights(model_folder + "/" + prefix + ".scale_shift_table.weights");
    // timestep_embedded is {1, 6*dim} -> reshape to {1, 6, dim}
    size_t t_reshape = gb->reshape(timestep_embedded, {1, 6, dim});
    size_t modulations = gb->add(t_reshape, scale_shift_table);

    size_t shift_msa = gb->slice(modulations, 1, 0, 1);
    size_t scale_msa = gb->slice(modulations, 1, 1, 1);
    size_t gate_msa  = gb->slice(modulations, 1, 2, 1);
    size_t shift_mlp = gb->slice(modulations, 1, 3, 1);
    size_t scale_mlp = gb->slice(modulations, 1, 4, 1);
    size_t gate_mlp  = gb->slice(modulations, 1, 5, 1);

    // Reshape modulations to 2D for broadcasting: {1, dim}
    shift_msa = gb->reshape(shift_msa, {1, dim});
    scale_msa = gb->reshape(scale_msa, {1, dim});
    gate_msa = gb->reshape(gate_msa, {1, dim});
    shift_mlp = gb->reshape(shift_mlp, {1, dim});
    scale_mlp = gb->reshape(scale_mlp, {1, dim});
    gate_mlp = gb->reshape(gate_mlp, {1, dim});

    size_t norm_hidden = gb->layernorm(hidden, 1e-6f);
    size_t mod_scale_msa = gb->scalar_add(scale_msa, 1.0f);
    norm_hidden = gb->multiply(norm_hidden, mod_scale_msa);
    norm_hidden = gb->add(norm_hidden, shift_msa);
    gb->register_debug_node(0, prefix + ".attn1_in", norm_hidden);

    size_t attn_out = build_sana_linear_attn(gb, norm_hidden, prefix + ".attn1", model_folder, num_heads, dim_head);
    gb->register_debug_node(0, prefix + ".attn1_out", attn_out);
    hidden = gb->add(hidden, gb->multiply(attn_out, gate_msa));

    size_t attn2_out = build_cross_attn(gb, hidden, enc_hidden, prefix + ".attn2", model_folder, num_cross_heads, cross_dim_head, num_cross_heads, encoder_mask_node);
    gb->register_debug_node(0, prefix + ".attn2_out", attn2_out);
    hidden = gb->add(hidden, attn2_out);

    norm_hidden = gb->layernorm(hidden, 1e-6f);
    size_t mod_scale_mlp = gb->scalar_add(scale_mlp, 1.0f);
    norm_hidden = gb->multiply(norm_hidden, mod_scale_mlp);
    norm_hidden = gb->add(norm_hidden, shift_mlp);

    // PyTorch path is [B, L, C] -> [B, H, W, C] -> [B, C, H, W].
    norm_hidden = gb->reshape(norm_hidden, {1, height, width, dim});
    norm_hidden = gb->transposeN(norm_hidden, {0, 3, 1, 2});

    size_t ff_out = build_glumbconv(gb, norm_hidden, prefix + ".ff", model_folder, dim);
    gb->register_debug_node(0, prefix + ".ff_out", ff_out);

    // Convert back: [B, C, H, W] -> [B, H, W, C] -> [L, C]
    ff_out = gb->transposeN(ff_out, {0, 2, 3, 1});
    ff_out = gb->reshape(ff_out, {L, dim});

    hidden = gb->add(hidden, gb->multiply(ff_out, gate_mlp));
    gb->register_debug_node(0, prefix + ".out", hidden);

    return hidden;
}

} // namespace engine
} // namespace cactus
