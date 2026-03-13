#include "sana_vae_ops.h"
#include "model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>

namespace cactus {
namespace engine {

namespace {

size_t build_res_block(CactusGraph* gb, size_t input, const std::string& prefix, const std::string& model_folder) {
    size_t x = input;
    // conv1
    size_t w1 = gb->mmap_weights(model_folder + "/" + prefix + ".conv1.weight.weights");
    size_t b1 = gb->mmap_weights(model_folder + "/" + prefix + ".conv1.bias.weights");
    x = gb->conv2d(x, w1, b1, 1, 1, 1, 1);

    // silu
    x = gb->silu(x);

    // conv2
    size_t w2 = gb->mmap_weights(model_folder + "/" + prefix + ".conv2.weight.weights");
    size_t b2 = 0; // bias=False
    x = gb->conv2d(x, w2, b2, 1, 1, 1, 1);

    // norm + residual
    x = gb->transposeN(x, {0, 2, 3, 1}); // (N, C, H, W) -> (N, H, W, C)
    auto shape = gb->get_output_buffer(x).shape;
    size_t N = shape[0], H = shape[1], W = shape[2], C = shape[3];
    x = gb->reshape(x, {N * H * W, C});
    size_t w_norm = gb->mmap_weights(model_folder + "/" + prefix + ".norm.weight.weights");
    size_t b_norm = gb->mmap_weights(model_folder + "/" + prefix + ".norm.bias.weights");
    x = gb->rms_norm(x, w_norm, 1e-5f);
    x = gb->add(x, b_norm);
    x = gb->reshape(x, {N, H, W, C});
    x = gb->transposeN(x, {0, 3, 1, 2}); // (N, H, W, C) -> (N, C, H, W)

    size_t out = gb->add(x, input);
    gb->register_debug_node(1, prefix + ".out", out);
    return out;
}

size_t build_dc_down_block(CactusGraph* gb, size_t input, const std::string& prefix, const std::string& model_folder, size_t in_channels, size_t out_channels, bool downsample) {
    size_t stride = downsample ? 1 : 2;
    size_t w = gb->mmap_weights(model_folder + "/" + prefix + ".conv.weight.weights");
    size_t b = gb->mmap_weights(model_folder + "/" + prefix + ".conv.bias.weights");
    size_t x = gb->conv2d(input, w, b, stride, stride, 1, 1);

    auto shrink_spatial = [&](size_t t, size_t c) {
        auto shape = gb->get_output_buffer(t).shape;
        size_t N = shape[0], H = shape[2], W = shape[3];
        size_t r1 = gb->reshape(t, {N, c, H / 2, 2, W / 2, 2});
        size_t r2 = gb->transposeN(r1, {0, 1, 3, 5, 2, 4});
        return gb->reshape(r2, {N, c * 4, H / 2, W / 2});
    };

    if (downsample) {
        x = shrink_spatial(x, out_channels);
    }

    // shortcut
    size_t y = shrink_spatial(input, in_channels);

    size_t group_size = in_channels * 4 / out_channels;
    auto shape_y = gb->get_output_buffer(y).shape;
    size_t N = shape_y[0], H = shape_y[2], W = shape_y[3];

    y = gb->reshape(y, {N, out_channels, group_size, H, W});
    y = gb->mean(y, 2); // mean over group_size

    return gb->add(x, y);
}

size_t build_dc_up_block(CactusGraph* gb, size_t input, const std::string& prefix, const std::string& model_folder, size_t in_channels, size_t out_channels) {
    // DCUpBlock2d with interpolate=True (this model uses upsample_block_type="interpolate"):
    // main path: nearest-neighbor 2x upsample, then conv(in_channels -> out_channels, 3x3)
    // shortcut:  repeat_interleave(input, out_channels*4/in_channels, axis=1) then pixel_shuffle(2)

    // Main path: nearest-neighbor 2x spatial upsample + conv
    size_t x = gb->repeat_interleave(input, 2, 2); // [N, C, H*2, W]
    x = gb->repeat_interleave(x, 2, 3);             // [N, C, H*2, W*2]
    size_t w = gb->mmap_weights(model_folder + "/" + prefix + ".conv.weight.weights");
    size_t b = gb->mmap_weights(model_folder + "/" + prefix + ".conv.bias.weights");
    x = gb->conv2d(x, w, b, 1, 1, 1, 1);  // [N, out_channels, H*2, W*2]

    // Shortcut: pixel_shuffle (channel->spatial expansion)
    auto expand_spatial = [&](size_t t, size_t c4) {
        auto shape = gb->get_output_buffer(t).shape;
        size_t N = shape[0], H = shape[2], W = shape[3];
        size_t r1 = gb->reshape(t, {N, c4 / 4, 2, 2, H, W});
        size_t r2 = gb->transposeN(r1, {0, 1, 4, 2, 5, 3});
        return gb->reshape(r2, {N, c4 / 4, H * 2, W * 2});
    };

    size_t repeats = out_channels * 4 / in_channels;
    size_t y = gb->repeat_interleave(input, repeats, 1);  // [N, in_channels*repeats = out_channels*4, H, W]
    y = expand_spatial(y, in_channels * repeats);          // [N, out_channels, H*2, W*2]

    return gb->add(x, y);
}

size_t build_autoencoderdc_glumbconv(CactusGraph* gb, size_t input, const std::string& prefix, const std::string& model_folder, size_t dim) {
    size_t x = input; // N, C, H, W
    size_t hidden_channels = 4 * dim;

    // Save residual
    size_t res = x;

    // conv_inverted, then SiLU on all channels (HF: act before conv_depth)
    size_t w_inv = gb->mmap_weights(model_folder + "/" + prefix + ".conv_inverted.weight.weights");
    size_t b_inv = gb->mmap_weights(model_folder + "/" + prefix + ".conv_inverted.bias.weights");
    x = gb->conv2d(x, w_inv, b_inv, 1, 1, 0, 0);
    x = gb->silu(x);

    // conv_depth (groups = hidden_channels * 2)
    size_t w_dep = gb->mmap_weights(model_folder + "/" + prefix + ".conv_depth.weight.weights");
    size_t b_dep = gb->mmap_weights(model_folder + "/" + prefix + ".conv_depth.bias.weights");
    x = gb->conv2d(x, w_dep, b_dep, 1, 1, 1, 1, 1, 1, hidden_channels * 2);

    // chunk into hidden and gate, SiLU on gate only
    size_t hidden = gb->slice(x, 1, 0, hidden_channels);
    size_t gate = gb->slice(x, 1, hidden_channels, hidden_channels);
    gate = gb->silu(gate);
    x = gb->multiply(hidden, gate); // GLU

    // conv_point
    size_t w_pt = gb->mmap_weights(model_folder + "/" + prefix + ".conv_point.weight.weights");
    x = gb->conv2d(x, w_pt, 0, 1, 1, 0, 0); // no bias

    // norm (RMSNorm) - applied post-convs in GLUMBConv
    size_t w_norm = gb->mmap_weights(model_folder + "/" + prefix + ".norm.weight.weights");
    size_t b_norm = gb->mmap_weights(model_folder + "/" + prefix + ".norm.bias.weights");
    x = gb->transposeN(x, {0, 2, 3, 1}); // N, C, H, W -> N, H, W, C
    auto shape = gb->get_output_buffer(x).shape;
    size_t N = shape[0], H = shape[1], W = shape[2], C = shape[3];
    x = gb->reshape(x, {N * H * W, C});
    
    // PyTorch AutoencoderDC GLUMBConv default eps=1e-5
    x = gb->rms_norm(x, w_norm, 1e-5f);
    x = gb->add(x, b_norm);
    
    x = gb->reshape(x, {N, H, W, C});
    x = gb->transposeN(x, {0, 3, 1, 2}); // N, H, W, C -> N, C, H, W

    size_t out = gb->add(res, x);
    gb->register_debug_node(1, prefix + ".out", out);
    return out;
}

size_t build_autoencoderdc_linear_attn(CactusGraph* gb, size_t input, const std::string& prefix, const std::string& model_folder, size_t /*dim*/, size_t heads) {
    size_t x = input; // N, C, H, W
    size_t res = x;

    auto shape = gb->get_output_buffer(x).shape;
    size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    size_t L = H * W;

    // N, C, H, W -> N, H, W, C -> N*L, C
    size_t x_2d = gb->transposeN(x, {0, 2, 3, 1});
    x_2d = gb->reshape(x_2d, {N * L, C});

    size_t q = gb->matmul(x_2d, gb->mmap_weights(model_folder + "/" + prefix + ".to_q.weight.weights"), true);
    size_t k = gb->matmul(x_2d, gb->mmap_weights(model_folder + "/" + prefix + ".to_k.weight.weights"), true);
    size_t v = gb->matmul(x_2d, gb->mmap_weights(model_folder + "/" + prefix + ".to_v.weight.weights"), true);

    size_t dim_head = C / heads;
    size_t inner_dim = heads * dim_head;

    // Reshape back to N, C, H, W
    q = gb->transposeN(gb->reshape(q, {N, H, W, inner_dim}), {0, 3, 1, 2});
    k = gb->transposeN(gb->reshape(k, {N, H, W, inner_dim}), {0, 3, 1, 2});
    v = gb->transposeN(gb->reshape(v, {N, H, W, inner_dim}), {0, 3, 1, 2});

    // Concat q,k,v -> [N, 3 * inner_dim, H, W]
    size_t qk = gb->concat(q, k, 1);
    size_t qkv = gb->concat(qk, v, 1);

    // Multiscale projection
    // proj_in: kernel=5, padding=2, groups=channels
    size_t w_proj_in = gb->mmap_weights(model_folder + "/" + prefix + ".to_qkv_multiscale.0.proj_in.weight.weights");
    size_t qkv_ms = gb->conv2d(qkv, w_proj_in, 0, 1, 1, 2, 2, 1, 1, 3 * inner_dim);

    // proj_out: kernel=1, padding=0, groups=3*heads
    size_t w_proj_out = gb->mmap_weights(model_folder + "/" + prefix + ".to_qkv_multiscale.0.proj_out.weight.weights");
    qkv_ms = gb->conv2d(qkv_ms, w_proj_out, 0, 1, 1, 0, 0, 1, 1, 3 * heads);

    // Concat with original -> [N, 6 * inner_dim, H, W]
    size_t hidden = gb->concat(qkv, qkv_ms, 1);

    // Reproduce diffusers bug logic:
    // hidden_states.reshape(B, -1, 3 * dim_head, H * W) => [N, 2 * heads, 3 * dim_head, L]
    hidden = gb->reshape(hidden, {N, 2 * heads, 3 * dim_head, L});

    // Chunk into qc, kc, vc
    size_t qc = gb->slice(hidden, 2, 0, dim_head);
    size_t kc = gb->slice(hidden, 2, dim_head, dim_head);
    size_t vc = gb->slice(hidden, 2, 2 * dim_head, dim_head);

    // qc is currently [N, 2*heads, dim_head, L]. Need [N, 2*heads, L, dim_head].
    qc = gb->transposeN(qc, {0, 1, 3, 2});
    kc = gb->transposeN(kc, {0, 1, 3, 2});
    vc = gb->transposeN(vc, {0, 1, 3, 2});

    float scale = 1.0f / std::sqrt(static_cast<float>(dim_head));
    size_t out = gb->linear_attention(qc, kc, vc, scale, ComputeBackend::CPU);

    // out is [N, 2*heads, L, dim_head] -> transpose to [N, 2*heads, dim_head, L]
    // (HF: apply_linear_attention returns [B, 2*heads, dim_head, L], then reshape to [B, 2*inner_dim, H, W])
    out = gb->transposeN(out, {0, 1, 3, 2}); // [N, 2*heads, dim_head, L]
    out = gb->reshape(out, {N, 2 * inner_dim, H, W});

    // out = attn.to_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
    out = gb->transposeN(out, {0, 2, 3, 1}); // N, H, W, 2*C
    out = gb->reshape(out, {N * L, 2 * inner_dim});

    size_t w_out = gb->mmap_weights(model_folder + "/" + prefix + ".to_out.weight.weights");
    out = gb->matmul(out, w_out, true);

    out = gb->reshape(out, {N, H, W, C});
    out = gb->transposeN(out, {0, 3, 1, 2}); // N, C, H, W

    // norm_out
    size_t w_norm_out = gb->mmap_weights(model_folder + "/" + prefix + ".norm_out.weight.weights");
    size_t b_norm_out = gb->mmap_weights(model_folder + "/" + prefix + ".norm_out.bias.weights");
    out = gb->transposeN(out, {0, 2, 3, 1}); // N, H, W, C
    out = gb->reshape(out, {N * H * W, C});
    out = gb->rms_norm(out, w_norm_out, 1e-5f);
    out = gb->add(out, b_norm_out);
    out = gb->reshape(out, {N, H, W, C});
    out = gb->transposeN(out, {0, 3, 1, 2}); // N, C, H, W

    size_t out_add = gb->add(res, out);
    gb->register_debug_node(2, prefix + ".out", out_add);
    return out_add;
}

size_t build_efficientvit_block(CactusGraph* gb, size_t input, const std::string& prefix, const std::string& model_folder, size_t dim, size_t heads) {
    size_t x = input; // N, C, H, W

    // 1. Linear Attention block
    x = build_autoencoderdc_linear_attn(gb, x, prefix + ".attn", model_folder, dim, heads);

    // 2. GLUMBConv block (with residual internally)
    x = build_autoencoderdc_glumbconv(gb, x, prefix + ".conv_out", model_folder, dim);

    gb->register_debug_node(3, prefix + ".out", x);
    return x;
}

} // namespace

bool sana_weight_file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

size_t build_autoencoder_dc_encode(CactusGraph* gb, size_t input, const std::string& prefix, const std::string& model_folder) {
    size_t x = input; // assume (N, C, H, W)

    std::string pfx = prefix.empty() ? "" : prefix;
    std::string sep = prefix.empty() ? "" : "/";
    auto make_path = [&](const std::string& name) {
        return model_folder + "/" + pfx + sep + name + ".weights";
    };

    // conv_in
    size_t conv_in_w = gb->mmap_weights(make_path("conv_in.weight"));
    size_t conv_in_b = gb->mmap_weights(make_path("conv_in.bias"));
    x = gb->conv2d(x, conv_in_w, conv_in_b, 1, 1, 1, 1);

    const std::vector<size_t> block_out_channels = {128, 256, 512, 512, 1024, 1024};
    const std::vector<size_t> layers_per_block = {2, 2, 2, 3, 3, 3};
    size_t num_blocks = block_out_channels.size();

    for (size_t i = 0; i < num_blocks; ++i) {
        std::string block_prefix = pfx + sep + "down_blocks." + std::to_string(i);
        size_t out_channel = block_out_channels[i];
        size_t num_layers = layers_per_block[i];

        for (size_t j = 0; j < num_layers; ++j) {
            std::string layer_prefix = block_prefix + "." + std::to_string(j);
            if (i >= 3) {
                // Deeper encoder stages use EfficientViT blocks (same as decoder)
                size_t heads = out_channel / 32;
                x = build_efficientvit_block(gb, x, layer_prefix, model_folder, out_channel, heads);
            } else {
                x = build_res_block(gb, x, layer_prefix, model_folder);
            }
        }

        if (i < num_blocks - 1 && num_layers > 0) {
            x = build_dc_down_block(
                gb,
                x,
                block_prefix + "." + std::to_string(num_layers),
                model_folder,
                out_channel,
                block_out_channels[i + 1],
                false  // false = stride-2 strided conv (weights are [C_out, C_in, 3, 3])
            );
        }
    }

    size_t conv_out_w = gb->mmap_weights(make_path("conv_out.weight"));
    size_t conv_out_b = gb->mmap_weights(make_path("conv_out.bias"));
    size_t out_conv = gb->conv2d(x, conv_out_w, conv_out_b, 1, 1, 1, 1);

    // out shortcut
    size_t latent_channels = 32;
    size_t group_size = block_out_channels.back() / latent_channels;

    auto shape = gb->get_output_buffer(x).shape;
    size_t N = shape[0], H = shape[2], W = shape[3];
    size_t y = gb->reshape(x, {N, latent_channels, group_size, H, W});
    y = gb->mean(y, 2);

    return gb->add(out_conv, y);
}

size_t build_autoencoder_dc_decode(CactusGraph* gb, size_t latent_input, const std::string& prefix, const std::string& model_folder) {
    size_t x = latent_input;
    const std::vector<size_t> block_out_channels = {128, 256, 512, 512, 1024, 1024};
    const std::vector<size_t> layers_per_block = {3, 3, 3, 3, 3, 3};
    size_t latent_channels = 32;

    // Use "/" as separator for path construction (supports subdirectory layout)
    std::string sep = prefix.empty() ? "" : "/";
    std::string pfx = prefix.empty() ? "" : prefix;
    auto make_path = [&](const std::string& name) {
        return model_folder + "/" + pfx + sep + name + ".weights";
    };

    size_t conv_in_w = gb->mmap_weights(make_path("conv_in.weight"));
    size_t conv_in_b = gb->mmap_weights(make_path("conv_in.bias"));

    // in shortcut
    size_t in_shortcut_repeats = block_out_channels.back() / latent_channels;
    size_t y = gb->repeat_interleave(x, in_shortcut_repeats, 1);

    x = gb->conv2d(x, conv_in_w, conv_in_b, 1, 1, 1, 1);
    x = gb->add(x, y);

    // Decoder processes from deepest (1024ch) to shallowest (128ch).
    for (int i = 5; i >= 0; --i) {
        size_t out_channel = block_out_channels[i];
        size_t num_layers = layers_per_block[i];
        std::string block_prefix = pfx + sep + "up_blocks." + std::to_string(i);

        // Upsampling block (layer 0)
        if (i < 5) {
            // All i < 5 blocks use DCUpBlock2d at layer .0
            size_t in_channel = block_out_channels[i + 1];
            x = build_dc_up_block(gb, x, block_prefix + ".0", model_folder, in_channel, out_channel);
        }

        // Processing layers
        size_t layer_start = (i == 5) ? 0 : 1;
        size_t layer_end = (i == 5) ? num_layers : num_layers + 1;
        for (size_t j = layer_start; j < layer_end; ++j) {
            std::string layer_prefix = block_prefix + "." + std::to_string(j);
            if (i >= 3) {
                size_t heads = out_channel / 32;
                x = build_efficientvit_block(gb, x, layer_prefix, model_folder, out_channel, heads);
            } else {
                x = build_res_block(gb, x, layer_prefix, model_folder);
            }
        }
    }

    // norm_out + conv_out
    x = gb->transposeN(x, {0, 2, 3, 1}); // N,C,H,W -> N,H,W,C
    auto shape_out = gb->get_output_buffer(x).shape;
    size_t No = shape_out[0], Ho = shape_out[1], Wo = shape_out[2], Co = shape_out[3];
    x = gb->reshape(x, {No * Ho * Wo, Co});
    size_t w_norm_out = gb->mmap_weights(make_path("norm_out.weight"));
    size_t b_norm_out = gb->mmap_weights(make_path("norm_out.bias"));
    x = gb->rms_norm(x, w_norm_out, 1e-5);
    x = gb->add(x, b_norm_out); // RMSNorm with bias
    x = gb->reshape(x, {No, Ho, Wo, Co});
    x = gb->transposeN(x, {0, 3, 1, 2}); // back to N,C,H,W

    x = gb->relu(x); // conv act

    size_t conv_out_w = gb->mmap_weights(make_path("conv_out.weight"));
    size_t conv_out_b = gb->mmap_weights(make_path("conv_out.bias"));
    x = gb->conv2d(x, conv_out_w, conv_out_b, 1, 1, 1, 1);

    return x;
}

std::vector<float> resize_rgb_bilinear(const uint8_t* src, int src_w, int src_h, int dst_w, int dst_h) {
    std::vector<float> out(static_cast<size_t>(dst_w) * dst_h * 3);
    
    // PixArt style: resize and center crop
    // Calculate the scale to fit the smaller dimension (so the larger dimension is cropped)
    const float ratio = std::max(static_cast<float>(dst_w) / src_w, static_cast<float>(dst_h) / src_h);
    
    // Center of target in source coordinates
    const float src_center_x = src_w / 2.0f;
    const float src_center_y = src_h / 2.0f;
    
    // Half-size of target in source coordinates
    const float src_half_w = (dst_w / ratio) / 2.0f;
    const float src_half_h = (dst_h / ratio) / 2.0f;
    
    // Top-left of target in source coordinates
    const float src_start_x = src_center_x - src_half_w;
    const float src_start_y = src_center_y - src_half_h;

    for (int y = 0; y < dst_h; ++y) {
        // Map y to source coordinates
        const float src_y = src_start_y + (static_cast<float>(y) + 0.5f) / ratio - 0.5f;
        int y0 = static_cast<int>(std::floor(src_y));
        int y1 = y0 + 1;
        const float wy = src_y - y0;
        
        // Clamp to edge
        y0 = std::max(0, std::min(y0, src_h - 1));
        y1 = std::max(0, std::min(y1, src_h - 1));

        for (int x = 0; x < dst_w; ++x) {
            // Map x to source coordinates
            const float src_x = src_start_x + (static_cast<float>(x) + 0.5f) / ratio - 0.5f;
            int x0 = static_cast<int>(std::floor(src_x));
            int x1 = x0 + 1;
            const float wx = src_x - x0;
            
            x0 = std::max(0, std::min(x0, src_w - 1));
            x1 = std::max(0, std::min(x1, src_w - 1));

            for (int c = 0; c < 3; ++c) {
                const float p00 = static_cast<float>(src[(y0 * src_w + x0) * 3 + c]);
                const float p01 = static_cast<float>(src[(y0 * src_w + x1) * 3 + c]);
                const float p10 = static_cast<float>(src[(y1 * src_w + x0) * 3 + c]);
                const float p11 = static_cast<float>(src[(y1 * src_w + x1) * 3 + c]);
                const float top = p00 + (p01 - p00) * wx;
                const float bot = p10 + (p11 - p10) * wx;
                out[(static_cast<size_t>(y) * dst_w + x) * 3 + c] = (top + (bot - top) * wy) / 255.0f;
            }
        }
    }
    return out;
}

std::vector<__fp16> rgb_to_latents(const std::vector<float>& rgb, size_t width, size_t height,
                                   size_t latent_channels, size_t lat_h, size_t lat_w) {
    const size_t total_latents = latent_channels * lat_h * lat_w;
    std::vector<__fp16> latents(total_latents, static_cast<__fp16>(0.0f));

    const size_t patch_h = height / lat_h;
    const size_t patch_w = width / lat_w;

    for (size_t ly = 0; ly < lat_h; ++ly) {
        const size_t y0 = ly * patch_h;
        const size_t y1 = std::min(height, (ly + 1) * patch_h);
        for (size_t lx = 0; lx < lat_w; ++lx) {
            const size_t x0 = lx * patch_w;
            const size_t x1 = std::min(width, (lx + 1) * patch_w);

            float r_sum = 0.0f;
            float g_sum = 0.0f;
            float b_sum = 0.0f;
            size_t count = 0;

            for (size_t y = y0; y < y1; ++y) {
                for (size_t x = x0; x < x1; ++x) {
                    const size_t idx = (y * width + x) * 3;
                    r_sum += rgb[idx];
                    g_sum += rgb[idx + 1];
                    b_sum += rgb[idx + 2];
                    ++count;
                }
            }

            if (count == 0) {
                count = 1;
            }
            float r = r_sum / static_cast<float>(count);
            float g = g_sum / static_cast<float>(count);
            float b = b_sum / static_cast<float>(count);
            r = r * 2.0f - 1.0f;
            g = g * 2.0f - 1.0f;
            b = b * 2.0f - 1.0f;
            const float lum = 0.299f * r + 0.587f * g + 0.114f * b;

            for (size_t c = 0; c < latent_channels; ++c) {
                float v = 0.0f;
                switch (c % 4) {
                    case 0: v = r; break;
                    case 1: v = g; break;
                    case 2: v = b; break;
                    default: v = lum; break;
                }
                const float band = 1.0f + 0.02f * static_cast<float>(c / 4);
                const float clipped = std::max(-1.5f, std::min(1.5f, v * band));
                latents[(c * lat_h + ly) * lat_w + lx] = static_cast<__fp16>(clipped);
            }
        }
    }

    return latents;
}

} // namespace engine
} // namespace cactus
