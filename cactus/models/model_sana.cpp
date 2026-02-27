#include "model_sana.h"
#include "model.h"
#include "sana_transformer_ops.h"
#include "sana_vae_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace cactus {
namespace engine {

SanaModel::SanaModel() : Model() {}

SanaModel::SanaModel(const Config& config) : Model(config) {}

SanaModel::~SanaModel() {
    if (decoder_graph_handle_) {
        delete static_cast<CactusGraph*>(decoder_graph_handle_);
        decoder_graph_handle_ = nullptr;
    }
    if (encoder_graph_handle_) {
        delete static_cast<CactusGraph*>(encoder_graph_handle_);
        encoder_graph_handle_ = nullptr;
    }
}

bool SanaModel::init(const std::string& model_folder, size_t context_size, const std::string& system_prompt, bool do_warmup) {
    (void)context_size;
    (void)system_prompt;
    if (initialized_) {
        return true;
    }

    auto* gb = new CactusGraph();
    graph_handle_ = gb;
    owns_graph_ = true;
    model_folder_path_ = model_folder;

    if (!config_.from_json(model_folder + "/config.txt")) {
        return false;
    }

    std::string te_path = model_folder + "/text_encoder";
    text_encoder_ = create_model(te_path);
    if (!text_encoder_) {
        return false;
    }
    if (!text_encoder_->init(te_path, kMaxPromptTokens, "", do_warmup)) {
        return false;
    }

    text_encoder_dim_ = text_encoder_->get_config().hidden_dim;
    latent_channels_ = 32;

    size_t image_width = 1024;
    size_t image_height = 1024;
    if (const char* size_env = std::getenv("CACTUS_SANA_IMAGE_SIZE")) {
        const long parsed = std::strtol(size_env, nullptr, 10);
        if (parsed > 0 && parsed % 32 == 0) {
            image_width = static_cast<size_t>(parsed);
            image_height = static_cast<size_t>(parsed);
        }
    }
    if (const char* width_env = std::getenv("CACTUS_SANA_IMAGE_WIDTH")) {
        const long parsed = std::strtol(width_env, nullptr, 10);
        if (parsed > 0 && parsed % 32 == 0) {
            image_width = static_cast<size_t>(parsed);
        }
    }
    if (const char* height_env = std::getenv("CACTUS_SANA_IMAGE_HEIGHT")) {
        const long parsed = std::strtol(height_env, nullptr, 10);
        if (parsed > 0 && parsed % 32 == 0) {
            image_height = static_cast<size_t>(parsed);
        }
    }

    latents_h_ = image_height / 32;
    latents_w_ = image_width / 32;
    if (const char* steps_env = std::getenv("CACTUS_SANA_STEPS")) {
        const long parsed = std::strtol(steps_env, nullptr, 10);
        if (parsed > 0 && parsed <= 128) {
            diffusion_steps_ = static_cast<size_t>(parsed);
        }
    }

    latents_node_ = gb->input({1, latent_channels_, latents_h_, latents_w_}, Precision::FP16);
    prompt_embeds_node_ = gb->input({1, kMaxPromptTokens, text_encoder_dim_}, Precision::FP16);
    timestep_node_ = gb->input({1, kTimestepDim}, Precision::FP16);
    dt_node_ = gb->input({1}, Precision::FP16);

    size_t cp_w1 = gb->mmap_weights(model_folder + "/caption_projection.linear_1.weight.weights");
    size_t cp_b1 = gb->mmap_weights(model_folder + "/caption_projection.linear_1.bias.weights");
    size_t cp_w2 = gb->mmap_weights(model_folder + "/caption_projection.linear_2.weight.weights");
    size_t cp_b2 = gb->mmap_weights(model_folder + "/caption_projection.linear_2.bias.weights");

    size_t proj_2d = gb->reshape(prompt_embeds_node_, {kMaxPromptTokens, text_encoder_dim_});
    size_t projected_embeds = gb->matmul(proj_2d, cp_w1, true);
    projected_embeds = gb->add(projected_embeds, cp_b1);
    projected_embeds = gb->gelu(projected_embeds);
    projected_embeds = gb->matmul(projected_embeds, cp_w2, true);
    projected_embeds = gb->add(projected_embeds, cp_b2);
    projected_embeds = gb->reshape(projected_embeds, {kMaxPromptTokens, config_.hidden_dim});
    projected_embeds = gb->rms_norm(projected_embeds, gb->mmap_weights(model_folder + "/caption_norm.weight.weights"), 1e-6f);

    size_t t_emb = gb->matmul(timestep_node_, gb->mmap_weights(model_folder + "/t_embedder.mlp.0.weight.weights"), true);
    t_emb = gb->add(t_emb, gb->mmap_weights(model_folder + "/t_embedder.mlp.0.bias.weights"));
    t_emb = gb->silu(t_emb);
    t_emb = gb->matmul(t_emb, gb->mmap_weights(model_folder + "/t_embedder.mlp.2.weight.weights"), true);
    t_emb = gb->add(t_emb, gb->mmap_weights(model_folder + "/t_embedder.mlp.2.bias.weights"));
    t_emb = gb->silu(t_emb);
    t_emb = gb->matmul(t_emb, gb->mmap_weights(model_folder + "/t_embedder.linear.weight.weights"), true);
    t_emb = gb->add(t_emb, gb->mmap_weights(model_folder + "/t_embedder.linear.bias.weights"));

    size_t hidden = gb->conv2d(
        latents_node_,
        gb->mmap_weights(model_folder + "/pos_embed.proj.weight.weights"),
        gb->mmap_weights(model_folder + "/pos_embed.proj.bias.weights"),
        1, 1, 0, 0
    );

    auto h_shape = gb->get_output_buffer(hidden).shape;
    size_t N = h_shape[0];
    size_t C = h_shape[1];
    size_t H = h_shape[2];
    size_t W = h_shape[3];
    size_t L = H * W;

    hidden = gb->reshape(hidden, {N, C, L});
    hidden = gb->transposeN(hidden, {0, 2, 1});
    hidden = gb->reshape(hidden, {L, C});

    const size_t self_heads = config_.attention_heads;
    const size_t self_head_dim = config_.attention_head_dim;
    const size_t cross_heads = config_.num_cross_attention_heads > 0
        ? config_.num_cross_attention_heads
        : config_.attention_heads;
    const size_t cross_head_dim = config_.cross_attention_head_dim > 0
        ? config_.cross_attention_head_dim
        : config_.attention_head_dim;

    for (uint32_t i = 0; i < config_.num_layers; ++i) {
        hidden = build_sana_transformer_block(
            gb, hidden, projected_embeds, t_emb,
            "blocks." + std::to_string(i), model_folder, config_.hidden_dim, H, W,
            self_heads, self_head_dim,
            cross_heads, cross_head_dim
        );
    }

    size_t scale_shift_table = gb->mmap_weights(model_folder + "/final_layer.scale_shift_table.weights");
    size_t t_res_final = gb->reshape(t_emb, {1, 6, config_.hidden_dim});
    size_t t_first2 = gb->slice(t_res_final, 1, 0, 2);
    size_t mod = gb->add(t_first2, scale_shift_table);
    size_t shift = gb->reshape(gb->slice(mod, 1, 0, 1), {1, config_.hidden_dim});
    size_t scale = gb->reshape(gb->slice(mod, 1, 1, 1), {1, config_.hidden_dim});

    hidden = gb->reshape(hidden, {1, H, W, config_.hidden_dim});
    hidden = gb->multiply(hidden, gb->scalar_add(scale, 1.0f));
    hidden = gb->add(hidden, shift);
    hidden = gb->reshape(hidden, {L, config_.hidden_dim});
    hidden = gb->matmul(hidden, gb->mmap_weights(model_folder + "/final_layer.linear.weight.weights"), true);
    hidden = gb->add(hidden, gb->mmap_weights(model_folder + "/final_layer.linear.bias.weights"));

    hidden = gb->reshape(hidden, {1, H, W, latent_channels_});
    denoiser_output_node_ = gb->transposeN(hidden, {0, 3, 1, 2});
    next_latents_node_ = gb->add(latents_node_, gb->multiply(denoiser_output_node_, dt_node_));

    auto* decoder_gb = new CactusGraph();
    decoder_graph_handle_ = decoder_gb;
    decoder_latents_node_ = decoder_gb->input({1, latent_channels_, latents_h_, latents_w_}, Precision::FP16);
    decoder_output_node_ = build_autoencoder_dc_decode(decoder_gb, decoder_latents_node_, "vae", model_folder);

    const std::string vae_encoder_probe = model_folder + "/vae_encoder/conv_in.weight.weights";
    if (sana_weight_file_exists(vae_encoder_probe)) {
        try {
            auto* encoder_gb = new CactusGraph();
            encoder_graph_handle_ = encoder_gb;
            encoder_image_node_ = encoder_gb->input({1, 3, latents_h_ * 32, latents_w_ * 32}, Precision::FP16);
            encoder_latents_node_ = build_autoencoder_dc_encode(encoder_gb, encoder_image_node_, "vae_encoder", model_folder);
            has_vae_encoder_ = true;
        } catch (...) {
            if (encoder_graph_handle_) {
                delete static_cast<CactusGraph*>(encoder_graph_handle_);
                encoder_graph_handle_ = nullptr;
            }
            has_vae_encoder_ = false;
        }
    }

    initialized_ = true;
    return true;
}

void SanaModel::validate_dimensions(size_t width, size_t height) const {
    if (width % 32 != 0 || height % 32 != 0) {
        throw std::runtime_error("SanaModel requires width and height divisible by 32.");
    }
    const size_t expected_w = latents_w_ * 32;
    const size_t expected_h = latents_h_ * 32;
    if (width != expected_w || height != expected_h) {
        throw std::runtime_error(
            "SanaModel currently uses a fixed latent grid; expected " +
            std::to_string(expected_w) + "x" + std::to_string(expected_h) + "."
        );
    }
}

std::vector<__fp16> SanaModel::encode_prompt_to_fp16(const std::string& prompt) const {
    auto tokens = text_encoder_->get_tokenizer()->encode(prompt);
    if (tokens.size() > kMaxPromptTokens) {
        tokens.resize(kMaxPromptTokens);
    }
    while (tokens.size() < kMaxPromptTokens) {
        tokens.push_back(text_encoder_->get_tokenizer()->get_eos_token());
    }

    auto embeds = text_encoder_->get_embeddings(tokens, false, false);
    const size_t expected = kMaxPromptTokens * text_encoder_dim_;
    if (embeds.size() != expected) {
        throw std::runtime_error("Unexpected text embedding size from Gemma2 encoder.");
    }

    std::vector<__fp16> out(expected);
    for (size_t i = 0; i < expected; ++i) {
        out[i] = static_cast<__fp16>(embeds[i]);
    }
    return out;
}

size_t SanaModel::generate_image(const std::string& prompt, size_t width, size_t height) {
    if (!initialized_) {
        throw std::runtime_error("SanaModel not initialized");
    }
    validate_dimensions(width, height);

    const size_t total_latents = latent_channels_ * latents_h_ * latents_w_;
    auto prompt_embeds = encode_prompt_to_fp16(prompt);
    auto latents = make_noise_latents(total_latents);
    run_diffusion(prompt_embeds, latents, diffusion_steps_);
    return decode_latents(latents);
}

size_t SanaModel::generate_image_to_image(const std::string& prompt, const std::string& init_image_path,
                                          size_t width, size_t height, float strength) {
    if (!initialized_) {
        throw std::runtime_error("SanaModel not initialized");
    }
    validate_dimensions(width, height);

    strength = std::max(0.0f, std::min(1.0f, strength));

    auto prompt_embeds = encode_prompt_to_fp16(prompt);
    auto base_latents = make_image_conditioned_latents(init_image_path, width, height);
    auto noise_latents = make_noise_latents(base_latents.size());

    const float keep = 1.0f - strength;
    for (size_t i = 0; i < base_latents.size(); ++i) {
        const float mixed = keep * static_cast<float>(base_latents[i]) +
                            strength * static_cast<float>(noise_latents[i]);
        base_latents[i] = static_cast<__fp16>(mixed);
    }

    const size_t steps = std::max<size_t>(
        1, static_cast<size_t>(std::round(static_cast<float>(diffusion_steps_) * (0.25f + 0.75f * strength))));
    run_diffusion(prompt_embeds, base_latents, steps);
    return decode_latents(base_latents);
}

void* SanaModel::get_output_pointer(size_t encoded_node_id) const {
    if ((encoded_node_id & kDecoderNodeFlag) != 0) {
        auto* decoder = static_cast<CactusGraph*>(decoder_graph_handle_);
        if (!decoder) {
            return nullptr;
        }
        const size_t node_id = encoded_node_id & ~kDecoderNodeFlag;
        return decoder->get_output(node_id);
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (!gb) {
        return nullptr;
    }
    return gb->get_output(encoded_node_id);
}

void SanaModel::load_weights_to_graph(CactusGraph* gb) {
    (void)gb;
}

size_t SanaModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    (void)tokens;
    (void)use_cache;
    throw std::runtime_error("SanaModel::forward is not used for diffusion models.");
}

size_t SanaModel::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset) {
    (void)gb;
    (void)normalized_input;
    (void)layer_idx;
    (void)backend;
    (void)use_cache;
    (void)position_offset;
    throw std::runtime_error("SanaModel::build_attention not implemented");
}

size_t SanaModel::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx, ComputeBackend backend) const {
    (void)gb;
    (void)normalized_h;
    (void)layer_idx;
    (void)backend;
    throw std::runtime_error("SanaModel::build_mlp not implemented");
}

size_t SanaModel::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset) {
    (void)gb;
    (void)hidden;
    (void)layer_idx;
    (void)backend;
    (void)use_cache;
    (void)position_offset;
    throw std::runtime_error("SanaModel::build_transformer_block not implemented");
}

} // namespace engine
} // namespace cactus
