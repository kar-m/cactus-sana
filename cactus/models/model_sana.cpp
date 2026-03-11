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
#include <iostream>

namespace cactus {
namespace engine {

static const char* kChiPrompt =
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual "
    "descriptions suitable for image generation. Evaluate the level of detail in the user prompt:\n"
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, "
    "and spatial relationships to create vivid and concrete scenes.\n"
    "- If the prompt is already detailed, refine and enhance the existing details slightly "
    "without overcomplicating.\n"
    "Here are examples of how to transform or refine prompts:\n"
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round "
    "shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n"
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, "
    "featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a "
    "double-decker bus passing by towering glass skyscrapers.\n"
    "Please generate only the enhanced description for the prompt below and avoid including any "
    "additional commentary or evaluations:\n"
    "User Prompt: ";

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
    std::cout << "[Sana] Initializing model from: " << model_folder << std::endl;
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
    if (!text_encoder_->init(te_path, kExtendedPromptTokens, "", do_warmup)) {
        return false;
    }

    text_encoder_dim_ = text_encoder_->get_config().hidden_dim;
    // Precompute chi token count (without BOS) for correct tail_start in token selection
    chi_token_count_ = text_encoder_->get_tokenizer()->encode(std::string(kChiPrompt)).size();
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

    lat_in_node_ = gb->input({1, latent_channels_, latents_h_, latents_w_}, Precision::FP16);
    gb->register_debug_node(0, "latents", lat_in_node_);

    prompt_embeds_node_ = gb->input({1, kMaxPromptTokens, text_encoder_dim_}, Precision::FP16);
    gb->register_debug_node(0, "prompt_embeds", prompt_embeds_node_);

    timestep_node_ = gb->input({1, kTimestepDim}, Precision::FP16);

    // Check if guidance_mlp weights exist to detect Sprint vs Sana 1.0
    const std::string guidance_probe = model_folder + "/t_embedder.guidance_mlp.0.weight.weights";
    has_guidance_embeds_ = sana_weight_file_exists(guidance_probe);
    // Set default steps based on model type if not overridden by env var
    if (!std::getenv("CACTUS_SANA_STEPS")) {
        diffusion_steps_ = has_guidance_embeds_ ? 2 : 20;
    }
    if (has_guidance_embeds_) {
        guidance_node_ = gb->input({1, kTimestepDim}, Precision::FP16);
    }

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
    projected_embeds = gb->rms_norm(projected_embeds, gb->mmap_weights(model_folder + "/caption_norm.weight.weights"), 1e-5f);

    // Timestep embedder: sinusoidal -> linear_1 -> silu -> linear_2
    size_t timestep_emb = gb->matmul(timestep_node_, gb->mmap_weights(model_folder + "/t_embedder.mlp.0.weight.weights"), true);
    timestep_emb = gb->add(timestep_emb, gb->mmap_weights(model_folder + "/t_embedder.mlp.0.bias.weights"));
    timestep_emb = gb->silu(timestep_emb);
    timestep_emb = gb->matmul(timestep_emb, gb->mmap_weights(model_folder + "/t_embedder.mlp.2.weight.weights"), true);
    timestep_emb = gb->add(timestep_emb, gb->mmap_weights(model_folder + "/t_embedder.mlp.2.bias.weights"));

    // Guidance embedder (Sprint only): sinusoidal(guidance_val) -> linear_1 -> silu -> linear_2
    size_t embedded_timestep;
    if (has_guidance_embeds_) {
        size_t guidance_emb = gb->matmul(guidance_node_, gb->mmap_weights(model_folder + "/t_embedder.guidance_mlp.0.weight.weights"), true);
        guidance_emb = gb->add(guidance_emb, gb->mmap_weights(model_folder + "/t_embedder.guidance_mlp.0.bias.weights"));
        guidance_emb = gb->silu(guidance_emb);
        guidance_emb = gb->matmul(guidance_emb, gb->mmap_weights(model_folder + "/t_embedder.guidance_mlp.2.weight.weights"), true);
        guidance_emb = gb->add(guidance_emb, gb->mmap_weights(model_folder + "/t_embedder.guidance_mlp.2.bias.weights"));
        // embedded_timestep = timestep_emb + guidance_emb  [1, hidden_dim]
        embedded_timestep = gb->add(timestep_emb, guidance_emb);
    } else {
        embedded_timestep = timestep_emb;
    }

    // t_emb (6D conditioning for transformer blocks): silu(embedded_timestep) -> linear  [1, 6*hidden_dim]
    size_t t_emb = gb->silu(embedded_timestep);
    t_emb = gb->matmul(t_emb, gb->mmap_weights(model_folder + "/t_embedder.linear.weight.weights"), true);
    t_emb = gb->add(t_emb, gb->mmap_weights(model_folder + "/t_embedder.linear.bias.weights"));

    size_t hidden = gb->conv2d(
        lat_in_node_,
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

    // Encoder attention mask input: [L, kMaxPromptTokens] fp16 (0=ignore, 1=keep)
    // Filled in run_diffusion based on actual token count from encode_prompt_to_fp16.
    encoder_mask_node_ = gb->input({L, kMaxPromptTokens}, Precision::FP16);

    hidden = gb->reshape(hidden, {N, C, L});
    hidden = gb->transposeN(hidden, {0, 2, 1});
    hidden = gb->reshape(hidden, {L, C});
    gb->register_debug_node(0, "hidden_init", hidden);

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
            cross_heads, cross_head_dim,
            encoder_mask_node_
        );
    }

    // Final layer: LayerNorm then modulate with scale_shift_table + embedded_timestep
    // HF SanaModulatedNorm: shift, scale = (sst[None] + temb[:, None]).chunk(2, dim=1)
    // => sst row 0 -> shift (first chunk), sst row 1 -> scale (second chunk)
    hidden = gb->layernorm(hidden, 1e-6f);

    size_t scale_shift_table = gb->mmap_weights(model_folder + "/final_layer.scale_shift_table.weights");
    size_t shift_sst = gb->slice(scale_shift_table, 0, 0, 1);  // row 0 -> shift [1, hidden_dim]
    size_t scale_sst = gb->slice(scale_shift_table, 0, 1, 1);  // row 1 -> scale [1, hidden_dim]
    size_t shift = gb->add(shift_sst, embedded_timestep);  // [1, hidden_dim]
    size_t scale = gb->add(scale_sst, embedded_timestep);  // [1, hidden_dim]

    hidden = gb->reshape(hidden, {L, config_.hidden_dim});
    hidden = gb->multiply(hidden, gb->scalar_add(scale, 1.0f));
    hidden = gb->add(hidden, shift);
    hidden = gb->matmul(hidden, gb->mmap_weights(model_folder + "/final_layer.linear.weight.weights"), true);
    hidden = gb->add(hidden, gb->mmap_weights(model_folder + "/final_layer.linear.bias.weights"));

    hidden = gb->reshape(hidden, {1, H, W, latent_channels_});
    denoiser_output_node_ = gb->transposeN(hidden, {0, 3, 1, 2});

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
        } catch (const std::exception& e) {
            std::cerr << "[Sana] VAE encoder build failed: " << e.what() << std::endl;
            if (encoder_graph_handle_) {
                delete static_cast<CactusGraph*>(encoder_graph_handle_);
                encoder_graph_handle_ = nullptr;
            }
            has_vae_encoder_ = false;
        } catch (...) {
            std::cerr << "[Sana] VAE encoder build failed: unknown exception" << std::endl;
            if (encoder_graph_handle_) {
                delete static_cast<CactusGraph*>(encoder_graph_handle_);
                encoder_graph_handle_ = nullptr;
            }
            has_vae_encoder_ = false;
        }
    }

    // Try to load CoreML VAE decoder for ANE acceleration
    std::string mlpackage_path = model_folder + "/vae_decoder.mlpackage";
    if (npu::is_npu_available()) {
        npu_vae_decoder_ = npu::create_encoder();
        if (npu_vae_decoder_ && npu_vae_decoder_->load(mlpackage_path)) {
            use_npu_vae_decoder_ = true;
            npu_vae_decoder_->preallocate(
                {1, static_cast<int>(latent_channels_), static_cast<int>(latents_h_), static_cast<int>(latents_w_)},
                "latents", "rgb"
            );
            size_t image_h = latents_h_ * 32;
            size_t image_w = latents_w_ * 32;
            npu_vae_output_.resize(3 * image_h * image_w);
            std::cout << "[Sana] ANE VAE decoder loaded from " << mlpackage_path << std::endl;
        } else {
            use_npu_vae_decoder_ = false;
            npu_vae_decoder_.reset();
        }

        // Try to load CoreML transformer denoiser
        std::string transformer_mlpackage = model_folder + "/transformer.mlpackage";
        npu_transformer_ = npu::create_encoder();
        if (npu_transformer_ && npu_transformer_->load(transformer_mlpackage)) {
            use_npu_transformer_ = true;
            npu_transformer_output_.resize(latent_channels_ * latents_h_ * latents_w_);
            std::cout << "[Sana] ANE transformer loaded from " << transformer_mlpackage << std::endl;
        } else {
            use_npu_transformer_ = false;
            npu_transformer_.reset();
        }
    }

    initialized_ = true;

    if (!has_guidance_embeds_) {
        std::cout << "[Sana] Pre-encoding null prompt for CFG..." << std::endl;
        null_prompt_embeds_ = encode_prompt_to_fp16("");
        null_encoder_mask_data_ = encoder_mask_data_;  // mask from last encode_prompt call
    }

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

std::vector<__fp16> SanaModel::encode_prompt_to_fp16(const std::string& prompt) {
    std::cout << "[Sana] Encoding prompt: \"" << prompt << "\"" << std::endl;

    // All Sana models use complex human instruction prefix for non-empty prompts (HF default)
    const bool use_chi = !prompt.empty();
    const std::string full_prompt = use_chi ? (std::string(kChiPrompt) + prompt) : prompt;
    const size_t max_tokens = use_chi ? kExtendedPromptTokens : kMaxPromptTokens;

    auto raw_tokens = text_encoder_->get_tokenizer()->encode(full_prompt);
    std::cout << "[Sana] Tokenized into " << raw_tokens.size() << " tokens." << std::endl;

    // HF uses add_special_tokens=True which prepends BOS (token ID 2 for Gemma2)
    std::vector<uint32_t> tokens;
    tokens.push_back(text_encoder_->get_tokenizer()->get_bos_token());
    for (auto t : raw_tokens) tokens.push_back(t);

    if (tokens.size() > max_tokens) tokens.resize(max_tokens);
    const size_t full_real_len = tokens.size();
    while (tokens.size() < max_tokens)
        tokens.push_back(text_encoder_->get_tokenizer()->get_eos_token());

    auto embeds = text_encoder_->get_embeddings(tokens, false, false);
    const size_t expected_full = max_tokens * text_encoder_dim_;
    if (embeds.size() != expected_full) {
        throw std::runtime_error("Unexpected text embedding size from Gemma2 encoder.");
    }

    // For Sana 1.0: select positions [0] + [max_tokens-299 ... max_tokens-1] → 300 tokens
    // For Sprint: positions [0..299] (direct copy, max_tokens == kMaxPromptTokens)
    std::vector<__fp16> out(kMaxPromptTokens * text_encoder_dim_);
    std::vector<bool> selected_real(kMaxPromptTokens, false);

    if (!use_chi) {
        for (size_t i = 0; i < kMaxPromptTokens * text_encoder_dim_; ++i)
            out[i] = static_cast<__fp16>(embeds[i]);
        for (size_t j = 0; j < full_real_len; ++j)
            selected_real[j] = true;
    } else {
        // Position 0 (BOS) → always real
        for (size_t d = 0; d < text_encoder_dim_; ++d)
            out[d] = static_cast<__fp16>(embeds[d]);
        selected_real[0] = (0 < full_real_len);  // always true

        // Last 299 positions from the extended sequence.
        // HF formula: tail_start = num_chi_tokens_with_bos - 1 = chi_token_count_ (no BOS)
        const size_t tail_start = chi_token_count_;
        for (size_t k = 1; k < kMaxPromptTokens; ++k) {
            const size_t src_pos = tail_start + (k - 1);
            for (size_t d = 0; d < text_encoder_dim_; ++d)
                out[k * text_encoder_dim_ + d] = static_cast<__fp16>(embeds[src_pos * text_encoder_dim_ + d]);
            selected_real[k] = (src_pos < full_real_len);
        }
    }

    // Precompute cross-attention mask: [L * kMaxPromptTokens] (1=real, 0=padding)
    const size_t L = latents_h_ * latents_w_;
    const size_t S = kMaxPromptTokens;
    encoder_mask_data_.assign(L * S, __fp16(0.0f));
    for (size_t l = 0; l < L; ++l) {
        for (size_t j = 0; j < S; ++j) {
            if (selected_real[j])
                encoder_mask_data_[l * S + j] = __fp16(1.0f);
        }
    }

    return out;
}

size_t SanaModel::generate_image(const std::string& prompt, size_t width, size_t height) {
    if (!initialized_) {
        throw std::runtime_error("SanaModel not initialized");
    }
    validate_dimensions(width, height);
    std::cout << "[Sana] Starting image generation (" << width << "x" << height << ")" << std::endl;
    const size_t total_latents = latent_channels_ * latents_h_ * latents_w_;
    
    auto prompt_embeds = encode_prompt_to_fp16(prompt);

    std::cout << "[Sana] Generating initial noise..." << std::endl;
    // Initial latents: plain randn (SCMScheduler init_noise_sigma=1.0)
    auto latents = make_noise_latents(total_latents);

    if (has_guidance_embeds_) {
        run_diffusion(prompt_embeds, latents, diffusion_steps_);
    } else {
        run_diffusion_flow_euler(prompt_embeds, latents, diffusion_steps_);
    }

    return decode_latents(latents);
}

size_t SanaModel::generate_image_to_image(const std::string& prompt, const std::string& init_image_path,
                                          size_t width, size_t height, float strength) {
    if (!initialized_) {
        throw std::runtime_error("SanaModel not initialized");
    }
    validate_dimensions(width, height);

    strength = std::max(0.0f, std::min(1.0f, strength));
    const size_t total_steps = diffusion_steps_;

    std::cout << "[Sana] Starting img2img generation (" << width << "x" << height
              << ", strength=" << strength << ")" << std::endl;

    auto prompt_embeds = encode_prompt_to_fp16(prompt);

    // Encode the input image to clean latents x0 via DC-AE encoder
    auto x0 = make_image_conditioned_latents(init_image_path, width, height);

    // t_start is the step index from which to begin denoising.
    // strength=1.0 → t_start=0 (full txt2img, pure noise)
    // strength=0.0 → t_start=total_steps (no denoising, just decode encoded image)
    const size_t t_start = static_cast<size_t>(
        std::round((1.0f - strength) * static_cast<float>(total_steps)));

    if (t_start >= total_steps) {
        std::cout << "[Sana] strength≈0, skipping diffusion — decoding image latents directly." << std::endl;
        return decode_latents(x0);
    }

    auto noise = make_noise_latents(x0.size());
    std::vector<__fp16> noisy_latents(x0.size());

    if (has_guidance_embeds_) {
        // SCM (Sprint): noisy = cos(angle)·x0 + sin(angle)·ε
        static constexpr float kPiHalf = 1.5707963267948966f;
        auto scm_angle = [&](size_t idx) -> float {
            if (total_steps == 2) {
                constexpr float kAngles2[3] = {kPiHalf, 1.3f, 0.0f};
                return kAngles2[std::min(idx, size_t(2))];
            }
            return kPiHalf * (1.0f - static_cast<float>(idx) / static_cast<float>(total_steps));
        };
        const float angle_s = scm_angle(t_start);
        const float cos_s = std::cos(angle_s), sin_s = std::sin(angle_s);
        std::cout << "[Sana] Adding SCM noise at angle=" << angle_s
                  << " (cos=" << cos_s << ", sin=" << sin_s << ")" << std::endl;
        for (size_t i = 0; i < x0.size(); ++i) {
            noisy_latents[i] = static_cast<__fp16>(
                cos_s * static_cast<float>(x0[i]) + sin_s * static_cast<float>(noise[i]));
        }
        run_diffusion(prompt_embeds, noisy_latents, total_steps, t_start);
    } else {
        // Flow matching (Sana 1.0): noisy = (1-sigma)·x0 + sigma·ε
        // sigma[i] = shift * s_raw / (1 + (shift-1) * s_raw), s_raw = 1 - i/steps
        // Must match exactly what run_diffusion_flow_euler uses for sigmas[t_start]
        constexpr float kShift = 3.0f;
        const float s_raw = 1.0f - static_cast<float>(t_start) / static_cast<float>(total_steps);
        const float sigma = kShift * s_raw / (1.0f + (kShift - 1.0f) * s_raw);
        std::cout << "[Sana] Adding flow noise at sigma=" << sigma
                  << " (1-sigma=" << (1.0f - sigma) << ")" << std::endl;
        for (size_t i = 0; i < x0.size(); ++i) {
            noisy_latents[i] = static_cast<__fp16>(
                (1.0f - sigma) * static_cast<float>(x0[i]) + sigma * static_cast<float>(noise[i]));
        }
        run_diffusion_flow_euler(prompt_embeds, noisy_latents, total_steps, t_start);
    }

    return decode_latents(noisy_latents);
}

void* SanaModel::get_output_pointer(size_t encoded_node_id) const {
    if ((encoded_node_id & kDecoderNodeFlag) != 0) {
        // ANE path: return the pre-allocated output buffer
        if (use_npu_vae_decoder_ && !npu_vae_output_.empty()) {
            return const_cast<void*>(static_cast<const void*>(npu_vae_output_.data()));
        }
        // CPU path: return graph output
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
