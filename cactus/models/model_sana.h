#pragma once

#include "model.h"
#include "../npu/npu.h"

namespace cactus {
namespace engine {

class SanaModel : public Model {
public:
    SanaModel();
    explicit SanaModel(const Config& config);
    ~SanaModel() override;

    bool init(const std::string& model_folder, size_t context_size, const std::string& system_prompt = "", bool do_warmup = true) override;
    size_t generate_image(const std::string& prompt, size_t width, size_t height) override;
    size_t generate_image_to_image(const std::string& prompt, const std::string& init_image_path,
                                   size_t width, size_t height, float strength = 0.6f) override;
    void* get_output_pointer(size_t encoded_node_id) const;
    
protected:
    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx, ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;
    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx, ComputeBackend backend) const override;
    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

private:
    static constexpr size_t kMaxPromptTokens = 300;
    static constexpr size_t kExtendedPromptTokens = 512; // chi_prefix(208) + 300 - 2 = 506, rounded up
    static constexpr size_t kTimestepDim = 256;
    static constexpr float kSigmaData = 0.5f;
    static constexpr float kScalingFactor = 0.41407f;
    static constexpr float kGuidanceScale = 4.5f;
    static constexpr float kGuidanceEmbedScale = 0.1f;
    std::vector<__fp16> encode_prompt_to_fp16(const std::string& prompt);
    std::vector<__fp16> make_noise_latents(size_t total_latents) const;
    std::vector<__fp16> make_image_conditioned_latents(const std::string& image_path, size_t width, size_t height) const;
    // t_start: first denoising step index (0 = full txt2img; >0 for img2img partial denoising)
    size_t run_diffusion(const std::vector<__fp16>& prompt_embeds, std::vector<__fp16>& current_latents, size_t steps, size_t t_start = 0) const;
    size_t run_diffusion_flow_euler(const std::vector<__fp16>& prompt_embeds, std::vector<__fp16>& current_latents, size_t steps, size_t t_start_idx = 0) const;
    size_t decode_latents(const std::vector<__fp16>& final_latents);
    void validate_dimensions(size_t width, size_t height) const;

    static constexpr size_t kDecoderNodeFlag = (size_t(1) << (sizeof(size_t) * 8 - 1));

    std::unique_ptr<Model> text_encoder_;

    size_t prompt_embeds_node_ = 0;
    size_t encoder_mask_node_ = 0;  // cross-attention mask [L * kMaxPromptTokens] fp16
    size_t lat_in_node_ = 0;
    size_t timestep_node_ = 0;
    size_t guidance_node_ = 0;
    size_t denoiser_output_node_ = 0;
    std::vector<__fp16> encoder_mask_data_;  // precomputed mask, set from encode_prompt_to_fp16
    bool has_guidance_embeds_ = true;        // true for Sprint (has guidance_mlp weights), false for Sana 1.0
    std::vector<__fp16> null_prompt_embeds_;     // null/empty prompt embedding for CFG (Sana 1.0 only)
    std::vector<__fp16> null_encoder_mask_data_; // null attention mask for CFG

    void* decoder_graph_handle_ = nullptr;
    size_t decoder_latents_node_ = 0;
    size_t decoder_output_node_ = 0;
    void* encoder_graph_handle_ = nullptr;
    size_t encoder_image_node_ = 0;
    size_t encoder_latents_node_ = 0;
    bool has_vae_encoder_ = false;

    std::unique_ptr<npu::NPUEncoder> npu_vae_decoder_;
    bool use_npu_vae_decoder_ = false;
    std::vector<__fp16> npu_vae_output_;

    // Fused diffusion+VAE pipeline (4-step diffusion + VAE decode in one ANE call)
    std::unique_ptr<npu::NPUEncoder> npu_full_pipeline_;
    bool use_npu_full_pipeline_ = false;
    mutable std::vector<__fp16> npu_pipeline_output_;

    // Text encoder on ANE
    std::unique_ptr<npu::NPUEncoder> npu_text_encoder_;
    bool use_npu_text_encoder_ = false;

    size_t text_encoder_dim_ = 2304;
    size_t chi_token_count_ = 0;  // precomputed: encode(kChiPrompt).size() (no BOS) = tail_start for token selection
    size_t latent_channels_ = 32;
    size_t latents_h_ = 32;
    size_t latents_w_ = 32;
    size_t diffusion_steps_ = 2;
};

}
}
