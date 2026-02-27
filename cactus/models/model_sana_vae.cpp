#include "model_sana.h"
#include "sana_vae_ops.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace cactus {
namespace engine {

std::vector<__fp16> SanaModel::make_image_conditioned_latents(const std::string& image_path, size_t width, size_t height) const {
    int src_w = 0;
    int src_h = 0;
    int src_c = 0;
    uint8_t* image = stbi_load(image_path.c_str(), &src_w, &src_h, &src_c, 3);
    if (!image) {
        throw std::runtime_error("Failed to read init image for img2img: " + image_path);
    }

    auto resized = resize_rgb_bilinear(image, src_w, src_h, static_cast<int>(width), static_cast<int>(height));
    stbi_image_free(image);

    if (has_vae_encoder_ && encoder_graph_handle_) {
        std::vector<__fp16> image_nchw(3 * width * height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const size_t hw = y * width + x;
                const size_t rgb_idx = hw * 3;
                image_nchw[0 * width * height + hw] = static_cast<__fp16>(resized[rgb_idx] * 2.0f - 1.0f);
                image_nchw[1 * width * height + hw] = static_cast<__fp16>(resized[rgb_idx + 1] * 2.0f - 1.0f);
                image_nchw[2 * width * height + hw] = static_cast<__fp16>(resized[rgb_idx + 2] * 2.0f - 1.0f);
            }
        }

        auto* encoder = static_cast<CactusGraph*>(encoder_graph_handle_);
        encoder->set_input(encoder_image_node_, image_nchw.data(), Precision::FP16);
        encoder->execute();

        const auto& out_buf = encoder->get_output_buffer(encoder_latents_node_);
        void* out_ptr = encoder->get_output(encoder_latents_node_);
        const size_t expected = latent_channels_ * latents_h_ * latents_w_;
        if (out_buf.total_size != expected) {
            throw std::runtime_error("Unexpected VAE encoder output size for Sana img2img.");
        }

        std::vector<__fp16> latents(expected);
        if (out_buf.precision == Precision::FP16) {
            std::memcpy(latents.data(), out_ptr, expected * sizeof(__fp16));
        } else if (out_buf.precision == Precision::FP32) {
            const float* src = static_cast<const float*>(out_ptr);
            for (size_t i = 0; i < expected; ++i) {
                latents[i] = static_cast<__fp16>(src[i]);
            }
        } else {
            const int8_t* src = static_cast<const int8_t*>(out_ptr);
            for (size_t i = 0; i < expected; ++i) {
                latents[i] = static_cast<__fp16>(src[i]);
            }
        }
        return latents;
    }

    return rgb_to_latents(resized, width, height, latent_channels_, latents_h_, latents_w_);
}

size_t SanaModel::decode_latents(const std::vector<__fp16>& final_latents) {
    auto* decoder = static_cast<CactusGraph*>(decoder_graph_handle_);
    if (!decoder) {
        throw std::runtime_error("Sana decoder graph is not initialized.");
    }

    decoder->set_input(decoder_latents_node_, final_latents.data(), Precision::FP16);
    decoder->execute();
    return kDecoderNodeFlag | decoder_output_node_;
}

} // namespace engine
} // namespace cactus
