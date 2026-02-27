#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class CactusGraph;

namespace cactus {
namespace engine {

bool sana_weight_file_exists(const std::string& path);

size_t build_autoencoder_dc_encode(
    CactusGraph* gb,
    size_t input,
    const std::string& prefix,
    const std::string& model_folder
);

size_t build_autoencoder_dc_decode(
    CactusGraph* gb,
    size_t latent_input,
    const std::string& prefix,
    const std::string& model_folder
);

std::vector<float> resize_rgb_bilinear(
    const uint8_t* src,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h
);

std::vector<__fp16> rgb_to_latents(
    const std::vector<float>& rgb,
    size_t width,
    size_t height,
    size_t latent_channels,
    size_t lat_h,
    size_t lat_w
);

} // namespace engine
} // namespace cactus
