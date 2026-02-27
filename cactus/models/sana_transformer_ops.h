#pragma once

#include <cstddef>
#include <string>

class CactusGraph;

namespace cactus {
namespace engine {

size_t build_sana_transformer_block(
    CactusGraph* gb,
    size_t hidden,
    size_t enc_hidden,
    size_t timestep_embedded,
    const std::string& prefix,
    const std::string& model_folder,
    size_t dim,
    size_t height,
    size_t width,
    size_t num_heads,
    size_t dim_head,
    size_t num_cross_heads,
    size_t cross_dim_head
);

} // namespace engine
} // namespace cactus
