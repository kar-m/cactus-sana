#include "model_sana.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace cactus {
namespace engine {

namespace {

std::vector<__fp16> make_sinusoidal_embedding_fp16(float timestep, size_t dim) {
    std::vector<__fp16> emb(dim);
    const size_t half = dim / 2;
    const float log_freq = -std::log(10000.0f) / static_cast<float>(half - 1);
    for (size_t i = 0; i < half; ++i) {
        const float freq = std::exp(static_cast<float>(i) * log_freq);
        emb[i] = static_cast<__fp16>(std::sin(timestep * freq));
        emb[i + half] = static_cast<__fp16>(std::cos(timestep * freq));
    }
    return emb;
}

} // namespace

std::vector<__fp16> SanaModel::make_noise_latents(size_t total_latents) const {
    std::vector<__fp16> noise(total_latents);
    thread_local std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (size_t i = 0; i < total_latents; ++i) {
        noise[i] = static_cast<__fp16>(normal(rng));
    }
    return noise;
}

size_t SanaModel::run_diffusion(const std::vector<__fp16>& prompt_embeds, std::vector<__fp16>& current_latents, size_t steps) const {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (!gb) {
        throw std::runtime_error("Sana denoiser graph is not initialized.");
    }
    if (steps == 0) {
        steps = 1;
    }

    gb->set_input(prompt_embeds_node_, prompt_embeds.data(), Precision::FP16);

    std::vector<float> sigmas(steps + 1);
    for (size_t i = 0; i <= steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
        sigmas[i] = (kFlowShift * t) / (1.0f + (kFlowShift - 1.0f) * t);
    }

    std::vector<std::vector<__fp16>> timestep_embeds(steps);
    std::vector<__fp16> dt_values(steps);
    for (size_t i = 0; i < steps; ++i) {
        const float s_curr = sigmas[steps - i];
        const float s_next = sigmas[steps - i - 1];
        timestep_embeds[i] = make_sinusoidal_embedding_fp16(s_curr * 1000.0f, kTimestepDim);
        dt_values[i] = static_cast<__fp16>(s_next - s_curr);
    }

    for (size_t i = 0; i < steps; ++i) {
        gb->set_input(latents_node_, current_latents.data(), Precision::FP16);
        gb->set_input(timestep_node_, timestep_embeds[i].data(), Precision::FP16);
        gb->set_input(dt_node_, &dt_values[i], Precision::FP16);
        gb->execute();
        const auto& next_buf = gb->get_output_buffer(next_latents_node_);
        void* next_z_ptr = gb->get_output(next_latents_node_);

        if (next_buf.precision == Precision::FP16) {
            const __fp16* src = static_cast<const __fp16*>(next_z_ptr);
            for (size_t j = 0; j < current_latents.size(); ++j) {
                float v = static_cast<float>(src[j]);
                if (!std::isfinite(v)) {
                    v = 0.0f;
                } else {
                    v = std::max(-16.0f, std::min(16.0f, v));
                }
                current_latents[j] = static_cast<__fp16>(v);
            }
        } else if (next_buf.precision == Precision::FP32) {
            const float* src = static_cast<const float*>(next_z_ptr);
            for (size_t j = 0; j < current_latents.size(); ++j) {
                float v = src[j];
                if (!std::isfinite(v)) {
                    v = 0.0f;
                } else {
                    v = std::max(-16.0f, std::min(16.0f, v));
                }
                current_latents[j] = static_cast<__fp16>(v);
            }
        } else {
            const int8_t* src = static_cast<const int8_t*>(next_z_ptr);
            for (size_t j = 0; j < current_latents.size(); ++j) {
                current_latents[j] = static_cast<__fp16>(src[j]);
            }
        }
    }

    return next_latents_node_;
}

} // namespace engine
} // namespace cactus
