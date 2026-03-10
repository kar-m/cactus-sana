#include "model_sana.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace cactus {
namespace engine {

namespace {

static constexpr float kPiHalf = 1.5707963267948966f; // pi/2

// Compute sinusoidal embedding matching HF's Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
// Output: [cos(t*f0), cos(t*f1), ..., cos(t*f127), sin(t*f0), ..., sin(t*f127)]
// where f_i = 10000^(-i/half_dim), half_dim = dim/2
std::vector<__fp16> make_sinusoidal_embedding_fp16(float timestep, size_t dim) {
    std::vector<__fp16> emb(dim);
    const size_t half = dim / 2;
    // HF: exponent = -log(10000) * arange(half) / half  (downscale_freq_shift=0 -> divide by half)
    const float log_freq = -std::log(10000.0f) / static_cast<float>(half);
    for (size_t i = 0; i < half; ++i) {
        const float freq = std::exp(static_cast<float>(i) * log_freq);
        emb[i]        = static_cast<__fp16>(std::cos(timestep * freq)); // cos first (flip_sin_to_cos=True)
        emb[i + half] = static_cast<__fp16>(std::sin(timestep * freq)); // sin second
    }
    return emb;
}

// Compute SCM angle schedule.
// For 2 steps: [pi/2, 1.3, 0.0] (uses intermediate_timesteps=1.3)
// For other counts: linspace(pi/2, 0, steps+1)
std::vector<float> compute_scm_angles(size_t steps) {
    std::vector<float> angles(steps + 1);
    if (steps == 2) {
        angles[0] = kPiHalf;
        angles[1] = 1.3f;
        angles[2] = 0.0f;
    } else {
        for (size_t i = 0; i <= steps; ++i) {
            angles[i] = kPiHalf * (1.0f - static_cast<float>(i) / static_cast<float>(steps));
        }
    }
    return angles;
}

} // namespace

std::vector<__fp16> SanaModel::make_noise_latents(size_t total_latents) const {
    std::vector<__fp16> noise(total_latents);
    // Allow fixed seed for reproducible testing via CACTUS_SANA_SEED env var
    uint32_t seed = 0;
    if (const char* seed_env = std::getenv("CACTUS_SANA_SEED")) {
        seed = static_cast<uint32_t>(std::strtoul(seed_env, nullptr, 10));
    } else {
        seed = std::random_device{}();
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (size_t i = 0; i < total_latents; ++i) {
        noise[i] = static_cast<__fp16>(normal(rng));
    }
    return noise;
}

size_t SanaModel::run_diffusion(const std::vector<__fp16>& prompt_embeds, std::vector<__fp16>& current_latents, size_t steps, size_t t_start) const {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (!gb) {
        throw std::runtime_error("Sana denoiser graph is not initialized.");
    }
    if (steps == 0) steps = 1;
    if (t_start >= steps) t_start = 0;
    const size_t steps_to_run = steps - t_start;
    std::cout << "[Sana] Starting SCM diffusion (" << steps_to_run << "/" << steps
              << " steps, t_start=" << t_start << ")..." << std::endl;

    gb->set_input(prompt_embeds_node_, prompt_embeds.data(), Precision::FP16);

    // Set encoder attention mask (precomputed in encode_prompt_to_fp16)
    if (!encoder_mask_data_.empty()) {
        gb->set_input(encoder_mask_node_, encoder_mask_data_.data(), Precision::FP16);
    }

    // Guidance sinusoidal: HF pipeline scales guidance by guidance_embeds_scale (0.1) before sinusoidal embedding
    // guidance = guidance_scale * guidance_embeds_scale = 4.5 * 0.1 = 0.45
    auto guidance_sinusoidal = make_sinusoidal_embedding_fp16(kGuidanceScale * kGuidanceEmbedScale, kTimestepDim);
    gb->set_input(guidance_node_, guidance_sinusoidal.data(), Precision::FP16);

    // SCM angle schedule: [pi/2, 1.3, 0.0] for 2 steps
    auto angles = compute_scm_angles(steps);

    thread_local std::mt19937 step_rng(12345);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    const size_t total = current_latents.size();
    std::vector<float> latents_f(total);
    std::vector<float> model_output_f(total);
    std::vector<float> pred_x0_f(total);
    std::vector<__fp16> lat_fp16(total);

    // Convert initial latents to float
    for (size_t j = 0; j < total; ++j) {
        latents_f[j] = static_cast<float>(current_latents[j]);
    }

    for (size_t i = t_start; i < steps; ++i) {
        const float angle_s = angles[i];       // current angle (e.g. pi/2 or 1.3)
        const float angle_t = angles[i + 1];   // next angle (e.g. 1.3 or 0)

        const float cos_s = std::cos(angle_s);
        const float sin_s = std::sin(angle_s);

        // Sana Sprint pipeline:
        //   scm_t = sin(angle) / (cos(angle) + sin(angle))  -- normalized timestep in [0,1]
        //   A = 1 / (cos(angle) + sin(angle))              -- normalization factor
        //   model_input = latents * A     (in unit-scale latent space)
        //   timestep_emb = sinusoidal(scm_t)
        //   raw_pred = transformer(model_input, scm_t)
        //   noise_pred = [(1 - 2*scm_t)*model_input + (1 - 2*scm_t + 2*scm_t^2)*raw_pred] / A
        //   pred_x0 = cos(angle)*latents - sin(angle)*noise_pred
        const float scm_t = sin_s / (cos_s + sin_s);
        const float A = 1.0f / (cos_s + sin_s);
        const float c1 = 1.0f - 2.0f * scm_t;
        const float c2 = 1.0f - 2.0f * scm_t + 2.0f * scm_t * scm_t;

        // Set model input: latents * A
        for (size_t j = 0; j < total; ++j) {
            lat_fp16[j] = static_cast<__fp16>(latents_f[j] * A);
        }
        gb->set_input(lat_in_node_, lat_fp16.data(), Precision::FP16);

        // Timestep: embed scm_t (normalized [0,1]), NOT the raw angle
        auto ts_emb = make_sinusoidal_embedding_fp16(scm_t, kTimestepDim);
        gb->set_input(timestep_node_, ts_emb.data(), Precision::FP16);

        gb->execute();

        // Retrieve raw model output
        void* raw_ptr = gb->get_output(denoiser_output_node_);
        const auto& raw_buf = gb->get_output_buffer(denoiser_output_node_);
        if (raw_buf.precision == Precision::FP16) {
            const __fp16* src = static_cast<const __fp16*>(raw_ptr);
            for (size_t j = 0; j < total; ++j) model_output_f[j] = static_cast<float>(src[j]);
        } else if (raw_buf.precision == Precision::FP32) {
            const float* src = static_cast<const float*>(raw_ptr);
            std::copy(src, src + total, model_output_f.begin());
        } else {
            throw std::runtime_error("Unexpected model output precision");
        }

        // Post-process output and compute pred_x0 (unit-scale latent space)
        for (size_t j = 0; j < total; ++j) {
            const float model_input_j = latents_f[j] * A;
            const float noise_pred_unit = (c1 * model_input_j + c2 * model_output_f[j]) / A;
            pred_x0_f[j] = cos_s * latents_f[j] - sin_s * noise_pred_unit;
        }

        if (angle_t > 1e-6f) {
            // Re-noise at next timestep: cos(t)*pred_x0 + sin(t)*N(0,1)
            const float cos_t = std::cos(angle_t);
            const float sin_t = std::sin(angle_t);
            for (size_t j = 0; j < total; ++j) {
                const float noise = normal(step_rng);
                latents_f[j] = cos_t * pred_x0_f[j] + sin_t * noise;
            }
        } else {
            // Last step: output is pred_x0 (denoised)
            for (size_t j = 0; j < total; ++j) {
                latents_f[j] = pred_x0_f[j];
            }
        }
    }

    // Write final latents back to output buffer
    for (size_t j = 0; j < total; ++j) {
        current_latents[j] = static_cast<__fp16>(latents_f[j]);
    }

    return denoiser_output_node_;
}

size_t SanaModel::run_diffusion_flow_euler(
    const std::vector<__fp16>& prompt_embeds,
    std::vector<__fp16>& current_latents,
    size_t steps,
    size_t t_start_idx) const
{
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (!gb) throw std::runtime_error("Sana denoiser graph is not initialized.");
    if (steps == 0) steps = 20;
    if (t_start_idx >= steps) t_start_idx = 0;
    const size_t steps_to_run = steps - t_start_idx;
    std::cout << "[Sana] Flow-Euler diffusion (" << steps_to_run << "/" << steps
              << " steps, t_start=" << t_start_idx << ", CFG=" << kGuidanceScale << ")..." << std::endl;

    // FlowMatchEuler schedule: sigmas = linspace(1, 0, steps+1), with shift=3.0
    // sigma_raw[i] = 1.0 - i/steps
    // sigma_shifted = shift * sigma_raw / (1 + (shift-1) * sigma_raw)
    constexpr float kShift = 3.0f;
    std::vector<float> sigmas(steps + 1);
    for (size_t i = 0; i <= steps; ++i) {
        const float s_raw = 1.0f - static_cast<float>(i) / static_cast<float>(steps);
        sigmas[i] = kShift * s_raw / (1.0f + (kShift - 1.0f) * s_raw);
    }
    sigmas[steps] = 0.0f;  // final clean image

    // Set constant inputs (shared across all steps)
    gb->set_input(encoder_mask_node_, encoder_mask_data_.data(), Precision::FP16);

    const size_t total = current_latents.size();
    std::vector<float> latents_f(total), v_cond(total), v_uncond(total);
    std::vector<__fp16> lat_fp16(total);

    for (size_t j = 0; j < total; ++j)
        latents_f[j] = static_cast<float>(current_latents[j]);

    for (size_t i = t_start_idx; i < steps; ++i) {
        const float sigma_curr = sigmas[i];
        const float sigma_next = sigmas[i + 1];
        const float dt = sigma_next - sigma_curr;  // negative (going 1->0)

        // Timestep embedding: sinusoidal(sigma * 1000)
        auto ts_emb = make_sinusoidal_embedding_fp16(sigma_curr * 1000.0f, kTimestepDim);

        // Feed current latents (unscaled for flow matching)
        for (size_t j = 0; j < total; ++j)
            lat_fp16[j] = static_cast<__fp16>(latents_f[j]);
        gb->set_input(lat_in_node_, lat_fp16.data(), Precision::FP16);
        gb->set_input(timestep_node_, ts_emb.data(), Precision::FP16);

        // === CFG: conditional pass ===
        gb->set_input(prompt_embeds_node_, prompt_embeds.data(), Precision::FP16);
        gb->execute();
        {
            void* raw_ptr = gb->get_output(denoiser_output_node_);
            const auto& buf = gb->get_output_buffer(denoiser_output_node_);
            if (buf.precision == Precision::FP16) {
                const __fp16* src = static_cast<const __fp16*>(raw_ptr);
                for (size_t j = 0; j < total; ++j) v_cond[j] = static_cast<float>(src[j]);
            } else {
                const float* src = static_cast<const float*>(raw_ptr);
                std::copy(src, src + total, v_cond.begin());
            }
        }

        // === CFG: unconditional pass ===
        gb->set_input(prompt_embeds_node_, null_prompt_embeds_.data(), Precision::FP16);
        gb->set_input(encoder_mask_node_, null_encoder_mask_data_.data(), Precision::FP16);
        gb->execute();
        {
            void* raw_ptr = gb->get_output(denoiser_output_node_);
            const auto& buf = gb->get_output_buffer(denoiser_output_node_);
            if (buf.precision == Precision::FP16) {
                const __fp16* src = static_cast<const __fp16*>(raw_ptr);
                for (size_t j = 0; j < total; ++j) v_uncond[j] = static_cast<float>(src[j]);
            } else {
                const float* src = static_cast<const float*>(raw_ptr);
                std::copy(src, src + total, v_uncond.begin());
            }
        }
        // Restore cond mask for next step
        gb->set_input(encoder_mask_node_, encoder_mask_data_.data(), Precision::FP16);

        // CFG combine + Euler step
        for (size_t j = 0; j < total; ++j) {
            const float v_cfg = v_uncond[j] + kGuidanceScale * (v_cond[j] - v_uncond[j]);
            latents_f[j] += dt * v_cfg;
        }
    }

    for (size_t j = 0; j < total; ++j)
        current_latents[j] = static_cast<__fp16>(latents_f[j]);

    return denoiser_output_node_;
}

} // namespace engine
} // namespace cactus
