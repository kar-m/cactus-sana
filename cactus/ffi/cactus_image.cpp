#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "../models/model_sana.h"
#include <chrono>
#include <cstring>
#include <sstream>
#include <iomanip>

namespace {

// Portable IEEE 754 half-precision to single-precision conversion
static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t frac = h & 0x3ffu;
    uint32_t f32;
    if (exp == 0u) {
        f32 = sign << 31;  // zero (treat subnormals as zero for images)
    } else if (exp == 31u) {
        f32 = (sign << 31) | (0xffu << 23) | (frac << 13);  // inf / nan
    } else {
        f32 = (sign << 31) | ((exp + 112u) << 23) | (frac << 13);
    }
    float result;
    std::memcpy(&result, &f32, sizeof(float));
    return result;
}

// Convert NCHW fp16 VAE output [-1,1] to packed RGB uint8 HWC and store in handle.
static void store_image_pixels(CactusModelHandle* handle, void* raw_ptr,
                               size_t width, size_t height) {
    const size_t hw = width * height;
    handle->last_image_pixels.resize(3 * hw);
    handle->last_image_width  = width;
    handle->last_image_height = height;

    const uint16_t* fp16 = static_cast<const uint16_t*>(raw_ptr);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            for (size_t c = 0; c < 3; ++c) {
                float f = fp16_to_float(fp16[c * hw + y * width + x]);
                f = (f + 1.0f) * 0.5f;
                if (f < 0.0f) f = 0.0f;
                if (f > 1.0f) f = 1.0f;
                handle->last_image_pixels[(y * width + x) * 3 + c] =
                    static_cast<uint8_t>(f * 255.0f + 0.5f);
            }
        }
    }
}

} // namespace

using namespace cactus::engine;
using namespace cactus::ffi;

extern "C" {

int cactus_generate_image(
    cactus_model_t model,
    const char* prompt,
    size_t width,
    size_t height,
    char* response_buffer,
    size_t buffer_size
) {
    if (!model) {
        std::string error_msg = last_error_message.empty() ?
            "Model not initialized." : last_error_message;
        CACTUS_LOG_ERROR("generate_image", error_msg);
        handle_error_response(error_msg, response_buffer, buffer_size);
        return -1;
    }

    if (!prompt || !response_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("generate_image", "Invalid parameters");
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto* handle = static_cast<CactusModelHandle*>(model);

        CACTUS_LOG_INFO("generate_image", "Generating image: prompt=\"" << prompt
                        << "\", width=" << width << ", height=" << height);

        size_t output_node = handle->model->generate_image(
            std::string(prompt), width, height
        );

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count() / 1000.0;

        CACTUS_LOG_INFO("generate_image", "Image generation completed in "
                        << std::fixed << std::setprecision(2) << total_time_ms << "ms");

        // Extract and store pixels for cactus_get_last_image_pixels_rgb
        void* raw_ptr = cactus_get_output(model, output_node);
        if (raw_ptr) {
            store_image_pixels(handle, raw_ptr, width, height);
        }

        std::ostringstream json;
        json << "{";
        json << "\"success\":true,";
        json << "\"error\":null,";
        json << "\"output_node\":" << output_node << ",";
        json << "\"width\":" << width << ",";
        json << "\"height\":" << height << ",";
        json << "\"total_time_ms\":" << std::fixed << std::setprecision(2) << total_time_ms << ",";
        json << "\"ram_usage_mb\":" << std::fixed << std::setprecision(2) << get_ram_usage_mb();
        json << "}";

        std::string result = json.str();
        if (result.length() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, result.c_str());
        return static_cast<int>(result.length());

    } catch (const std::exception& e) {
        CACTUS_LOG_ERROR("generate_image", "Exception: " << e.what());
        handle_error_response(e.what(), response_buffer, buffer_size);
        return -1;
    } catch (...) {
        CACTUS_LOG_ERROR("generate_image", "Unknown exception during image generation");
        handle_error_response("Unknown error during image generation", response_buffer, buffer_size);
        return -1;
    }
}

int cactus_generate_image_to_image(
    cactus_model_t model,
    const char* prompt,
    const char* init_image_path,
    size_t width,
    size_t height,
    float strength,
    char* response_buffer,
    size_t buffer_size
) {
    if (!model) {
        std::string error_msg = last_error_message.empty() ?
            "Model not initialized." : last_error_message;
        CACTUS_LOG_ERROR("generate_image_to_image", error_msg);
        handle_error_response(error_msg, response_buffer, buffer_size);
        return -1;
    }

    if (!prompt || !init_image_path || !response_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("generate_image_to_image", "Invalid parameters");
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* sana = dynamic_cast<SanaModel*>(handle->model.get());
        if (!sana) {
            handle_error_response("Image-to-image is only supported for Sana models", response_buffer, buffer_size);
            return -1;
        }

        CACTUS_LOG_INFO("generate_image_to_image", "Generating image-to-image: prompt=\"" << prompt
                        << "\", init_image=\"" << init_image_path << "\", width=" << width
                        << ", height=" << height << ", strength=" << strength);

        size_t output_node = sana->generate_image_to_image(
            std::string(prompt),
            std::string(init_image_path),
            width,
            height,
            strength
        );

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count() / 1000.0;

        // Extract and store pixels for cactus_get_last_image_pixels_rgb
        void* raw_ptr = cactus_get_output(model, output_node);
        if (raw_ptr) {
            store_image_pixels(handle, raw_ptr, width, height);
        }

        std::ostringstream json;
        json << "{";
        json << "\"success\":true,";
        json << "\"error\":null,";
        json << "\"output_node\":" << output_node << ",";
        json << "\"width\":" << width << ",";
        json << "\"height\":" << height << ",";
        json << "\"strength\":" << std::fixed << std::setprecision(3) << strength << ",";
        json << "\"total_time_ms\":" << std::fixed << std::setprecision(2) << total_time_ms << ",";
        json << "\"ram_usage_mb\":" << std::fixed << std::setprecision(2) << get_ram_usage_mb();
        json << "}";

        std::string result = json.str();
        if (result.length() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, result.c_str());
        return static_cast<int>(result.length());

    } catch (const std::exception& e) {
        CACTUS_LOG_ERROR("generate_image_to_image", "Exception: " << e.what());
        handle_error_response(e.what(), response_buffer, buffer_size);
        return -1;
    } catch (...) {
        CACTUS_LOG_ERROR("generate_image_to_image", "Unknown exception during image generation");
        handle_error_response("Unknown error during image generation", response_buffer, buffer_size);
        return -1;
    }
}

int cactus_get_last_image_pixels_rgb(
    cactus_model_t model,
    uint8_t* out_buffer,
    size_t buffer_size,
    size_t* out_width,
    size_t* out_height
) {
    if (!model || !out_width || !out_height) return -1;
    auto* handle = static_cast<CactusModelHandle*>(model);
    if (handle->last_image_pixels.empty()) return -1;

    *out_width  = handle->last_image_width;
    *out_height = handle->last_image_height;
    const size_t needed = handle->last_image_pixels.size();

    // Allow null buffer to query required size
    if (!out_buffer) return static_cast<int>(needed);
    if (buffer_size < needed) return -1;

    std::memcpy(out_buffer, handle->last_image_pixels.data(), needed);
    return static_cast<int>(needed);
}

void* cactus_get_output(
    cactus_model_t model,
    size_t node_id
) {
    if (!model) return nullptr;
    auto* handle = static_cast<CactusModelHandle*>(model);
    if (!handle->model) {
        CACTUS_LOG_ERROR("get_output", "Model or graph handle is null");
        return nullptr;
    }
    try {
        void* ptr = nullptr;
        if (auto* sana = dynamic_cast<SanaModel*>(handle->model.get())) {
            ptr = sana->get_output_pointer(node_id);
        } else {
            if (!handle->model->graph_handle_) {
                CACTUS_LOG_ERROR("get_output", "Model graph handle is null");
                return nullptr;
            }
            auto* gb = static_cast<CactusGraph*>(handle->model->graph_handle_);
            ptr = gb->get_output(node_id);
        }
        if (!ptr) {
            CACTUS_LOG_ERROR("get_output", "Graph returned null for node " << node_id);
        }
        return ptr;
    } catch (const std::exception& e) {
        CACTUS_LOG_ERROR("get_output", "Exception: " << e.what());
        return nullptr;
    } catch (...) {
        CACTUS_LOG_ERROR("get_output", "Unknown exception");
        return nullptr;
    }
}

}
