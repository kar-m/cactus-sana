#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "../models/model_sana.h"
#include <chrono>
#include <cstring>
#include <sstream>
#include <iomanip>

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
