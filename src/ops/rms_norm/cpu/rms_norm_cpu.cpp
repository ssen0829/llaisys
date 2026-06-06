#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        float sum_sq = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float x = utils::cast<float>(in[r * cols + c]);
            sum_sq += x * x;
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);
        for (size_t c = 0; c < cols; ++c) {
            float x = utils::cast<float>(in[r * cols + c]);
            float w = utils::cast<float>(weight[c]);
            out[r * cols + c] = utils::cast<T>(x * inv_rms * w);
        }
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t dtype, size_t rows, size_t cols) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), eps, rows, cols);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const fp16_t *>(in),
                         reinterpret_cast<const fp16_t *>(weight), eps, rows, cols);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(in),
                         reinterpret_cast<const bf16_t *>(weight), eps, rows, cols);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
