#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta,
           size_t seqlen, size_t nhead, size_t head_dim) {
    const size_t half = head_dim / 2;
    for (size_t i = 0; i < seqlen; ++i) {
        const float pos = static_cast<float>(pos_ids[i]);
        for (size_t h = 0; h < nhead; ++h) {
            const size_t base = (i * nhead + h) * head_dim;
            for (size_t j = 0; j < half; ++j) {
                const float phi = pos / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
                const float cos_p = std::cos(phi);
                const float sin_p = std::sin(phi);
                const float a = utils::cast<float>(in[base + j]);
                const float b = utils::cast<float>(in[base + half + j]);
                out[base + j] = utils::cast<T>(a * cos_p - b * sin_p);
                out[base + half + j] = utils::cast<T>(b * cos_p + a * sin_p);
            }
        }
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          llaisysDataType_t dtype, size_t seqlen, size_t nhead, size_t head_dim) {
    auto *pos = reinterpret_cast<const int64_t *>(pos_ids);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pos, theta, seqlen, nhead, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const fp16_t *>(in), pos, theta, seqlen, nhead, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(in), pos, theta, seqlen, nhead, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
