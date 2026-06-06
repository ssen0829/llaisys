#include "linear_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t t = 0; t < k; ++t) {
                sum += utils::cast<float>(in[i * k + t]) * utils::cast<float>(weight[j * k + t]);
            }
            if (bias) {
                sum += utils::cast<float>(bias[j]);
            }
            out[i * n + j] = utils::cast<T>(sum);
        }
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t m, size_t n, size_t k) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       bias ? reinterpret_cast<const float *>(bias) : nullptr, m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const fp16_t *>(in),
                       reinterpret_cast<const fp16_t *>(weight),
                       bias ? reinterpret_cast<const fp16_t *>(bias) : nullptr, m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(in),
                       reinterpret_cast<const bf16_t *>(weight),
                       bias ? reinterpret_cast<const bf16_t *>(bias) : nullptr, m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
