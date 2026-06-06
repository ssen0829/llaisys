#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t n) {
    size_t best_idx = 0;
    float best_val = utils::cast<float>(vals[0]);
    for (size_t i = 1; i < n; ++i) {
        float v = utils::cast<float>(vals[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    max_idx[0] = static_cast<int64_t>(best_idx);
    max_val[0] = utils::cast<T>(best_val);
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t dtype, size_t n) {
    auto *idx = reinterpret_cast<int64_t *>(max_idx);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_(idx, reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), n);
    case LLAISYS_DTYPE_F16:
        return argmax_(idx, reinterpret_cast<fp16_t *>(max_val), reinterpret_cast<const fp16_t *>(vals), n);
    case LLAISYS_DTYPE_BF16:
        return argmax_(idx, reinterpret_cast<bf16_t *>(max_val), reinterpret_cast<const bf16_t *>(vals), n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
