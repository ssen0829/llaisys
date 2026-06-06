#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t seq_len, size_t hidden) {
    for (size_t i = 0; i < seq_len; ++i) {
        const T *row = weight + static_cast<size_t>(index[i]) * hidden;
        T *dst = out + i * hidden;
        for (size_t j = 0; j < hidden; ++j) {
            dst[j] = row[j];
        }
    }
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t seq_len, size_t hidden, size_t vocab) {
    (void)vocab;
    auto *idx = reinterpret_cast<const int64_t *>(index);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), idx, reinterpret_cast<const float *>(weight), seq_len, hidden);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<fp16_t *>(out), idx, reinterpret_cast<const fp16_t *>(weight), seq_len, hidden);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<bf16_t *>(out), idx, reinterpret_cast<const bf16_t *>(weight), seq_len, hidden);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
