#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale,
                     size_t qlen, size_t kvlen, size_t nhead, size_t nkvh, size_t hd) {
    const size_t q_per_kv = nhead / nkvh;
    std::vector<float> scores(kvlen);
    std::vector<float> probs(kvlen);

    for (size_t l = 0; l < qlen; ++l) {
        for (size_t h = 0; h < nhead; ++h) {
            const size_t kv_h = h / q_per_kv;

            float max_score = -INFINITY;
            for (size_t s = 0; s < kvlen; ++s) {
                if (s > kvlen - qlen + l) {
                    scores[s] = -INFINITY;
                    continue;
                }
                float dot = 0.0f;
                for (size_t d = 0; d < hd; ++d) {
                    dot += utils::cast<float>(q[(l * nhead + h) * hd + d])
                         * utils::cast<float>(k[(s * nkvh + kv_h) * hd + d]);
                }
                scores[s] = dot * scale;
                if (scores[s] > max_score) {
                    max_score = scores[s];
                }
            }

            float sum = 0.0f;
            for (size_t s = 0; s < kvlen; ++s) {
                if (scores[s] == -INFINITY) {
                    probs[s] = 0.0f;
                } else {
                    probs[s] = std::exp(scores[s] - max_score);
                    sum += probs[s];
                }
            }
            for (size_t s = 0; s < kvlen; ++s) {
                probs[s] /= sum;
            }

            for (size_t d = 0; d < hd; ++d) {
                float out_v = 0.0f;
                for (size_t s = 0; s < kvlen; ++s) {
                    out_v += probs[s] * utils::cast<float>(v[(s * nkvh + kv_h) * hd + d]);
                }
                attn_val[(l * nhead + h) * hd + d] = utils::cast<T>(out_v);
            }
        }
    }
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale,
                    llaisysDataType_t dtype, size_t qlen, size_t kvlen, size_t nhead, size_t nkvh, size_t hd) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), scale,
                               qlen, kvlen, nhead, nkvh, hd);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<fp16_t *>(attn_val), reinterpret_cast<const fp16_t *>(q),
                               reinterpret_cast<const fp16_t *>(k), reinterpret_cast<const fp16_t *>(v), scale,
                               qlen, kvlen, nhead, nkvh, hd);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<bf16_t *>(attn_val), reinterpret_cast<const bf16_t *>(q),
                               reinterpret_cast<const bf16_t *>(k), reinterpret_cast<const bf16_t *>(v), scale,
                               qlen, kvlen, nhead, nkvh, hd);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
