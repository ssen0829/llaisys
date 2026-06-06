#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && attn_val->ndim() == 3,
           "SelfAttention: q, k, v and attn_val must be 3D.");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(q->isContiguous() && k->isContiguous() && v->isContiguous() && attn_val->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    size_t qlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t hd = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];

    ASSERT(k->shape()[2] == hd, "SelfAttention: head dim mismatch between q and k.");
    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == nkvh && v->shape()[2] == hd,
           "SelfAttention: v shape mismatch.");
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nhead && attn_val->shape()[2] == hd,
           "SelfAttention: attn_val shape mismatch.");
    ASSERT(nhead % nkvh == 0, "SelfAttention: nhead must be divisible by nkvh.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                   qlen, kvlen, nhead, nkvh, hd);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                   qlen, kvlen, nhead, nkvh, hd);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
