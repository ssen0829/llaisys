#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }
    ASSERT(in->ndim() == 2 && weight->ndim() == 2 && out->ndim() == 2, "Linear: tensors must be 2D.");
    ASSERT(in->isContiguous() && weight->isContiguous() && out->isContiguous(),
           "Linear: out, in and weight must be contiguous.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        ASSERT(bias->ndim() == 1, "Linear: bias must be 1D.");
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->shape()[0] == weight->shape()[0], "Linear: bias size must match out features.");
    }

    size_t m = in->shape()[0];
    size_t k = in->shape()[1];
    size_t n = weight->shape()[0];
    ASSERT(weight->shape()[1] == k, "Linear: weight shape mismatch.");
    ASSERT(out->shape()[0] == m && out->shape()[1] == n, "Linear: out shape mismatch.");

    const std::byte *bias_data = bias ? bias->data() : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), m, n, k);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), m, n, k);
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
