#include <sycl/sycl.hpp>

extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((
    sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void vec_add(float* in1, float* in2, float* out) {
    size_t id = sycl::ext::oneapi::this_work_item::get_nd_item<1>()
                .get_global_linear_id();
    out[id] = in1[id] + in2[id];
}
