#include <chrono>
#include <iostream>

#include <sycl/sycl.hpp>

#include <mazorca/mazorca.hpp>

int main() {

    sycl::queue q(
        sycl::cpu_selector_v, 
        sycl::property::queue::enable_profiling{}
    );

    std::string sycl_source = R"""(
    #include <sycl/sycl.hpp>

    extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((
        sycl::ext::oneapi::experimental::nd_range_kernel<1>))
    void vec_add(float* in1, float* in2, float* out){
        size_t id = sycl::ext::oneapi::this_work_item::get_nd_item<1>()
                    .get_global_linear_id();
        out[id] = in1[id] + in2[id];
    }
    )""";

    if (!q.get_device().ext_oneapi_can_compile(sycl::ext::oneapi::experimental::source_language::sycl)) {
        std::cout 
            << "SYCL-RTC not supported for " 
            << q.get_device().get_info<sycl::info::device::name>() 
            << '\n';
    }

    auto source_bundle = sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
        q.get_context(), 
        sycl::ext::oneapi::experimental::source_language::sycl, 
        sycl_source
    );

    auto exec_bundle = sycl::ext::oneapi::experimental::build(source_bundle);

    if(exec_bundle.ext_oneapi_has_kernel("vec_add")) {
        std::cout << "SYCL kernel found!" << '\n';
    }

    return std::to_underlying(mazorca::ReturnCode::valid);

}
