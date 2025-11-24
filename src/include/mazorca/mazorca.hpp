#pragma once

#include <filesystem>

#include <sycl/sycl.hpp>

namespace mazorca {

enum class ReturnCode : int {
    valid = 0,
    invalid
};

// Example implementation of SYCL asynchronous exception handler
inline void sycl_async_handler(sycl::exception_list exceptions) {
    for (auto e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (sycl::exception const &e) {
            std::cout 
            << "Caught asynchronous SYCL exception:\n"
            << e.what() 
            << '\n';
        }
    }
};

inline void check_sycl_device_features(sycl::queue &queue) {
    if (queue.get_device().ext_oneapi_can_compile(sycl::ext::oneapi::experimental::source_language::sycl)) {
    std::cout 
        << "SYCL-RTC is supported for " 
        << queue.get_device().get_info<sycl::info::device::name>() 
        << '\n';
    } else {
    std::cout 
        << "SYCL-RTC is not supported for " 
        << queue.get_device().get_info<sycl::info::device::name>() 
        << '\n'; 
    }
}

struct Mazorca {

    sycl::device sycl_device;
    sycl::context sycl_context;
    sycl::queue sycl_queue;

    Mazorca(
        sycl::device device = sycl::device(sycl::default_selector_v)
    ) 
    : sycl_device(device),
      sycl_context(sycl_device),
      sycl_queue(sycl_context, sycl_device, mazorca::sycl_async_handler, sycl::property::queue::enable_profiling{}) {}

    int run();

    int create_kernel_bundle(std::filesystem::path& kernel_bundle_file_path);
};

} // namespace mazorca
