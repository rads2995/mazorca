#pragma once

#include <filesystem>
#include <expected>

#include <sycl/sycl.hpp>

namespace mazorca {

enum class ReturnCode : int {
    valid = 0,
    invalid
};

// SYCL asynchronous exception handler
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

    [[nodiscard]] int run();

    [[nodiscard]] std::expected<void, ReturnCode> work();

    [[nodiscard]] std::expected<void, ReturnCode> create_kernel_bundle(std::filesystem::path& kernel_bundle_file_path);
};

} // namespace mazorca
