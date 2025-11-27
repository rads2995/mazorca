#pragma once

#include <filesystem>
#include <expected>

#include <sycl/sycl.hpp>

namespace mazorca {

enum class error_code : int {
    invalid = 1
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

struct kernel {

    sycl::device sycl_device;
    sycl::context sycl_context;
    sycl::queue sycl_queue;

    kernel(
        sycl::device device = sycl::device(sycl::default_selector_v)
    ) 
    : sycl_device(device),
      sycl_context(sycl_device),
      sycl_queue(sycl_context, sycl_device, mazorca::sycl_async_handler, sycl::property::queue::enable_profiling{}) {}

    [[nodiscard]] std::expected<void, error_code> work();

    [[nodiscard]] std::expected<void, error_code> create_kernel_bundle(std::filesystem::path& kernel_bundle_file_path);
};

struct app {
    
    [[nodiscard]] std::expected<void, error_code> run();
};

struct compiler {

    int test();
};

} // namespace mazorca
