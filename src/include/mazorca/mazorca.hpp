#pragma once

#include <filesystem>
#include <expected>
#include <print>
#include <vector>

#include <sycl/sycl.hpp>

namespace mazorca {

enum class error_code : int {
    invalid = 1
};

// SYCL asynchronous exception handler
inline void sycl_async_handler(sycl::exception_list exceptions) {
    for (const std::exception_ptr& e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (const sycl::exception& e) {
            std::println(
                "Caught asynchronous SYCL exception:\n{}",
                e.what()
            );
        }
    }
};

struct grano {

    sycl::device sycl_device;
    sycl::context sycl_context;
    sycl::queue sycl_queue;

    grano(const sycl::device& device) 
    : sycl_device(device),
      sycl_context(sycl_device),
      sycl_queue(sycl_context, sycl_device, mazorca::sycl_async_handler, sycl::property::queue::enable_profiling{}) {}

    [[nodiscard]] std::expected<void, error_code> work();

    [[nodiscard]] std::expected<void, error_code> create_kernel_bundle(std::filesystem::path& kernel_bundle_file_path);

    [[nodiscard]] std::expected<void, error_code> nn_example();
};

struct app {
    
    std::vector<grano> granos;

    app(const grano& grano) : granos({grano}) {}

    app(const std::vector<grano>& granos_) : granos(granos_) {}

    [[nodiscard]] std::expected<void, error_code> run();
};

} // namespace mazorca
