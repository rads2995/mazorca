#pragma once

#include <filesystem>
#include <expected>
#include <print>
#include <vector>
#include <string>
#include <chrono>

#include <sycl/sycl.hpp>

namespace mazorca {

enum class error_code : int {
    invalid = 1
};

// Current point in time for application logging purposes
inline std::string current_time() {
    std::time_t time_now {
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())
    };
    std::string time_now_str {std::ctime(&time_now)};

    // If string is empty, we return to avoid undefined behavior
    if (time_now_str.empty()) return time_now_str;

    // If not empty, remove newline character from end of string
    time_now_str.pop_back();
    return time_now_str;
}

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

    app(std::vector<grano>&& granos_) : granos(std::move(granos_)) {}

    [[nodiscard]] std::expected<void, error_code> run();
};

} // namespace mazorca
