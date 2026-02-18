#pragma once

#include <expected>
#include <print>

#include <sycl/sycl.hpp>

namespace mazorca {

enum class error_code : int8_t {
    invalid = 1
};

// Current point in time for application logging purposes
[[nodiscard]] inline constexpr std::string current_time() {
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
inline constexpr void sycl_async_handler(sycl::exception_list exceptions) {
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

    const sycl::device sycl_device;
    const sycl::context sycl_context;
    sycl::queue sycl_queue;

    explicit grano(const sycl::device& device) 
    : sycl_device(device),
      sycl_context(sycl_device),
      sycl_queue(sycl_context, sycl_device, mazorca::sycl_async_handler, sycl::property::queue::enable_profiling{}) {}

    [[nodiscard]] constexpr std::expected<void, error_code> work();

    [[nodiscard]] constexpr std::expected<void, error_code> create_kernel_bundle(const std::filesystem::path& kernel_bundle_file_path) const;
};

struct app {
    
    const std::vector<grano> granos;

    explicit app(std::vector<grano>&& granos_) : granos(std::move(granos_)) {}

    [[nodiscard]] std::expected<void, error_code> run() const;
};

} // namespace mazorca

constexpr std::expected<void, mazorca::error_code> mazorca::grano::work() {

    constexpr int size = 10;

    // Create allocator for device associated with q
    sycl::usm_allocator<std::uint64_t, sycl::usm::alloc::host> host_allocator(this->sycl_queue);
    
    // Create std vectors with the allocator
    std::vector<std::uint64_t, sycl::usm_allocator<std::uint64_t, sycl::usm::alloc::host>> 
    a(size, host_allocator), b(size, host_allocator), c(size, host_allocator);

    // Get pointer to vector data for access in kernel
    auto A = a.data();
    auto B = b.data();
    auto C = c.data();

    for (std::size_t i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i;
        c[i] = i;
    }

    this->sycl_queue.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) { 
            C[idx] = A[idx] + B[idx]; }
        );
    }).wait();

    for (std::size_t i = 0; i < size; i++) {
        std::cout << c[i] << std::endl;
    }

    return {};
}

constexpr std::expected<void, mazorca::error_code> mazorca::grano::create_kernel_bundle(const std::filesystem::path& kernel_bundle_file_path) const {

    std::ifstream kernel_file(kernel_bundle_file_path, std::ios::binary);

    if (!kernel_file) {
        return std::unexpected(mazorca::error_code::invalid);
    }

    // Read SYCL kernel to string for kernel bundle source
    std::string sycl_source{
        std::istreambuf_iterator<char>(kernel_file), 
        std::istreambuf_iterator<char>()
    };

    // Check if SYCL run-time compilation feature is available for this device
    if (!this->sycl_queue.get_device().ext_oneapi_can_compile(sycl::ext::oneapi::experimental::source_language::sycl)) {
        std::cout 
            << "SYCL-RTC is not supported for " 
            << this->sycl_queue.get_device().get_info<sycl::info::device::name>() 
            << '\n'; 
        return std::unexpected(mazorca::error_code::invalid);
    }

    // Create kernel bundle for current device
    // TODO: what is the concrete type instead of auto?
    
    
    sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>  
    source_bundle = sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
        this->sycl_queue.get_context(), 
        sycl::ext::oneapi::experimental::source_language::sycl, 
        sycl_source
    );

    // Build kernel using run-time compilation (this is expensive!)
    // Note: build arguments can be passed using the build_options array
    sycl::ext::oneapi::experimental::build_options build_opts {};
    std::string compiler_output {};
    sycl::ext::oneapi::experimental::save_log log{&compiler_output};
    sycl::kernel_bundle<sycl::bundle_state::executable>
    exec_bundle = sycl::ext::oneapi::experimental::build(
        source_bundle,
        sycl::ext::oneapi::experimental::properties{build_opts, log}
    );
    println("SYCL RTC output:\n{}", compiler_output);

    // Query the kernels that were compiled for the current device
    if(exec_bundle.ext_oneapi_has_kernel("vec_add")) {
        std::cout 
            << "SYCL kernel found on " 
            << source_bundle.get_devices()[0].get_info<sycl::info::device::name>() 
            << '\n';
    }

    return {};
}
