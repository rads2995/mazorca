#include <mazorca/mazorca.hpp>

#include <fstream>
#include <string>
#include <cstdint>

#include <sycl/sycl.hpp>

std::expected<void, mazorca::error_code> mazorca::kernel::work() {
    
    constexpr int n = 1024;
    std::uint64_t *data = sycl::malloc_shared<std::uint64_t>(n, this->sycl_queue);

    this->sycl_queue.parallel_for(n, [=](sycl::id<1> idx) {
        data[idx] = idx;
    }).wait();
    sycl::free(data, this->sycl_queue);

    return {};
}

std::expected<void, mazorca::error_code> mazorca::kernel::create_kernel_bundle(std::filesystem::path& kernel_bundle_file_path) {

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

    // Create surcle bundle for current device
    // TODO: what is the concrete type instead of auto?
    auto source_bundle = sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
        this->sycl_queue.get_context(), 
        sycl::ext::oneapi::experimental::source_language::sycl, 
        sycl_source
    );

    // Build kernel using run-time compilation (this is expensive!)
    auto exec_bundle = sycl::ext::oneapi::experimental::build(source_bundle);

    // Query the kernels that were compiled for the current device
    if(exec_bundle.ext_oneapi_has_kernel("vec_add")) {
        std::cout 
            << "SYCL kernel found on " 
            << source_bundle.get_devices()[0].get_info<sycl::info::device::name>() 
            << '\n';
    }

    return {};
}
