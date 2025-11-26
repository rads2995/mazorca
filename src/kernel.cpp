#include <mazorca/mazorca.hpp>

#include <fstream>
#include <string>
#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

std::expected<void, mazorca::error_code> mazorca::kernel::work() {

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
