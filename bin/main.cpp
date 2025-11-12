#include <chrono>
#include <filesystem>
#include <fstream>

#include <sycl/sycl.hpp>

#include <mazorca/mazorca.hpp>

// Example implementation of SYCL asynchronous exception handler
void async_handler(sycl::exception_list exceptions) {
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

void check_device_features(sycl::queue &queue) {
    if (!queue.get_device().ext_oneapi_can_compile(sycl::ext::oneapi::experimental::source_language::sycl)) {
    std::cout 
        << "SYCL-RTC is not supported for " 
        << queue.get_device().get_info<sycl::info::device::name>() 
        << '\n';
    }
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    // Input file paths for run-time-compiled kernels
    // TODO: check if filepath is valid?
    std::filesystem::path kernel_file_path(argv[1]);
    
    // Create SYCL devices
    sycl::device cpu_device(sycl::cpu_selector_v);
    sycl::device gpu_device;
    try {
        gpu_device = sycl::device(sycl::gpu_selector_v);
    } catch (sycl::exception const &e) {
        // TODO: how to handle no GPU? Return invalid for now...
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    // Create SYCL contexts
    sycl::context cpu_context(cpu_device);
    sycl::context gpu_context(gpu_device);

    // Create SYCL queues
    sycl::queue cpu_queue(
        cpu_context,
        cpu_device,
        async_handler,
        sycl::property::queue::enable_profiling{}
    );
    sycl::queue gpu_queue(
        gpu_context,
        gpu_device,
        async_handler,
        sycl::property::queue::enable_profiling{}
    );

    std::ifstream kernel_file(kernel_file_path, std::ios::binary | std::ios::ate);

    if (!kernel_file) {
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    // Read SYCL kernel to string for kernel bundle source
    std::string sycl_source{
        std::istreambuf_iterator<char>(kernel_file), 
        std::istreambuf_iterator<char>()
    };

    // Check SYCL features that are available to each device
    check_device_features(cpu_queue);
    check_device_features(gpu_queue);

    // TODO: SYCL RTC currently not supported on AMD HIP backend
    auto source_bundle = sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
        cpu_queue.get_context(), 
        sycl::ext::oneapi::experimental::source_language::sycl, 
        sycl_source
    );

    auto exec_bundle = sycl::ext::oneapi::experimental::build(source_bundle);

    if(exec_bundle.ext_oneapi_has_kernel("vec_add")) {
        std::cout 
            << "SYCL kernel found on " 
            << source_bundle.get_devices()[0].get_info<sycl::info::device::name>() 
            << '\n';
    }

    // Try a sample kernel for the gpu device to make sure it works!
    constexpr int n = 10;
    int *data = sycl::malloc_shared<int>(n + 1, gpu_queue);
    std::memset(data, 0, sizeof(*data) * n);

    sycl::event e;
    for (int i = 1; i < n; i += 2) {
        e = gpu_queue.submit([&](sycl::handler &h) {
        // wait for previous device task
        e.wait();
        auto device_task = [=]() { data[i] = data[i - 1] + 1; };
        h.single_task(device_task);
        });

        gpu_queue.submit([&](sycl::handler &h) {
        // wait for device task to complete
        e.wait();
        auto host_task = [=]() { data[i + 1] = data[i] + 1; };
        h.host_task(host_task);
        });
    }
    for (int i = 0; i < n; i++)
        std::cout << i << ": " << data[i] << "\n";

    sycl::free(data, gpu_queue);

    return std::to_underlying(mazorca::ReturnCode::valid);
}
