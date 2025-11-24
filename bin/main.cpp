#include <mazorca/mazorca.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>

#include <sycl/sycl.hpp>

int main(int argc, char* argv[]) {

    // Create a mazorca using the CPU device
    mazorca::Mazorca cpu{sycl::device(sycl::cpu_selector_v)};

    // Create a mazorca using the GPU device
    // TODO: this throws if GPU not found..., how to handle that?
    mazorca::Mazorca gpu{sycl::device(sycl::gpu_selector_v)};

    // This starts the GUI application
    // TODO: the GUI should be separated from Mazorca (GPU kernels)
    cpu.run();
    
    // Right now we pass kernels as input arguments
    if (argc != 2) {
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    // Input file paths for run-time-compiled kernels
    std::filesystem::path kernel_file_path(argv[1]);

    std::ifstream kernel_file(kernel_file_path, std::ios::binary);

    if (!kernel_file) {
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    // Read SYCL kernel to string for kernel bundle source
    std::string sycl_source{
        std::istreambuf_iterator<char>(kernel_file), 
        std::istreambuf_iterator<char>()
    };

    // Check if SYCL run-time compilation feature is available for each device
    mazorca::check_sycl_device_features(cpu.sycl_queue);
    mazorca::check_sycl_device_features(gpu.sycl_queue);

    // Create surcle bundle for CPU device
    auto source_bundle = sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
        cpu.sycl_queue.get_context(), 
        sycl::ext::oneapi::experimental::source_language::sycl, 
        sycl_source
    );

    // Build kernel using run-time compilation (this is expensive!)
    auto exec_bundle = sycl::ext::oneapi::experimental::build(source_bundle);

    // Query the kernels that were compiled for the CPU device
    if(exec_bundle.ext_oneapi_has_kernel("vec_add")) {
        std::cout 
            << "SYCL kernel found on " 
            << source_bundle.get_devices()[0].get_info<sycl::info::device::name>() 
            << '\n';
    }

    // Try a sample kernel for the gpu device to make sure it works!
    constexpr int n = 10;
    int *data = sycl::malloc_shared<int>(n + 1, gpu.sycl_queue);
    std::memset(data, 0, sizeof(*data) * n);

    sycl::event e;
    for (int i = 1; i < n; i += 2) {
        e = gpu.sycl_queue.submit([&](sycl::handler &h) {
        // wait for previous device task
        e.wait();
        auto device_task = [=]() { data[i] = data[i - 1] + 1; };
        h.single_task(device_task);
        });

        gpu.sycl_queue.submit([&](sycl::handler &h) {
        // wait for device task to complete
        e.wait();
        auto host_task = [=]() { data[i + 1] = data[i] + 1; };
        h.host_task(host_task);
        });
    }
    for (int i = 0; i < n; i++)
        std::cout << i << ": " << data[i] << "\n";

    sycl::free(data, gpu.sycl_queue);

    return std::to_underlying(mazorca::ReturnCode::valid);
}
