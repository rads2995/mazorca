#include <mazorca/mazorca.hpp>

#include <filesystem>

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

    cpu.create_kernel_bundle(kernel_file_path);

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
