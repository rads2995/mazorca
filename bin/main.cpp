#include <mazorca/mazorca.hpp>

#include <filesystem>

#include <sycl/sycl.hpp>

int main(int argc, char* argv[]) {

    // Create a kernel object using the CPU device
    mazorca::kernel cpu{sycl::device(sycl::cpu_selector_v)};

    // Create a kernel object using the GPU device
    // TODO: this throws if GPU not found..., how to handle that?
    mazorca::kernel gpu{sycl::device(sycl::gpu_selector_v)};

    // Create app GUI
    mazorca::app app;

    // Run mazorca's GUI interface
    // Note: this app method performs compilation of shaders
    if (auto result = app.run(); !result.has_value()) {
        return std::to_underlying(mazorca::error_code::invalid);
    }

    // TODO: anything below is proof of concept
    // Ideally, flags should be parsed and inputs passed to the app object
    
    // Right now we pass the kernel bundle as input arguments
    if (argc != 2) {
        return std::to_underlying(mazorca::error_code::invalid);
    }

    // Input file path for run-time JIT-compiled kernel bundles
    std::filesystem::path kernel_file_path(argv[1]);

    // Create kernel bundle from input file on the CPU device
    if (auto result = cpu.create_kernel_bundle(kernel_file_path); !result.has_value()) {
        return std::to_underlying(mazorca::error_code::invalid);
    }

    // Perform some kernel work on the GPU device
    if (auto result = gpu.work(); !result.has_value()) {
        return std::to_underlying(mazorca::error_code::invalid);
    }
}
