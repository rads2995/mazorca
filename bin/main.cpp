#include <mazorca/mazorca.hpp>

#include <filesystem>

#include <sycl/sycl.hpp>

int main(int argc, char* argv[]) {

    // Create a mazorca using the CPU device
    mazorca::Mazorca cpu{sycl::device(sycl::cpu_selector_v)};

    // Create a mazorca using the GPU device
    // TODO: this throws if GPU not found..., how to handle that?
    mazorca::Mazorca gpu{sycl::device(sycl::gpu_selector_v)};

    // Right now we pass kernels as input arguments
    // TODO: make this work with a flag to pass kernel bundles
    if (argc != 2) {
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    // Input file path for run-time JIT-compiled kernel bundles
    std::filesystem::path kernel_file_path(argv[1]);

    // Create kernel bundle from input file on the CPU device
    if (auto result = cpu.create_kernel_bundle(kernel_file_path); !result.has_value()) {
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    // Perform some kernel work on the GPU device
    if (auto result = gpu.work(); !result.has_value()) {
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    return std::to_underlying(mazorca::ReturnCode::valid);
}
