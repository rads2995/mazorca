#include <mazorca/mazorca.hpp>

#include <unistd.h>
#include <filesystem>

#include <sycl/sycl.hpp>

int main(int argc, char** argv) {

    // Filesystem path for SYCL-RTC kernel bundle
    std::filesystem::path kernel_bundle_path;

    // Filesystem path for JIT-compiled shaders
    std::filesystem::path shader_path;

    int option;
    while ((option = getopt(argc, argv, "k:h")) != -1) {
        switch (option) {
            case 'k':
                kernel_bundle_path.assign(optarg);
                if (!kernel_bundle_path.has_filename()) {
                    std::cout
                        << "Invalid file type for SYCL-RTC kernel bundle: "
                        << kernel_bundle_path
                        << '\n';
                    return std::to_underlying(mazorca::error_code::invalid);
                }
                break;
            case 'h':
                // TODO: print a help menu detailing flags above
                break;
        }
    }
    
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

    // Create kernel bundle from input file on the CPU device
    if (!kernel_bundle_path.empty()) {
        if (auto result = cpu.create_kernel_bundle(kernel_bundle_path); !result.has_value()) {
            return std::to_underlying(mazorca::error_code::invalid);
        }
    }

    // Perform some kernel work on the GPU device
    if (auto result = gpu.work(); !result.has_value()) {
        return std::to_underlying(mazorca::error_code::invalid);
    }
}
