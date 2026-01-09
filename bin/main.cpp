#include <mazorca/mazorca.hpp>

int main() {

    // Create CPU and GPU (if available) SYCL device objects   
    std::vector<mazorca::grano> sycl_devices {sycl::device{sycl::cpu_selector_v}};
    try {
        sycl_devices.emplace_back(sycl::device{sycl::gpu_selector_v});
    } catch (const sycl::exception& e) {
        std::println(
            "Cannot create SYCL device from GPU:\n{}", 
            e.what()
        );
    }

    // Create app GUI object and register SYCL device objects
    mazorca::app app {sycl_devices};

    // Run mazorca's GUI interface
    if (auto result = app.run(); !result.has_value()) {
        return std::to_underlying(mazorca::error_code::invalid);
    }
}
