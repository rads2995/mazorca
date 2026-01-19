#include <mazorca/mazorca.hpp>

int main() {
    // Search all root devices from all SYCL backends available in the system
    std::vector<sycl::device> sycl_devices {sycl::device::get_devices(sycl::info::device_type::all)};

    std::println("[{}] [INFO] Obtained the following SYCL devices:", mazorca::current_time());
    for (const auto& device : sycl_devices) {
        std::println(
            "-> {} / {} / {} / SYCL backend version {}.", 
            device.get_info<sycl::info::device::name>(),
            device.get_info<sycl::info::device::driver_version>(),
            device.get_info<sycl::info::device::version>(),
            device.get_info<sycl::info::device::backend_version>()
        );
    }

    // Each grano object supports a single SYCL device, context and queue
    std::println("[{}] [INFO] Creating vector of grano objects from SYCL devices...", mazorca::current_time());
    std::vector<mazorca::grano> granos {};
    granos.reserve(sycl_devices.size());
    for (auto& device : sycl_devices) {
        granos.emplace_back(std::move(device));
    }
    std::println("[{}] [INFO] Vector of grano objects created from SYCL devices.", mazorca::current_time());

    // Create app GUI object and transfer ownership of SYCL device objects
    std::println("[{}] [INFO] Creating mazorca GUI application object...", mazorca::current_time());
    mazorca::app app {std::move(granos)};
    std::println("[{}] [INFO] mazorca GUI application object created.", mazorca::current_time());

    // Run mazorca's GUI interface
    if (auto result = app.run(); !result.has_value()) {
        std::println(
            "[{}] [ERROR] mazorca application run method returned error code: {}", 
            mazorca::current_time(),
            std::to_underlying(result.error())
        );
        return std::to_underlying(result.error());
    }
    std::println("[{}] [INFO] Returning from main function with value 0.", mazorca::current_time());
}
