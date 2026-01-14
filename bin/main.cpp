#include <mazorca/mazorca.hpp>

int main() {
    // Search all root devices from all SYCL backends available in the system
    std::vector<sycl::device> sycl_devices {sycl::device::get_devices(sycl::info::device_type::all)};

    #ifndef NDEBUG
    std::println("[{}] [DEBUG] Obtained the following SYCL devices:", mazorca::current_time());
    for (const auto& device : sycl_devices) {
        std::println(
            "-> {} / {} / {} / SYCL backend version {}.", 
            device.get_info<sycl::info::device::name>(),
            device.get_info<sycl::info::device::driver_version>(),
            device.get_info<sycl::info::device::version>(),
            device.get_info<sycl::info::device::backend_version>()
        );
    }
    #endif

    // Each grano object supports a single SYCL device, context and queue
    #ifndef NDEBUG
    std::println("[{}] [DEBUG] Creating vector of grano objects from SYCL devices...", mazorca::current_time());
    #endif
    std::vector<mazorca::grano> granos {};
    granos.reserve(sycl_devices.size());
    for (auto&& device : sycl_devices) {
        granos.emplace_back(std::move(device));
    }
    #ifndef NDEBUG
    std::println("[{}] [DEBUG] Vector of grano objects created from SYCL devices.", mazorca::current_time());
    #endif

    #ifndef NDEBUG
    std::println("[{}] [DEBUG] Creating mazorca GUI application object...", mazorca::current_time());
    #endif
    // Create app GUI object and transfer ownership of SYCL device objects
    mazorca::app app {std::move(granos)};
    #ifndef NDEBUG
    std::println(
        "[{}] [DEBUG] Vector of granos is empty after move operation: {}", 
        mazorca::current_time(),
        granos.empty()
    );
    std::println("[{}] [DEBUG] mazorca GUI application object created.", mazorca::current_time());
    #endif

    // Run mazorca's GUI interface
    if (auto result = app.run(); !result.has_value()) {
        auto error_value = std::to_underlying(mazorca::error_code::invalid);
        std::println(
            "[{}] [ERROR] mazorca run method returned error code: {}", 
            mazorca::current_time(),
            error_value
        );
        return error_value;
    }

    #ifndef NDEBUG
    std::println("[{}] [DEBUG] Returning from main function with value 0.", mazorca::current_time());
    #endif
}
