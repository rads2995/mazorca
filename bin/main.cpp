#include <mazorca/mazorca.hpp>

int main() {

    // Search all root devices from all SYCL backends available in the system
    std::vector<sycl::device> sycl_devices {sycl::device::get_devices(sycl::info::device_type::all)};
    
    // Each grano object supports a single SYCL device, context and queue
    std::vector<mazorca::grano> granos {};
    granos.reserve(sycl_devices.size());
    for (auto&& device : sycl_devices) {
        granos.emplace_back(std::move(device));
    }

    // Create app GUI object and transfer ownership of SYCL device objects
    mazorca::app app {std::move(granos)};

    // Run mazorca's GUI interface
    if (auto result = app.run(); !result.has_value()) {
        return std::to_underlying(mazorca::error_code::invalid);
    }
}
