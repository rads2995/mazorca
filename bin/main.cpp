#include <expected>
#include <print>
#include <utility>
#include <vector>

#include <mazorca/mazorca.hpp>
#include <sycl/device.hpp>
#include <sycl/info/info_desc.hpp>

auto main() -> int {
    // Search all root devices from all SYCL backends available in the system
    const std::vector<sycl::device> sycl_devices{sycl::device::get_devices(sycl::info::device_type::all)};

    if (sycl_devices.empty()) {
        std::println("[{}] [ERROR] No SYCL devices returned from get_devices method:", mazorca::current_time());
        return std::to_underlying(mazorca::error_code::invalid);
    }

    std::println("[{}] [INFO] Obtained the following SYCL devices from get_devices method:", mazorca::current_time());
    for (const auto& device : sycl_devices) {
        std::println("-> {} / {} / {} / SYCL backend version {}.", device.get_info<sycl::info::device::name>(),
                     device.get_info<sycl::info::device::driver_version>(),
                     device.get_info<sycl::info::device::version>(),
                     device.get_info<sycl::info::device::backend_version>());
    }

    // Each grano object supports a single SYCL device, context and queue
    std::println("[{}] [INFO] Creating vector of grano objects from SYCL devices...", mazorca::current_time());
    std::vector<mazorca::grano> granos{};
    granos.reserve(sycl_devices.size());
    for (const auto& device : sycl_devices) {
        granos.emplace_back(device);
    }
    std::println("[{}] [INFO] Vector of grano objects created from SYCL devices.", mazorca::current_time());

    // Create app GUI object and transfer ownership of SYCL device objects
    std::println("[{}] [INFO] Creating mazorca GUI application object...", mazorca::current_time());
    mazorca::app const app{std::move(granos)};
    std::println("[{}] [INFO] mazorca GUI application object created.", mazorca::current_time());

    // Run mazorca's GUI interface
    auto result =
        app.run().or_else([](const mazorca::error_code& error_code) -> std::expected<void, mazorca::error_code> {
            std::println(
                "[{}] [ERROR] mazorca application run method returned error "
                "code: {}",
                mazorca::current_time(), std::to_underlying(error_code));
            return std::expected<void, mazorca::error_code>(std::unexpect, error_code);
        });

    if (!result.has_value()) {
        const auto error_code{std::to_underlying(result.error())};
        std::println("[{}] [ERROR] mazorca main function returned error code: {}", mazorca::current_time(), error_code);
        return error_code;
    }

    std::println("[{}] [INFO] Returning from main function with value 0.", mazorca::current_time());
}
