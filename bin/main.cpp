#include <mazorca/mazorca.hpp>

#include <sycl/sycl.hpp>

int main() {

    // Create a SYCL device object
    sycl::device sycl_device;

    // Default SYCL device to CPU if GPU is not available
    // TODO: future SYCL versions will avoid throwing exceptions
    try {
        sycl_device = sycl::device(sycl::gpu_selector_v);
    } catch (const sycl::exception& e) {
        std::println(
            "Cannot select GPU:\n{}\nUsing CPU device instead.", 
            e.what()
        );
        sycl_device = sycl::device(sycl::cpu_selector_v);
    }

    // Create mazorca grano object from single SYCL device
    mazorca::grano grano{sycl_device};

    // Create app GUI object from grano object
    mazorca::app app(grano);

    // Run mazorca's GUI interface
    if (auto result = app.run(); !result.has_value()) {
        return std::to_underlying(mazorca::error_code::invalid);
    }
}
