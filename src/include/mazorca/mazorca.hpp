#pragma once

#include <sycl/sycl.hpp>

namespace mazorca {

enum class ReturnCode : int {
    valid = 0,
    invalid
};

// Example implementation of SYCL asynchronous exception handler
void sycl_async_handler(sycl::exception_list exceptions) {
  for (auto e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cout 
        << "Caught asynchronous SYCL exception:\n"
        << e.what() 
        << '\n';
    }
  }
};

void check_sycl_device_features(sycl::queue &queue) {
    if (!queue.get_device().ext_oneapi_can_compile(sycl::ext::oneapi::experimental::source_language::sycl)) {
    std::cout 
        << "SYCL-RTC is not supported for " 
        << queue.get_device().get_info<sycl::info::device::name>() 
        << '\n';
    }
}

} // namespace mazorca
