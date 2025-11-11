#include <chrono>
#include <iostream>
#include <filesystem>
#include <fstream>

#include <sycl/sycl.hpp>

#include <mazorca/mazorca.hpp>

int main(int argc, char* argv[]) {

    if (argc < 2) {
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    std::filesystem::path kernel_file_path(argv[1]);
    
    sycl::queue q(
        sycl::cpu_selector_v, 
        sycl::property::queue::enable_profiling{}
    );

    std::ifstream kernel_file(kernel_file_path, std::ios::binary | std::ios::ate);

    if (!kernel_file) {
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }
    
    std::streamsize file_size = kernel_file.tellg();
    kernel_file.seekg(0, std::ios::beg);

    std::string sycl_source(file_size, '\0');
    if (!kernel_file.read(sycl_source.data(), file_size)) {
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    if (!q.get_device().ext_oneapi_can_compile(sycl::ext::oneapi::experimental::source_language::sycl)) {
        std::cout 
            << "SYCL-RTC not supported for " 
            << q.get_device().get_info<sycl::info::device::name>() 
            << '\n';
    }

    auto source_bundle = sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
        q.get_context(), 
        sycl::ext::oneapi::experimental::source_language::sycl, 
        sycl_source // TODO: pass string as vector of bytes, a la Rust's str.as_bytes() -> Vec<u8>
    );

    auto exec_bundle = sycl::ext::oneapi::experimental::build(source_bundle);

    if(exec_bundle.ext_oneapi_has_kernel("vec_add")) {
        std::cout << "SYCL kernel found!" << '\n';
    }

    return std::to_underlying(mazorca::ReturnCode::valid);
}
