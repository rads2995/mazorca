cmake_minimum_required(VERSION 4.0.0)

project(mazorca)

# C++ compiler options
set(CXX_COMPILER_FLAGS
    -fsycl
    -fsycl-targets=spir64_x86_64,amd_gpu_gfx1010
)

# C++ linker options
set(CXX_LINKER_FLAGS
    -fsycl
    -fsycl-targets=spir64_x86_64,amd_gpu_gfx1010
)

add_subdirectory(src)
add_subdirectory(bin)

if(BUILD_TESTING)
    include(CTest)
    add_subdirectory(test)
endif()
