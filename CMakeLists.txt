cmake_minimum_required(VERSION 4.0.0)

project(mazorca
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Default build configuration is Release and changed using presets
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build configuration")

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

set(BUILD_TESTING OFF CACHE BOOL "Enable building tests")
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
