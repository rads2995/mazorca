cmake_minimum_required(VERSION 4.0.0)

project(mazorca)

# C++ compiler options
set(CXX_COMPILER_FLAGS
    -fsycl
    -fsycl-targets=spir64_x86_64,amd_gpu_gfx1010
    -O2 
    -Wall 
    -Wformat 
    -Wformat=2 
    -Wconversion 
    -Wimplicit-fallthrough
    -Werror=format-security
    -U_FORTIFY_SOURCE
    -D_FORTIFY_SOURCE=3
    -D_GLIBCXX_ASSERTIONS
    -fstrict-flex-arrays=3
    -fstack-clash-protection
    -fstack-protector-strong
    -fno-delete-null-pointer-checks
    -fno-strict-overflow
    -fno-strict-aliasing
    -ftrivial-auto-var-init=zero
)

# C++ linker options
set(CXX_LINKER_FLAGS
    -fsycl
    -fsycl-targets=spir64_x86_64,amd_gpu_gfx1010
    -Wl,-z,nodlopen 
    -Wl,-z,noexecstack
    -Wl,-z,relro 
    -Wl,-z,now
    -Wl,--as-needed
    -Wl,--no-copy-dt-needed-entries
)

add_subdirectory(src)
add_subdirectory(bin)

if(BUILD_TESTING)
    include(CTest)
    add_subdirectory(test)
endif()
