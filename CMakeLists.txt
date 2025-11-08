cmake_minimum_required(VERSION 3.25)

project(mazorca)

# Compiler hardened options
set(CXX_HARDENED_FLAGS
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

# Linker hardened options
set(LINKER_HARDENED_FLAGS
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
    add_subdirectory(test)
endif()
