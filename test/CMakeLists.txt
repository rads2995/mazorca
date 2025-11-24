include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        52eb8108c5bdec04579160ae17225d66034bd723 # Release 1.17.0
    FIND_PACKAGE_ARGS NAMES GTest
)
FetchContent_MakeAvailable(googletest)

add_executable(mazorca_test)

target_compile_options(mazorca_test
    PRIVATE
        ${CXX_HARDENED_FLAGS}
        -fsycl-targets=amd_gpu_gfx1010
)

target_link_options(mazorca_test
    PRIVATE
        ${LINKER_HARDENED_FLAGS}
        -fsycl-targets=amd_gpu_gfx1010
)

target_sources(mazorca_test
    PRIVATE
        test.cpp
)

target_link_libraries(mazorca_test
    PRIVATE
        mazorca::mazorca
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(mazorca_test)
