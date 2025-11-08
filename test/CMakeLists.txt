include(CTest)
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        52eb8108c5bdec04579160ae17225d66034bd723 # release-1.17.0
)
FetchContent_MakeAvailable(googletest)

add_executable(mazorca_test)

target_compile_options(mazorca_test
    PRIVATE
        ${CXX_HARDENED_FLAGS}
)

target_link_options(mazorca_test
    PRIVATE
        ${LINKER_HARDENED_FLAGS}
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
