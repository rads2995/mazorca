include(cmake/imgui.cmake)
include(cmake/slang.cmake)

find_package(SDL3 REQUIRED)
find_package(Vulkan REQUIRED)

add_library(mazorca STATIC)
add_library(mazorca::mazorca ALIAS mazorca)

set_target_properties(mazorca PROPERTIES 
    CXX_CLANG_TIDY "clang-tidy;-p;${CMAKE_BINARY_DIR}"
)

target_sources(mazorca
    PUBLIC
        FILE_SET mazorca_headers
        TYPE HEADERS
        BASE_DIRS ./include
        FILES include/mazorca/mazorca.hpp
    PRIVATE
        app.cpp
)

target_link_libraries(mazorca
    PRIVATE
        $<BUILD_LOCAL_INTERFACE:imgui::imgui>
        $<BUILD_LOCAL_INTERFACE:slang>
        SDL3::SDL3
        Vulkan::Vulkan
)

target_compile_options(mazorca
    PRIVATE
        ${CXX_COMPILER_FLAGS}
)

target_link_options(mazorca
    PRIVATE
        ${CXX_LINKER_FLAGS}
)

install(
    TARGETS mazorca
    EXPORT mazorcaTargets
    FILE_SET mazorca_headers
)

install(
    EXPORT mazorcaTargets
    DESTINATION share/cmake/mazorca
    NAMESPACE mazorca::
)

install(
    FILES ./cmake/mazorcaConfig.cmake
    DESTINATION share/cmake/mazorca
)
