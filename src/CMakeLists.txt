include(cmake/imgui.cmake)
include(cmake/slang.cmake)
add_subdirectory(glad)

find_package(SDL3 REQUIRED)
find_package(OpenGL REQUIRED)

add_library(mazorca STATIC)
add_library(mazorca::mazorca ALIAS mazorca)

target_sources(mazorca
    PUBLIC
        FILE_SET mazorca_headers
        TYPE HEADERS
        BASE_DIRS ./include
        FILES include/mazorca/mazorca.hpp
    PRIVATE
        kernel.cpp
        app.cpp
)

target_link_libraries(mazorca
    PRIVATE
        $<BUILD_INTERFACE:imgui::imgui>
        $<BUILD_INTERFACE:glad::glad>
        $<BUILD_INTERFACE:slang>
    PUBLIC
        SDL3::SDL3
        OpenGL::OpenGL
)

target_compile_options(mazorca
    PUBLIC
        ${CXX_HARDENED_FLAGS}
        -fsycl
        -fsycl-targets=spir64_x86_64,amd_gpu_gfx1010
)

target_link_options(mazorca
    PUBLIC
        ${LINKER_HARDENED_FLAGS}
        -fsycl
        -fsycl-targets=spir64_x86_64,amd_gpu_gfx1010
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
