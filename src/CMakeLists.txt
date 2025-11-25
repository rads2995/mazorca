find_package(SDL3 REQUIRED)
find_package(OpenGL REQUIRED)
include(cmake/imgui.cmake)

add_library(mazorca STATIC)
add_library(mazorca::mazorca ALIAS mazorca)

target_sources(mazorca
    PUBLIC
        FILE_SET mazorca_headers
        TYPE HEADERS
        BASE_DIRS ./include
        FILES include/mazorca/mazorca.hpp
    PRIVATE
        mazorca.cpp
)

target_link_libraries(mazorca
    PRIVATE
        imgui::imgui
        SDL3::SDL3
        OpenGL::OpenGL
    PUBLIC
)

target_compile_options(mazorca
    PUBLIC
        ${CXX_HARDENED_FLAGS}
        -fsycl
        -fsycl-targets=amd_gpu_gfx1010
)

target_link_options(mazorca
    PUBLIC
        ${LINKER_HARDENED_FLAGS}
        -fsycl
        -fsycl-targets=amd_gpu_gfx1010
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
