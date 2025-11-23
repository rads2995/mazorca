find_package(SDL3 REQUIRED)
find_package(OpenGL REQUIRED)
include(cmake/imgui.cmake)
include(cmake/glslang.cmake)

add_library(mazorca INTERFACE)
add_library(mazorca::mazorca ALIAS mazorca)

target_sources(mazorca
    INTERFACE
        FILE_SET mazorca_headers
        TYPE HEADERS
        BASE_DIRS ./include
        FILES include/mazorca/mazorca.hpp
)

target_link_libraries(mazorca
    INTERFACE
        SDL3::SDL3
        OpenGL::OpenGL
        imgui::imgui
        glslang::glslang
        sycl
)

install(
    TARGETS mazorca glslang
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
