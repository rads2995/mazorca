include(cmake/imgui.cmake)
include(cmake/slang.cmake)
include(cmake/oneDNN.cmake)
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
        grano.cpp
        compiler.cpp
        training.cpp
        inference.cpp
        vulkan.cpp
        app.cpp
)

target_link_libraries(mazorca
    PRIVATE
        $<BUILD_LOCAL_INTERFACE:imgui::imgui>
        $<BUILD_LOCAL_INTERFACE:glad::glad>
        $<BUILD_LOCAL_INTERFACE:slang>
        $<BUILD_LOCAL_INTERFACE:slang-rhi>
        $<BUILD_LOCAL_INTERFACE:dnnl>
        SDL3::SDL3
        OpenGL::OpenGL
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
