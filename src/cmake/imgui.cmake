include(FetchContent)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        6ded5230d043aa32c755e65c910c2af5002fb9f9 # v1.92.6
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    SYSTEM         TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC)
add_library(imgui::imgui ALIAS imgui)

target_sources(imgui
    PUBLIC
        FILE_SET imgui_headers
        TYPE HEADERS
        BASE_DIRS ${imgui_SOURCE_DIR}
        FILES 
            ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.h
            ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.h
    PRIVATE
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
