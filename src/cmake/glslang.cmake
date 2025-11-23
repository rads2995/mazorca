include(FetchContent)

set(ALLOW_EXTERNAL_SPIRV_TOOLS ON CACHE BOOL "" FORCE)
set(ENABLE_PCH OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG        a57276bf558f5cf94d3a9854ebdf5a2236849a5a # Release 16.0.0
)
FetchContent_MakeAvailable(glslang)
