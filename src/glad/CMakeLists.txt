add_library(glad STATIC)
add_library(glad::glad ALIAS glad)

target_sources(glad
    PUBLIC
        FILE_SET glad_headers
        TYPE HEADERS
        BASE_DIRS ./include
        FILES 
            include/glad/glad.h
            include/KHR/khrplatform.h
    PRIVATE
        src/glad.c
)
