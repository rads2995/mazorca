add_library(mazorca INTERFACE)
add_library(mazorca::mazorca ALIAS mazorca)

target_sources(mazorca
    PUBLIC
        FILE_SET mazorca_headers
        TYPE HEADERS
        BASE_DIRS ./include
        FILES include/mazorca/mazorca.hpp
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
    FILES
        ./cmake/mazorcaConfig.cmake
    DESTINATION
        share/cmake/mazorca
)
