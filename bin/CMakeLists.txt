add_executable(mazorca_main)
set_target_properties(mazorca_main PROPERTIES OUTPUT_NAME "mazorca")

target_sources(mazorca_main
    PRIVATE
        main.cpp
)

target_link_libraries(mazorca_main
    PRIVATE
        mazorca::mazorca
)

target_compile_options(mazorca_main
    PRIVATE
        ${CXX_COMPILER_FLAGS}
)

target_link_options(mazorca_main
    PRIVATE
        ${CXX_LINKER_FLAGS}
)
