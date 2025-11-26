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
