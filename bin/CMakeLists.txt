add_executable(mazorca_main)

target_compile_options(mazorca_main
    PRIVATE
        ${CXX_HARDENED_FLAGS} 
        -fsycl-targets=amd_gpu_gfx1010
)

target_link_options(mazorca_main
    PRIVATE
        ${LINKER_HARDENED_FLAGS}
        -fsycl-targets=amd_gpu_gfx1010
)

target_sources(mazorca_main
    PRIVATE
        main.cpp
)

target_link_libraries(mazorca_main
    PRIVATE
        mazorca::mazorca
)
