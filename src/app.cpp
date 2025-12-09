#include <mazorca/mazorca.hpp>
#include "compiler.hpp"

#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_opengl3.h>

std::expected<void, mazorca::error_code> mazorca::app::run() {

    // Initialize the SDL library
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::println("Error: SDL_Init(): {}", SDL_GetError());
        return std::unexpected(mazorca::error_code::invalid);
    }

    // OpenGL Version 4.6
    const char* glsl_version = "#version 460";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    
    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    float main_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    SDL_WindowFlags window_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    SDL_Window* window = SDL_CreateWindow(
        "mazorca",
        static_cast<int>(1920 * main_scale), 
        static_cast<int>(1080 * main_scale), 
        window_flags
    );

    if (window == nullptr) {
        std::cout << "Error: SDL_CreateWindow(): " << SDL_GetError() << '\n';
        return std::unexpected(mazorca::error_code::invalid);
    }

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (gl_context == nullptr) {
        std::cout << "Error: SDL_GL_CreateContext(): " << SDL_GetError() << '\n';
        return std::unexpected(mazorca::error_code::invalid);
    }
    
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1); // Enable vsync
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(window);

    gladLoadGLLoader((GLADloadproc) SDL_GL_GetProcAddress);

    // Setup ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); static_cast<void>(io);
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

    // Setup ImGui style
    ImGui::StyleColorsDark();

    // Setup scaling
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);        // Bake a fixed style scale
    // TODO: replace with io.ConfigDpiScaleFonts = true and io.ConfigDpiScaleViewports = true
    style.FontScaleDpi = 2.0f * main_scale;  // Set initial font scale

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);
    
    // Background color
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Main loop
    bool done = false;
    
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT)
                done = true;
            if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window))
                done = true;
        }

        // Reduce the number of frames per second while app is minimized
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED ) {
            SDL_Delay(30);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("SYCL Runtime Compilation (SYCL-RTC)");

            static std::array<char, 256> input_kernel_bundle_file_path {""};
            static std::string kernel_bundle_status_message {"OK"};

            ImGui::Text("File path to SYCL kernel bundle: ");
            ImGui::SameLine();
            ImGui::InputText(
                "", 
                input_kernel_bundle_file_path.data(), 
                input_kernel_bundle_file_path.size()
            );

            if (ImGui::Button("Compile SYCL kernel bundle")) {
                std::filesystem::path kernel_bundle_file_path{input_kernel_bundle_file_path.data()};                
                if (!kernel_bundle_file_path.empty()) {
                    if (auto result = this->granos[0].create_kernel_bundle(kernel_bundle_file_path); !result.has_value()) {
                        kernel_bundle_status_message = "Failed to create kernel bundle!";
                    } else {
                        kernel_bundle_status_message = "SYCL kernel compiled successfully!";
                    }
                } else {
                    kernel_bundle_status_message = "File path to SYCL kernel bundle is empty!";
                }
            }
            ImGui::Text("Status: %s", kernel_bundle_status_message.c_str());

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        {
            ImGui::Begin("Shader Compilation");
            
            static std::string shader_compiler_status_message {"OK"};
            
            if (ImGui::Button("Compile shaders")) {
                
                auto spirv_code = mazorca::compile_shader();
                if (!spirv_code.has_value()) {
                    shader_compiler_status_message = "Failed to compile shaders!";
                } else {
                    shader_compiler_status_message = "Shaders compiled successfully!";
                }

                // Create the shader object
                GLuint shader = glCreateShader(GL_COMPUTE_SHADER);

                // Load the SPIR-V module into the shader object
                glShaderBinary(
                    1, 
                    &shader,
                    GL_SHADER_BINARY_FORMAT_SPIR_V,
                    spirv_code.value()->getBufferPointer(), 
                    static_cast<GLsizei>(spirv_code.value()->getBufferSize())
                );

                glSpecializeShader(
                    shader,
                    "main",
                    0,
                    nullptr,
                    nullptr
                );

                // This will now return FALSE
                GLint status;
                glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
                if (status) {
                    std::cout << "ok!" << std::endl;
                } else {
                    std::cout << "not ok :(" << std::endl;
                }

                // This should now return TRUE
                glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

                // Create a program, attach our shader to it, and link
                GLuint program = glCreateProgram();

                glAttachShader(program, shader);

                glLinkProgram(program);

                glGetShaderiv(shader, GL_LINK_STATUS, &status);
                if (status) {
                    std::cout << "ok!" << std::endl;
                } else {
                    std::cout << "not ok :(" << std::endl;
                }

                GLuint resultBuffer = 0;
                glGenBuffers(1, &resultBuffer);
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuffer);
                glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * 1024, nullptr, GL_DYNAMIC_COPY);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, resultBuffer);

                glUseProgram(program);
                glDispatchCompute(128, 1, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                // DEBUG READ
                float* ptr = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * 128, GL_MAP_READ_BIT);
                for (int i = 0; i < 128; ++i)
                    std::cout << ptr[i] << " ";
                std::cout << std::endl;
                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            }
            
            ImGui::Text("Status: %s", shader_compiler_status_message.c_str());

            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        glViewport(
            0, 
            0, 
            static_cast<int>(io.DisplaySize.x), 
            static_cast<int>(io.DisplaySize.y)
        );
        glClearColor(
            clear_color.x * clear_color.w, 
            clear_color.y * clear_color.w, 
            clear_color.z * clear_color.w, 
            clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }
    
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DestroyContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return {};
}
