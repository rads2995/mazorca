#include <mazorca/mazorca.hpp>
#include "compiler.hpp"

#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_opengl3.h>

std::expected<void, mazorca::error_code> mazorca::app::run() {

    // Test run-time compilation of example shader
    auto spirv_code = mazorca::compile_shader();
    if (!spirv_code.has_value()) {
        return std::unexpected(mazorca::error_code::invalid);
    }

    // Initialize the SDL library
    if (!SDL_Init(SDL_INIT_AUDIO | SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        std::cout << "Error: SDL_Init(): " << SDL_GetError() << '\n';
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

    gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress);

    // Setup ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); static_cast<void>(io);
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

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
    
    // Our state
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    
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
        std::cout << "ok?" << std::endl;
    } else {
        std::cout << "not ok :(" << std::endl;
    }

    // This should now return TRUE
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

    // Create a program, attach our shader to it, and link
    GLuint program = glCreateProgram();

    glAttachShader(program, shader);

    glLinkProgram(program);

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

        // TODO: is this to reduce the number of frames while app is minimized?
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED ) {
            SDL_Delay(30);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // Use a Begin/End pair to create a named window
        {
            static int counter = 0;

            ImGui::Begin("Sample Window");  // Create a window

            if (ImGui::Button("Button"))    // Buttons return true when clicked
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
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
