#include <chrono>
#include <filesystem>
#include <fstream>

#include <sycl/sycl.hpp>
#include <SDL3/SDL.h>
#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_opengl3.h>
#include <backends/imgui_impl_opengl3_loader.h>
#include <SDL3/SDL_opengl.h>

#include <mazorca/mazorca.hpp>

int main(int argc, char* argv[]) {

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        printf("Error: SDL_Init(): %s\n", SDL_GetError());
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    // GL ES 3.0 + GLSL 300 es (WebGL 2.0)
    const char* glsl_version = "#version 300 es";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    
    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    float main_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    SDL_WindowFlags window_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    SDL_Window* window = SDL_CreateWindow("Dear ImGui SDL3+OpenGL3 example", (int)(1280 * main_scale), (int)(800 * main_scale), window_flags);
    
    if (window == nullptr)
    {
        printf("Error: SDL_CreateWindow(): %s\n", SDL_GetError());
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (gl_context == nullptr)
    {
        printf("Error: SDL_GL_CreateContext(): %s\n", SDL_GetError());
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }
    
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1); // Enable vsync
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(window);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup scaling
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);        // Bake a fixed style scale. (until we have a solution for dynamic style scaling, changing this requires resetting Style + calling this again)
    style.FontScaleDpi = main_scale;        // Set initial font scale. (using io.ConfigDpiScaleFonts=true makes this unnecessary. We leave both here for documentation purpose)

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);
    
    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Main loop
    bool done = false;
    
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT)
                done = true;
            if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window))
                done = true;
        }

        // [If using SDL_MAIN_USE_CALLBACKS: all code below would likely be your SDL_AppIterate() function]
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED)
        {
            SDL_Delay(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window)
        {
            ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }
    
    // Cleanup
    // [If using SDL_MAIN_USE_CALLBACKS: all code below would likely be your SDL_AppQuit() function]
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DestroyContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    // SDL_Init(SDL_INIT_VIDEO);

    // SDL_Window *window = nullptr;
    // SDL_Renderer *renderer = nullptr;
    
    // SDL_CreateWindowAndRenderer(
    //     "Hello, World!",
    //     800, 
    //     600,
    //     SDL_WINDOW_RESIZABLE, 
    //     &window, 
    //     &renderer
    // );

    // bool quit = false;
    // while (!quit) {
    //     SDL_Event event;
    //     while (SDL_PollEvent(&event)) {
    //         if (event.type == SDL_EVENT_QUIT) 
    //             quit = true;
    //     }
        
    //     SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    //     SDL_RenderClear(renderer);
    //     SDL_RenderPresent(renderer);
    // }

    // SDL_DestroyRenderer(renderer);
    // SDL_DestroyWindow(window);
    // SDL_Quit();
    
    if (argc != 2) {
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    // Input file paths for run-time-compiled kernels
    // TODO: check if filepath is valid?
    std::filesystem::path kernel_file_path(argv[1]);
    
    // Create SYCL devices
    sycl::device cpu_device(sycl::cpu_selector_v);
    sycl::device gpu_device;
    try {
        gpu_device = sycl::device(sycl::gpu_selector_v);
    } catch (sycl::exception const &e) {
        // TODO: how to handle no GPU? Return invalid for now...
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    // Create SYCL contexts
    sycl::context cpu_context(cpu_device);
    sycl::context gpu_context(gpu_device);

    // Create SYCL queues
    sycl::queue cpu_queue(
        cpu_context,
        cpu_device,
        mazorca::sycl_async_handler,
        sycl::property::queue::enable_profiling{}
    );
    sycl::queue gpu_queue(
        gpu_context,
        gpu_device,
        mazorca::sycl_async_handler,
        sycl::property::queue::enable_profiling{}
    );

    std::ifstream kernel_file(kernel_file_path, std::ios::binary | std::ios::ate);

    if (!kernel_file) {
        return std::to_underlying(mazorca::ReturnCode::invalid);
    }

    // Read SYCL kernel to string for kernel bundle source
    std::string sycl_source{
        std::istreambuf_iterator<char>(kernel_file), 
        std::istreambuf_iterator<char>()
    };

    // Check SYCL features that are available to each device
    mazorca::check_sycl_device_features(cpu_queue);
    mazorca::check_sycl_device_features(gpu_queue);

    // TODO: SYCL RTC currently not supported on AMD HIP backend
    auto source_bundle = sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
        cpu_queue.get_context(), 
        sycl::ext::oneapi::experimental::source_language::sycl, 
        sycl_source
    );

    auto exec_bundle = sycl::ext::oneapi::experimental::build(source_bundle);

    if(exec_bundle.ext_oneapi_has_kernel("vec_add")) {
        std::cout 
            << "SYCL kernel found on " 
            << source_bundle.get_devices()[0].get_info<sycl::info::device::name>() 
            << '\n';
    }

    // Try a sample kernel for the gpu device to make sure it works!
    constexpr int n = 10;
    int *data = sycl::malloc_shared<int>(n + 1, gpu_queue);
    std::memset(data, 0, sizeof(*data) * n);

    sycl::event e;
    for (int i = 1; i < n; i += 2) {
        e = gpu_queue.submit([&](sycl::handler &h) {
        // wait for previous device task
        e.wait();
        auto device_task = [=]() { data[i] = data[i - 1] + 1; };
        h.single_task(device_task);
        });

        gpu_queue.submit([&](sycl::handler &h) {
        // wait for device task to complete
        e.wait();
        auto host_task = [=]() { data[i + 1] = data[i] + 1; };
        h.host_task(host_task);
        });
    }
    for (int i = 0; i < n; i++)
        std::cout << i << ": " << data[i] << "\n";

    sycl::free(data, gpu_queue);

    return std::to_underlying(mazorca::ReturnCode::valid);
}
