#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>
#include <slang-com-ptr.h>
#include <slang.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <expected>
#include <filesystem>
#include <print>
#include <string>
#include <sycl/info/info_desc.hpp>
#include <vector>

#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_vulkan.h"
#include "compiler.hpp"
#include "graphics.hpp"
#include "imgui.h"
#include "mazorca/mazorca.hpp"

auto mazorca::app::run() const -> std::expected<void, mazorca::error_code> {
    mazorca::vulkan_data vulkan_data{};

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::println("[{}] [ERROR] SDL_Init(): {}", mazorca::current_time(), SDL_GetError());
        return std::unexpected(mazorca::error_code::invalid);
    }

    float const main_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    SDL_WindowFlags const window_flags =
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    SDL_Window* window = SDL_CreateWindow("mazorca", static_cast<int>(1920 * main_scale),
                                          static_cast<int>(1080 * main_scale), window_flags);

    if (window == nullptr) {
        std::println("[{}] [ERROR] SDL_CreateWindow(): {}", mazorca::current_time(), SDL_GetError());
        return std::unexpected(mazorca::error_code::invalid);
    }

    auto result = vulkan_data.setup_vulkan();

    VkSurfaceKHR surface = nullptr;
    VkResult err;
    if (static_cast<int>(
            SDL_Vulkan_CreateSurface(window, vulkan_data.vk_instance, vulkan_data.vk_allocator, &surface)) == 0) {
        std::println("[{}] [ERROR] Failed to create Vulkan surface.", mazorca::current_time());
        return std::unexpected(mazorca::error_code::invalid);
    }

    int w = 0;
    int h = 0;
    SDL_GetWindowSize(window, &w, &h);
    ImGui_ImplVulkanH_Window* window_data = &vulkan_data.vk_main_window_data;

    vulkan_data.setup_vulkan_window(window_data, surface, w, h);
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(window);

    // Setup ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    static_cast<void>(io);
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls

    // Setup ImGui style
    ImGui::StyleColorsDark();

    // Setup scaling
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);  // Bake a fixed style scale
    style.FontScaleDpi = 2.0F;        // Set initial font scale

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo init_info{
        .Instance = vulkan_data.vk_instance,
        .PhysicalDevice = vulkan_data.vk_physical_device,
        .Device = vulkan_data.vk_device,
        .QueueFamily = vulkan_data.vk_queue_family,
        .Queue = vulkan_data.vk_queue,
        .DescriptorPool = vulkan_data.vk_descriptor_pool,
        .MinImageCount = vulkan_data.vk_min_image_count,
        .ImageCount = window_data->ImageCount,
        .PipelineCache = vulkan_data.vk_pipeline_cache,
        .PipelineInfoMain = {.RenderPass = window_data->RenderPass, .Subpass = 0, .MSAASamples = VK_SAMPLE_COUNT_1_BIT},
        .Allocator = vulkan_data.vk_allocator,
        .CheckVkResultFn = mazorca::check_vk_result};
    ImGui_ImplVulkan_Init(&init_info);

    // Background color
    ImVec4 const clear_color = ImVec4(0.45F, 0.55F, 0.60F, 1.00F);

    // Input file paths for SYCL kernels and Slang shaders run-time compilation
    std::filesystem::path kernel_bundle_file_path{};
    std::filesystem::path shader_file_path{};

    // Create global session to be used by the Slang compilation API
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    if (SLANG_FAILED(slang::createGlobalSession(globalSession.writeRef()))) {
        std::println("[{}] [ERROR] Failed to create Slang global session.", mazorca::current_time());
        return std::unexpected(mazorca::error_code::invalid);
    }

    // Local vector with SYCL device names for running kernels on various devices
    std::vector<std::string> sycl_device_names;
    sycl_device_names.reserve(granos.size());
    for (const auto& grano : this->granos) {
        sycl_device_names.emplace_back(grano.sycl_device.get_info<sycl::info::device::name>());
    }
    int sycl_device_index = 0;
    std::string shader_compiler_status_message{"OK"};

    // Main loop
    bool done = false;

    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT) {
                done = true;
            }
            if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window)) {
                done = true;
            }
        }

        // Reduce the number of frames per second while app is minimized
        if ((SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) != 0U) {
            SDL_Delay(30);
            continue;
        }

        // Resize swap chain?
        int fb_width = 0;
        int fb_height = 0;
        SDL_GetWindowSize(window, &fb_width, &fb_height);
        if (fb_width > 0 && fb_height > 0 &&
            (vulkan_data.vk_swap_chain_rebuild || vulkan_data.vk_main_window_data.Width != fb_width ||
             vulkan_data.vk_main_window_data.Height != fb_height)) {
            ImGui_ImplVulkan_SetMinImageCount(vulkan_data.vk_min_image_count);
            ImGui_ImplVulkanH_CreateOrResizeWindow(vulkan_data.vk_instance, vulkan_data.vk_physical_device,
                                                   vulkan_data.vk_device, window_data, vulkan_data.vk_queue_family,
                                                   vulkan_data.vk_allocator, fb_width, fb_height,
                                                   vulkan_data.vk_min_image_count, 0);
            vulkan_data.vk_main_window_data.FrameIndex = 0;
            vulkan_data.vk_swap_chain_rebuild = false;
        }

        // Start the Dear ImGui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("SYCL Runtime Compiler (SYCL-RTC)");

            ImGui::Combo(
                "##sycl_devices", &sycl_device_index,
                [](void* data, int idx) -> const char* {
                    const auto& value = *static_cast<const std::vector<std::string>*>(data);
                    return value[idx].c_str();
                },
                &sycl_device_names, sycl_device_names.size());

            static std::array<char, 256> input_kernel_bundle_file_path{""};
            static std::string kernel_bundle_status_message{"OK"};

            ImGui::Text("File path to SYCL kernel bundle: ");
            ImGui::SameLine();
            ImGui::InputText("##sycl_path", input_kernel_bundle_file_path.data(), input_kernel_bundle_file_path.size());

            if (ImGui::Button("Compile SYCL kernel bundle")) {
                kernel_bundle_file_path.assign(input_kernel_bundle_file_path.data());
                if (!kernel_bundle_file_path.empty()) {
                    if (kernel_bundle_file_path.extension() != ".cpp") {
                        kernel_bundle_status_message = "Invalid file extension! Valid entries include: .cpp";
                        continue;
                    }

                    if (auto kernel_bundle =
                            this->granos[sycl_device_index].create_kernel_bundle(kernel_bundle_file_path);
                        !kernel_bundle.has_value()) {
                        kernel_bundle_status_message = "Failed to create kernel bundle!";
                    } else {
                        kernel_bundle_status_message = "SYCL kernel compiled successfully!";

                        // TODO: should we add a way to execute the kernel? What if there
                        // are multiple?
                    }
                } else {
                    kernel_bundle_status_message = "File path to SYCL kernel bundle is empty!";
                }
            }

            ImGui::Text("Status: %s", kernel_bundle_status_message.c_str());
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0F / io.Framerate, io.Framerate);
            ImGui::End();
        }

        {
            ImGui::Begin("Shader Runtime Compiler (Slang)");

            static std::array<char, 256> input_shader_file_path{"/home/omega/Git/mazorca/shader/kernels.slang"};

            ImGui::Text("File path to shader file: ");
            ImGui::SameLine();
            ImGui::InputText("##slang_path", input_shader_file_path.data(), input_shader_file_path.size());

            if (ImGui::Button("Compile shaders")) {
                shader_file_path.assign(input_shader_file_path.data());
                if (!shader_file_path.empty()) {
                    if (shader_file_path.extension() != ".slang") {
                        shader_compiler_status_message = "Invalid file extension! Valid entries include: .slang";
                        continue;
                    }

                    auto slang_program = mazorca::compile_shader(shader_file_path, globalSession);
                    if (!slang_program.has_value()) {
                        shader_compiler_status_message = "Failed to compile shaders!";
                        continue;
                    }

                    shader_compiler_status_message = "Compiled neural network!";

                } else {
                    shader_compiler_status_message = "File path to shader file is empty!";
                }
            }

            ImGui::Text("Status: %s", shader_compiler_status_message.c_str());
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        const bool is_minimized = (draw_data->DisplaySize.x <= 0.0F || draw_data->DisplaySize.y <= 0.0F);
        if (!is_minimized) {
            window_data->ClearValue.color.float32[0] = clear_color.x * clear_color.w;
            window_data->ClearValue.color.float32[1] = clear_color.y * clear_color.w;
            window_data->ClearValue.color.float32[2] = clear_color.z * clear_color.w;
            window_data->ClearValue.color.float32[3] = clear_color.w;
            FrameRender(vulkan_data, window_data, draw_data);
            FramePresent(vulkan_data, window_data);
        }
    }

    // Cleanup
    std::println("[{}] [INFO] Performing app clean-up and closing", mazorca::current_time());
    err = vkDeviceWaitIdle(vulkan_data.vk_device);
    mazorca::check_vk_result(err);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    vulkan_data.cleanup_vulkan_window(window_data);
    vulkan_data.cleanup_vulkan();

    SDL_DestroyWindow(window);
    SDL_Quit();

    return {};
}
