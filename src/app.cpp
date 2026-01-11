#include <mazorca/mazorca.hpp>
#include "compiler.hpp"
#include "training.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>

// Vulkan Data
static VkAllocationCallbacks* g_Allocator = nullptr;
static VkInstance g_Instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_PhysicalDevice = VK_NULL_HANDLE;
static VkDevice g_Device = VK_NULL_HANDLE;
static uint32_t g_QueueFamily = static_cast<uint32_t>(-1);
static VkQueue g_Queue = VK_NULL_HANDLE;
static VkPipelineCache g_PipelineCache = VK_NULL_HANDLE;
static VkDescriptorPool g_DescriptorPool = VK_NULL_HANDLE;

static ImGui_ImplVulkanH_Window g_MainWindowData;
static uint32_t g_MinImageCount = 2;
static bool g_SwapChainRebuild = false;

static void check_vk_result(VkResult err) {
    if (err != VK_SUCCESS) {
        std::println("[vulkan] Error: VkResult = {}", std::to_underlying(err));
    }
}

static bool isExtensionAvailable(const ImVector<VkExtensionProperties>& properties, const char* extension) {
    for (const VkExtensionProperties& property : properties) {
        if (strcmp(property.extensionName, extension) == 0) {
            return true;
        }
    }
    return false;
}

static void SetupVulkan(ImVector<const char*> instance_extensions) {

    VkResult err;

    // Create Vulkan Instance
    {
        VkInstanceCreateInfo create_info {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
        };

        // Enumerate available extensions
        uint32_t properties_count;
        ImVector<VkExtensionProperties> properties;
        vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, nullptr);
        properties.resize(properties_count);
        err = vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, properties.Data);
        check_vk_result(err);

        // Enable required extensions
        if (isExtensionAvailable(properties, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }

        // Create Vulkan Instance
        create_info.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.Size);
        create_info.ppEnabledExtensionNames = instance_extensions.Data;
        err = vkCreateInstance(&create_info, g_Allocator, &g_Instance);
        check_vk_result(err);
    }

    // Select Physical Device (GPU)
    g_PhysicalDevice = ImGui_ImplVulkanH_SelectPhysicalDevice(g_Instance);
    if(g_PhysicalDevice == VK_NULL_HANDLE) {
        std::println("[Error] Vulkan opaque handle to physical device object points to null!");
    }

    // Select graphics queue family
    g_QueueFamily = ImGui_ImplVulkanH_SelectQueueFamilyIndex(g_PhysicalDevice);
    IM_ASSERT(g_QueueFamily != static_cast<uint32_t>(-1));

    // Create Logical Device (with 1 queue)
    {
        ImVector<const char*> device_extensions;
        device_extensions.push_back("VK_KHR_swapchain");

        // Enumerate physical device extension
        uint32_t properties_count;
        ImVector<VkExtensionProperties> properties;
        vkEnumerateDeviceExtensionProperties(g_PhysicalDevice, nullptr, &properties_count, nullptr);
        properties.resize(properties_count);
        vkEnumerateDeviceExtensionProperties(g_PhysicalDevice, nullptr, &properties_count, properties.Data);

        const float queue_priority[] = { 1.0f };
        VkDeviceQueueCreateInfo queue_info[1] = {};
        queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info[0].queueFamilyIndex = g_QueueFamily;
        queue_info[0].queueCount = 1;
        queue_info[0].pQueuePriorities = queue_priority;
        VkDeviceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount = std::size(queue_info);
        create_info.pQueueCreateInfos = queue_info;
        create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.Size);
        create_info.ppEnabledExtensionNames = device_extensions.Data;
        err = vkCreateDevice(g_PhysicalDevice, &create_info, g_Allocator, &g_Device);
        check_vk_result(err);
        vkGetDeviceQueue(g_Device, g_QueueFamily, 0, &g_Queue);
    }

    // Create Descriptor Pool
    // If you wish to load e.g. additional textures you may need to alter pools sizes and maxSets.
    {
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE },
        };
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 0;
        for (VkDescriptorPoolSize& pool_size : pool_sizes) {
            pool_info.maxSets += pool_size.descriptorCount;
        }
        pool_info.poolSizeCount = std::size(pool_sizes);
        pool_info.pPoolSizes = pool_sizes;
        err = vkCreateDescriptorPool(g_Device, &pool_info, g_Allocator, &g_DescriptorPool);
        check_vk_result(err);
    }
}

// All the ImGui_ImplVulkanH_XXX structures/functions are optional helpers used by the demo.
// Your real engine/app may not use them.
static void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height) {
    wd->Surface = surface;

    // Check for WSI support
    VkBool32 res;
    vkGetPhysicalDeviceSurfaceSupportKHR(g_PhysicalDevice, g_QueueFamily, wd->Surface, &res);
    if (res != VK_TRUE) {
        std::println("Error no WSI support on physical device.");
        return;
    }

    // Select surface format
    const VkFormat requestSurfaceImageFormat[] = {
        VK_FORMAT_B8G8R8A8_UNORM, 
        VK_FORMAT_R8G8B8A8_UNORM, 
        VK_FORMAT_B8G8R8_UNORM, 
        VK_FORMAT_R8G8B8_UNORM
    };
    const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        g_PhysicalDevice, 
        wd->Surface, 
        requestSurfaceImageFormat, 
        std::size(requestSurfaceImageFormat), 
        requestSurfaceColorSpace
    );

    VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
    wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        g_PhysicalDevice, 
        wd->Surface, 
        &present_modes[0], 
        std::size(present_modes)
    );

    IM_ASSERT(g_MinImageCount >= 2);
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        g_Instance, 
        g_PhysicalDevice, 
        g_Device, 
        wd, 
        g_QueueFamily, 
        g_Allocator, 
        width, height, 
        g_MinImageCount, 
        0
    );
}

static void CleanupVulkan() {
    vkDestroyDescriptorPool(g_Device, g_DescriptorPool, g_Allocator);

    vkDestroyDevice(g_Device, g_Allocator);
    vkDestroyInstance(g_Instance, g_Allocator);
}

static void CleanupVulkanWindow() {
    ImGui_ImplVulkanH_DestroyWindow(g_Instance, g_Device, &g_MainWindowData, g_Allocator);
}

static void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data)
{
    VkSemaphore image_acquired_semaphore  = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkResult err = vkAcquireNextImageKHR(g_Device, wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
        g_SwapChainRebuild = true;
    if (err == VK_ERROR_OUT_OF_DATE_KHR)
        return;
    if (err != VK_SUBOPTIMAL_KHR)
        check_vk_result(err);

    ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
    {
        err = vkWaitForFences(g_Device, 1, &fd->Fence, VK_TRUE, UINT64_MAX);    // wait indefinitely instead of periodically checking
        check_vk_result(err);

        err = vkResetFences(g_Device, 1, &fd->Fence);
        check_vk_result(err);
    }
    {
        err = vkResetCommandPool(g_Device, fd->CommandPool, 0);
        check_vk_result(err);
        VkCommandBufferBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
        check_vk_result(err);
    }
    {
        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass = wd->RenderPass;
        info.framebuffer = fd->Framebuffer;
        info.renderArea.extent.width = wd->Width;
        info.renderArea.extent.height = wd->Height;
        info.clearValueCount = 1;
        info.pClearValues = &wd->ClearValue;
        vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
    }

    // Record dear imgui primitives into command buffer
    ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

    // Submit command buffer
    vkCmdEndRenderPass(fd->CommandBuffer);
    {
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &image_acquired_semaphore;
        info.pWaitDstStageMask = &wait_stage;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &fd->CommandBuffer;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &render_complete_semaphore;

        err = vkEndCommandBuffer(fd->CommandBuffer);
        check_vk_result(err);
        err = vkQueueSubmit(g_Queue, 1, &info, fd->Fence);
        check_vk_result(err);
    }
}

static void FramePresent(ImGui_ImplVulkanH_Window* wd)
{
    if (g_SwapChainRebuild)
        return;
    VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &render_complete_semaphore;
    info.swapchainCount = 1;
    info.pSwapchains = &wd->Swapchain;
    info.pImageIndices = &wd->FrameIndex;
    VkResult err = vkQueuePresentKHR(g_Queue, &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
        g_SwapChainRebuild = true;
    if (err == VK_ERROR_OUT_OF_DATE_KHR)
        return;
    if (err != VK_SUBOPTIMAL_KHR)
        check_vk_result(err);
    wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->SemaphoreCount; // Now we can use the next set of semaphores
}

std::expected<void, mazorca::error_code> mazorca::app::run() {

    // Initialize the SDL library
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::println("Error: SDL_Init(): {}", SDL_GetError());
        return std::unexpected(mazorca::error_code::invalid);
    }

    float main_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    SDL_WindowFlags window_flags = SDL_WINDOW_VULKAN | SDL_WINDOW_FULLSCREEN | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    SDL_Window* window = SDL_CreateWindow(
        "mazorca",
        static_cast<int>(1920 * main_scale), 
        static_cast<int>(1080 * main_scale), 
        window_flags
    );

    if (window == nullptr) {
        std::println("Error: SDL_CreateWindow(): {}", SDL_GetError());
        return std::unexpected(mazorca::error_code::invalid);
    }

    ImVector<const char*> extensions;
    {
        uint32_t sdl_extensions_count = 0;
        const char* const* sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_extensions_count);
        for (uint32_t n = 0; n < sdl_extensions_count; n++) {
            extensions.push_back(sdl_extensions[n]);
        }
    }
    SetupVulkan(extensions);

    // Create window surface
    VkSurfaceKHR surface;
    VkResult err;
    if (SDL_Vulkan_CreateSurface(window, g_Instance, g_Allocator, &surface) == 0) {
        std::println("Failed to create Vulkan surface.");
        return std::unexpected(mazorca::error_code::invalid);
    }
    
    // Create Framebuffers
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    ImGui_ImplVulkanH_Window* wd = &g_MainWindowData;
    SetupVulkanWindow(wd, surface, w, h);
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(window);

    // Setup ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); static_cast<void>(io);
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;   // Enable Keyboard Controls

    // Setup ImGui style
    ImGui::StyleColorsDark();

    // Setup scaling
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);    // Bake a fixed style scale
    style.FontScaleDpi = 2.0f;          // Set initial font scale

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo init_info {
        .Instance = g_Instance,
        .PhysicalDevice = g_PhysicalDevice,
        .Device = g_Device,
        .QueueFamily = g_QueueFamily,
        .Queue = g_Queue,
        .DescriptorPool = g_DescriptorPool,
        .MinImageCount = g_MinImageCount,
        .ImageCount = wd->ImageCount,
        .PipelineCache = g_PipelineCache,
        .PipelineInfoMain = {
            .RenderPass = wd->RenderPass,
            .Subpass = 0,
            .MSAASamples = VK_SAMPLE_COUNT_1_BIT
        },
        .Allocator = g_Allocator,
        .CheckVkResultFn = check_vk_result
    };
    ImGui_ImplVulkan_Init(&init_info);

    // Background color
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Input file paths for SYCL kernels and Slang shaders run-time compilation
    std::filesystem::path kernel_bundle_file_path {};
    std::filesystem::path shader_file_path {};

    // Create single global session to be used by the Slang shader compiler
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    slang::createGlobalSession(globalSession.writeRef());

    if (auto neural_net = this->granos[0].nn_example(); !neural_net.has_value()) {
        std::println("Failed to run neural network example!");
    }

    // Main loop
    bool done = false;

    // Local vector with SYCL device names for running kernels on various devices
    std::vector<std::string> sycl_device_names;
    sycl_device_names.reserve(granos.size());
    for (const auto& grano : this->granos) {
        sycl_device_names.emplace_back(grano.sycl_device.get_info<sycl::info::device::name>());
    }
    int sycl_device_index = 0;

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
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED ) {
            SDL_Delay(30);
            continue;
        }

        // Resize swap chain?
        int fb_width, fb_height;
        SDL_GetWindowSize(window, &fb_width, &fb_height);
        if (fb_width > 0 && fb_height > 0 && (g_SwapChainRebuild || g_MainWindowData.Width != fb_width || g_MainWindowData.Height != fb_height)) {
            ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
            ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance, g_PhysicalDevice, g_Device, wd, g_QueueFamily, g_Allocator, fb_width, fb_height, g_MinImageCount, 0);
            g_MainWindowData.FrameIndex = 0;
            g_SwapChainRebuild = false;
        }

        // Start the Dear ImGui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("SYCL Runtime Compiler (SYCL-RTC)");

            ImGui::Combo(
                "##sycl_devices", 
                &sycl_device_index, 
                [](void* data, int idx) -> const char* {
                    auto& value = *static_cast<const std::vector<std::string>*>(data);
                    return value[idx].c_str();
                }, 
                &sycl_device_names,
                sycl_device_names.size()
            );
            
            static std::array<char, 256> input_kernel_bundle_file_path {""};
            static std::string kernel_bundle_status_message {"OK"};

            ImGui::Text("File path to SYCL kernel bundle: ");
            ImGui::SameLine();
            ImGui::InputText(
                "##sycl_path", 
                input_kernel_bundle_file_path.data(), 
                input_kernel_bundle_file_path.size()
            );

            if (ImGui::Button("Compile SYCL kernel bundle")) {
                kernel_bundle_file_path.assign(input_kernel_bundle_file_path.data());
                if (!kernel_bundle_file_path.empty()) {

                    if (kernel_bundle_file_path.extension() != ".cpp") {
                        kernel_bundle_status_message = "Invalid file extension! Valid entries include: .cpp";
                        continue;
                    }

                    if (auto kernel_bundle = this->granos[sycl_device_index].create_kernel_bundle(kernel_bundle_file_path); !kernel_bundle.has_value()) {
                        kernel_bundle_status_message = "Failed to create kernel bundle!";
                    } else {
                        kernel_bundle_status_message = "SYCL kernel compiled successfully!";

                        // TODO: should we add a way to execute the kernel? What if there are multiple?
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
            ImGui::Begin("Shader Runtime Compiler (Slang)");
            
            static std::array<char, 256> input_shader_file_path {""};
            static std::string shader_compiler_status_message {"OK"};
            
            ImGui::Text("File path to shader file: ");
            ImGui::SameLine();
            ImGui::InputText(
                "##slang_path", 
                input_shader_file_path.data(), 
                input_shader_file_path.size()
            );

            if (ImGui::Button("Compile shaders")) {
                shader_file_path.assign(input_shader_file_path.data()); 
                if (!shader_file_path.empty()) {
                    
                    if (shader_file_path.extension() != ".slang") {
                        shader_compiler_status_message = "Invalid file extension! Valid entries include: .slang";
                        continue;
                    }

                    auto spirv_map = mazorca::compile_shader(shader_file_path, globalSession);
                    if (!spirv_map.has_value()) {
                        shader_compiler_status_message = "Failed to compile shaders!";
                    } else {
                        
                        if (true) {

                        }

                        else {
                            shader_compiler_status_message = "Provided shader file not implemented yet...";
                            continue;
                        }

                        shader_compiler_status_message = "Shaders compiled successfully!";
                    }

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
        const bool is_minimized = (
            draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f
        );
        if (!is_minimized) {
            wd->ClearValue.color.float32[0] = clear_color.x * clear_color.w;
            wd->ClearValue.color.float32[1] = clear_color.y * clear_color.w;
            wd->ClearValue.color.float32[2] = clear_color.z * clear_color.w;
            wd->ClearValue.color.float32[3] = clear_color.w;
            FrameRender(wd, draw_data);
            FramePresent(wd);
        }
    }

    // Cleanup
    std::println("Performing app clean-up and closing...");
    err = vkDeviceWaitIdle(g_Device);
    check_vk_result(err);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    CleanupVulkanWindow();
    CleanupVulkan();

    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return {};
}
