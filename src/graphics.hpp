#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>

#include <cstdint>

#include "mazorca/mazorca.hpp"

namespace mazorca {

constexpr auto check_vk_result(VkResult err) -> void {
    if (err != VK_SUCCESS) {
        std::println("[{}] [ERROR] VkResult = {}", mazorca::current_time(), std::to_underlying(err));
    }
}

struct vulkan_data {
    VkAllocationCallbacks* vk_allocator{nullptr};
    VkInstance vk_instance{VK_NULL_HANDLE};
    VkPhysicalDevice vk_physical_device{VK_NULL_HANDLE};
    VkDevice vk_device{VK_NULL_HANDLE};
    std::uint32_t vk_queue_family{static_cast<uint32_t>(-1)};
    VkQueue vk_queue{VK_NULL_HANDLE};
    VkPipelineCache vk_pipeline_cache{VK_NULL_HANDLE};
    VkDescriptorPool vk_descriptor_pool{VK_NULL_HANDLE};
    ImGui_ImplVulkanH_Window vk_main_window_data;
    std::uint32_t vk_min_image_count{2};
    bool vk_swap_chain_rebuild{false};
    VkDescriptorSetLayout vk_descriptor_set_layout{VK_NULL_HANDLE};

    explicit vulkan_data() = default;

    [[nodiscard]] constexpr auto setup_vulkan() -> std::expected<void, mazorca::error_code>;

    [[nodiscard]] constexpr auto cleanup_vulkan() const -> std::expected<void, mazorca::error_code>;

    [[nodiscard]] constexpr auto cleanup_vulkan_window(ImGui_ImplVulkanH_Window* window_data) const
        -> std::expected<void, mazorca::error_code>;

    [[nodiscard]] constexpr auto setup_vulkan_window(ImGui_ImplVulkanH_Window* window_data, VkSurfaceKHR surface,
                                                     int width, int height) const
        -> std::expected<void, mazorca::error_code>;

    [[nodiscard]] constexpr auto render_frame(ImGui_ImplVulkanH_Window* window_data, ImDrawData* draw_data)
        -> std::expected<void, mazorca::error_code>;

    [[nodiscard]] constexpr auto present_frame(ImGui_ImplVulkanH_Window* window_data)
        -> std::expected<void, mazorca::error_code>;
};

constexpr auto mazorca::vulkan_data::setup_vulkan() -> std::expected<void, mazorca::error_code> {
    std::vector<const char*> extensions{};
    std::vector<const char*> enabled_validation_layers{};
    {
        // Enumerate SDL3 extensions for Vulkan
        std::uint32_t sdl_extensions_count = 0;
        const char* const* sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_extensions_count);

        if (sdl_extensions == nullptr) {
            std::println("[{}] [ERROR] SDL_Vulkan_GetInstanceExtensions(): {}", mazorca::current_time(),
                         SDL_GetError());
            return std::unexpected(mazorca::error_code::invalid);
        }

        extensions.assign(sdl_extensions, sdl_extensions + sdl_extensions_count);
    }

    VkResult err{};
    {
        // Describe Vulkan application information
        constexpr VkApplicationInfo application_info{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                                     .applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
                                                     .engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
                                                     .apiVersion = VK_API_VERSION_1_4};

        // Enumerate available Vulkan validation layers
        std::uint32_t layers_count = 0;
        // If pProperties is nullptr, the number of layer properties available is returned in pPropertyCount
        vkEnumerateInstanceLayerProperties(&layers_count, nullptr);
        std::vector<VkLayerProperties> layer_properties{layers_count};
        err = vkEnumerateInstanceLayerProperties(&layers_count, layer_properties.data());
        mazorca::check_vk_result(err);

        for (const auto& layer : layer_properties) {
            std::println("[{}] [INFO] Vulkan validation layer available: {}", mazorca::current_time(), layer.layerName);
        }

#ifndef NDEBUG
        // Enable Vulkan validation layers (Debug builds only)
        constexpr const char* validation_layer{"VK_LAYER_KHRONOS_validation"};
        if (auto it =
                std::ranges::find(layer_properties, std::string_view{validation_layer}, &VkLayerProperties::layerName);
            it != layer_properties.end()) {
            std::println("[{}] [INFO] Vulkan validation layer enabled: {}", mazorca::current_time(), it->layerName);
            enabled_validation_layers.push_back(validation_layer);
        }
#endif

        // Enumerate available Vulkan extensions
        std::uint32_t properties_count = 0;
        std::vector<VkExtensionProperties> properties{};
        // If pProperties is nullptr, the number of extensions properties available is returned in pPropertyCount
        vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, nullptr);
        properties.resize(properties_count);
        err = vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, properties.data());
        mazorca::check_vk_result(err);

        for (const auto& property : properties) {
            std::println("[{}] [INFO] Vulkan extension available: {}", mazorca::current_time(), property.extensionName);
        }

        // Enable required Vulkan extensions
        constexpr std::array<const char*, 3> const desired_extensions{
            "VK_KHR_surface", "VK_KHR_get_physical_device_properties2", "VK_KHR_wayland_surface"};
        for (const auto& desired_extension : desired_extensions) {
            if (auto it = std::ranges::find(properties, std::string_view{desired_extension},
                                            &VkExtensionProperties::extensionName);
                it != properties.end()) {
                std::println("[{}] [INFO] Vulkan extension enabled: {}", mazorca::current_time(), it->extensionName);
                extensions.push_back(desired_extension);
            }
        }

        VkInstanceCreateInfo const create_info{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &application_info,
            .enabledLayerCount = static_cast<std::uint32_t>(enabled_validation_layers.size()),
            .ppEnabledLayerNames = enabled_validation_layers.data(),
            .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data()};

        err = vkCreateInstance(&create_info, this->vk_allocator, &this->vk_instance);
        mazorca::check_vk_result(err);
    }

    // Select Physical Device (GPU)
    {
        std::uint32_t gpu_count = 0;
        err = vkEnumeratePhysicalDevices(this->vk_instance, &gpu_count, nullptr);
        mazorca::check_vk_result(err);
        assert(gpu_count > 0);

        std::vector<VkPhysicalDevice> gpus{};
        gpus.resize(gpu_count);
        err = vkEnumeratePhysicalDevices(this->vk_instance, &gpu_count, gpus.data());
        mazorca::check_vk_result(err);

        for (const VkPhysicalDevice& gpu_device : gpus) {
            VkPhysicalDeviceProperties2 device_properties{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            };

            vkGetPhysicalDeviceProperties2(gpu_device, &device_properties);
            if (device_properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                this->vk_physical_device = gpu_device;
                break;
            }
        }

        if (this->vk_physical_device == VK_NULL_HANDLE) {
            std::println("[{}] [ERROR] Vulkan opaque handle to physical device object points to nullptr!",
                         mazorca::current_time());
            this->vk_physical_device = gpus[0];
        }
    }

    // Select graphics queue family
    this->vk_queue_family = ImGui_ImplVulkanH_SelectQueueFamilyIndex(this->vk_physical_device);
    assert(std::cmp_not_equal(this->vk_queue_family, -1));

    // Create Logical Device (with 1 queue)
    {
        // Enumerate physical device extension
        std::uint32_t properties_count = 0;
        std::vector<VkExtensionProperties> properties{};
        // If pProperties is nullptr, the number of extensions properties available is returned in pPropertyCount
        vkEnumerateDeviceExtensionProperties(this->vk_physical_device, nullptr, &properties_count, nullptr);
        properties.resize(properties_count);
        vkEnumerateDeviceExtensionProperties(this->vk_physical_device, nullptr, &properties_count, properties.data());

        for (const auto& property : properties) {
            std::println("[{}] [INFO] Vulkan device extension available: {}", mazorca::current_time(),
                         property.extensionName);
        }

        // Enable required Vulkan device extensions
        constexpr std::array<const char*, 2> const device_extensions{"VK_KHR_swapchain",
                                                                     "VK_KHR_buffer_device_address"};
        for (const auto& device_extension : device_extensions) {
            if (auto it = std::ranges::find(properties, std::string_view{device_extension},
                                            &VkExtensionProperties::extensionName);
                it != properties.end()) {
                std::println("[{}] [INFO] Vulkan device extension enabled: {}", mazorca::current_time(),
                             it->extensionName);
                extensions.push_back(device_extension);
            }
        }

        constexpr std::array<float, 1> queue_priority{1.0F};
        std::array<VkDeviceQueueCreateInfo, 1> queue_info{{{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                                            .queueFamilyIndex = this->vk_queue_family,
                                                            .queueCount = 1,
                                                            .pQueuePriorities = queue_priority.data()}}};

        VkPhysicalDeviceVulkan13Features vulkan_13_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .dynamicRendering = 1U};

        VkDeviceCreateInfo const create_info{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                             .pNext = &vulkan_13_features,
                                             .queueCreateInfoCount = static_cast<uint32_t>(queue_info.size()),
                                             .pQueueCreateInfos = queue_info.data(),
                                             .enabledExtensionCount = static_cast<uint32_t>(device_extensions.size()),
                                             .ppEnabledExtensionNames = device_extensions.data()};

        err = vkCreateDevice(this->vk_physical_device, &create_info, this->vk_allocator, &this->vk_device);
        check_vk_result(err);
        vkGetDeviceQueue(this->vk_device, this->vk_queue_family, 0, &this->vk_queue);
    }

    // Create Descriptor Pool
    // If you wish to load e.g. additional textures you may need to alter pools sizes and maxSets.
    {
        std::array<VkDescriptorPoolSize, 1> pool_sizes{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE},
        };

        std::uint32_t pool_size_count = 0;
        for (const VkDescriptorPoolSize& pool_size : pool_sizes) {
            pool_size_count += pool_size.descriptorCount;
        }

        VkDescriptorPoolCreateInfo const pool_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                   .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                                   .maxSets = pool_size_count,
                                                   .poolSizeCount = pool_sizes.size(),
                                                   .pPoolSizes = pool_sizes.data()};

        err = vkCreateDescriptorPool(this->vk_device, &pool_info, this->vk_allocator, &this->vk_descriptor_pool);
        check_vk_result(err);
    }

    return {};
}

constexpr auto mazorca::vulkan_data::cleanup_vulkan() const -> std::expected<void, mazorca::error_code> {
    vkDestroyDescriptorPool(this->vk_device, this->vk_descriptor_pool, this->vk_allocator);
    vkDestroyDevice(this->vk_device, this->vk_allocator);
    vkDestroyInstance(this->vk_instance, this->vk_allocator);

    return {};
}

constexpr auto mazorca::vulkan_data::cleanup_vulkan_window(ImGui_ImplVulkanH_Window* window_data) const
    -> std::expected<void, mazorca::error_code> {
    ImGui_ImplVulkanH_DestroyWindow(this->vk_instance, this->vk_device, window_data, this->vk_allocator);

    return {};
}

constexpr auto mazorca::vulkan_data::setup_vulkan_window(ImGui_ImplVulkanH_Window* window_data, VkSurfaceKHR surface,
                                                         int width, int height) const
    -> std::expected<void, mazorca::error_code> {
    window_data->Surface = surface;

    // Check for WSI support
    VkBool32 res = 0;
    vkGetPhysicalDeviceSurfaceSupportKHR(this->vk_physical_device, this->vk_queue_family, window_data->Surface, &res);

    if (res != VK_TRUE) {
        std::println("[{}] [ERROR] no WSI support on physical device.", mazorca::current_time());
    }

    // Select surface format
    constexpr std::array<VkFormat, 4> requestSurfaceImageFormat{VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
                                                                VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
    constexpr VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    window_data->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        this->vk_physical_device, window_data->Surface, requestSurfaceImageFormat.data(),
        requestSurfaceImageFormat.size(), requestSurfaceColorSpace);

    constexpr std::array<VkPresentModeKHR, 1> present_modes{VK_PRESENT_MODE_FIFO_KHR};
    window_data->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(this->vk_physical_device, window_data->Surface,
                                                                   present_modes.data(), present_modes.size());

    IM_ASSERT(this->vk_min_image_count >= 2);
    ImGui_ImplVulkanH_CreateOrResizeWindow(this->vk_instance, this->vk_physical_device, this->vk_device, window_data,
                                           this->vk_queue_family, this->vk_allocator, width, height,
                                           this->vk_min_image_count, 0);

    return {};
}

constexpr auto mazorca::vulkan_data::render_frame(ImGui_ImplVulkanH_Window* window_data, ImDrawData* draw_data)
    -> std::expected<void, mazorca::error_code> {
    VkSemaphore image_acquired_semaphore =
        window_data->FrameSemaphores[window_data->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete_semaphore =
        window_data->FrameSemaphores[window_data->SemaphoreIndex].RenderCompleteSemaphore;

    VkResult err = vkAcquireNextImageKHR(this->vk_device, window_data->Swapchain, UINT64_MAX, image_acquired_semaphore,
                                         VK_NULL_HANDLE, &window_data->FrameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        this->vk_swap_chain_rebuild = true;
    }
    if (err == VK_ERROR_OUT_OF_DATE_KHR) {
        std::println("[{}] [ERROR] VK_ERROR_OUT_OF_DATE_KHR.", mazorca::current_time());
    }
    if (err != VK_SUBOPTIMAL_KHR) {
        mazorca::check_vk_result(err);
    }

    ImGui_ImplVulkanH_Frame const* frame_data = &window_data->Frames[window_data->FrameIndex];
    {
        err = vkWaitForFences(this->vk_device, 1, &frame_data->Fence, VK_TRUE,
                              UINT64_MAX);  // wait indefinitely instead of periodically checking
        mazorca::check_vk_result(err);

        err = vkResetFences(this->vk_device, 1, &frame_data->Fence);
        mazorca::check_vk_result(err);
    }
    {
        err = vkResetCommandPool(this->vk_device, frame_data->CommandPool, 0);
        mazorca::check_vk_result(err);
        VkCommandBufferBeginInfo const info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        err = vkBeginCommandBuffer(frame_data->CommandBuffer, &info);
        mazorca::check_vk_result(err);
    }
    {
        VkRenderPassBeginInfo const info = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                            .renderPass = window_data->RenderPass,
                                            .framebuffer = frame_data->Framebuffer,
                                            .renderArea.extent.width = static_cast<uint32_t>(window_data->Width),
                                            .renderArea.extent.height = static_cast<uint32_t>(window_data->Height),
                                            .clearValueCount = 1,
                                            .pClearValues = &window_data->ClearValue};
        vkCmdBeginRenderPass(frame_data->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
    }

    // Record dear imgui primitives into command buffer
    ImGui_ImplVulkan_RenderDrawData(draw_data, frame_data->CommandBuffer);

    // Submit command buffer
    vkCmdEndRenderPass(frame_data->CommandBuffer);
    {
        VkPipelineStageFlags const wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo const info = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                   .waitSemaphoreCount = 1,
                                   .pWaitSemaphores = &image_acquired_semaphore,
                                   .pWaitDstStageMask = &wait_stage,
                                   .commandBufferCount = 1,
                                   .pCommandBuffers = &frame_data->CommandBuffer,
                                   .signalSemaphoreCount = 1,
                                   .pSignalSemaphores = &render_complete_semaphore};
        err = vkEndCommandBuffer(frame_data->CommandBuffer);
        mazorca::check_vk_result(err);
        err = vkQueueSubmit(this->vk_queue, 1, &info, frame_data->Fence);
        mazorca::check_vk_result(err);
    }

    return {};
}

constexpr auto mazorca::vulkan_data::present_frame(ImGui_ImplVulkanH_Window* window_data)
    -> std::expected<void, mazorca::error_code> {
    if (this->vk_swap_chain_rebuild) {
        return {};
    }

    VkSemaphore render_complete_semaphore =
        window_data->FrameSemaphores[window_data->SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR const info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                .waitSemaphoreCount = 1,
                                .pWaitSemaphores = &render_complete_semaphore,
                                .swapchainCount = 1,
                                .pSwapchains = &window_data->Swapchain,
                                .pImageIndices = &window_data->FrameIndex};

    VkResult const err = vkQueuePresentKHR(this->vk_queue, &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        this->vk_swap_chain_rebuild = true;
    }
    if (err == VK_ERROR_OUT_OF_DATE_KHR) {
        std::println("[{}] [ERROR] VK_ERROR_OUT_OF_DATE_KHR.", mazorca::current_time());
        return {};
    }
    if (err != VK_SUBOPTIMAL_KHR) {
        mazorca::check_vk_result(err);
    }
    window_data->SemaphoreIndex =
        (window_data->SemaphoreIndex + 1) % window_data->SemaphoreCount;  // Now we can use the next set of semaphores

    return {};
}

}  // namespace mazorca
