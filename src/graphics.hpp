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

[[nodiscard]] constexpr auto isExtensionAvailable(const std::vector<VkExtensionProperties>& properties,
                                                  const char* extension) -> bool {
    for (const VkExtensionProperties& property : properties) {
        if (std::strcmp(property.extensionName, extension) == 0) {
            return true;
        }
    }
    return false;
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
};

constexpr auto mazorca::vulkan_data::setup_vulkan() -> std::expected<void, mazorca::error_code> {
    std::vector<const char*> extensions{};
    {
        std::uint32_t sdl_extensions_count = 0;
        const char* const* sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_extensions_count);

        if (sdl_extensions == nullptr) {
            std::println("[{}] [ERROR] SDL_Vulkan_GetInstanceExtensions(): {}", mazorca::current_time(),
                         SDL_GetError());
            return std::unexpected(mazorca::error_code::invalid);
        }

        extensions.reserve(sdl_extensions_count);
        for (std::uint32_t n = 0; n < sdl_extensions_count; n++) {
            extensions.push_back(sdl_extensions[n]);
        }
    }

    VkResult err{};
    {
        // Enumerate available Vulkan extensions
        std::uint32_t properties_count = 0;
        std::vector<VkExtensionProperties> properties{};
        // If pProperties is nullptr, the number of extensions properties available is returned in pPropertyCount
        vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, nullptr);
        properties.resize(properties_count);
        err = vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, properties.data());
        mazorca::check_vk_result(err);

        // Enable required Vulkan extensions
        if (mazorca::isExtensionAvailable(properties, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }

        VkInstanceCreateInfo const create_info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                               .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
                                               .ppEnabledExtensionNames = extensions.data()};

        err = vkCreateInstance(&create_info, this->vk_allocator, &this->vk_instance);
        mazorca::check_vk_result(err);
    }

    // Select Physical Device (GPU)
    this->vk_physical_device = ImGui_ImplVulkanH_SelectPhysicalDevice(this->vk_instance);
    if (this->vk_physical_device == VK_NULL_HANDLE) {
        std::println("[{}] [ERROR] Vulkan opaque handle to physical device object points to null!",
                     mazorca::current_time());
    }

    // Select graphics queue family
    this->vk_queue_family = ImGui_ImplVulkanH_SelectQueueFamilyIndex(this->vk_physical_device);
    std::cmp_not_equal(this->vk_queue_family, -1);

    // Create Logical Device (with 1 queue)
    {
        std::vector<const char*> device_extensions{};
        device_extensions.push_back("VK_KHR_swapchain");

        // Enumerate physical device extension
        std::uint32_t properties_count = 0;
        std::vector<VkExtensionProperties> properties{};
        // If pProperties is nullptr, the number of extensions properties available is returned in pPropertyCount
        vkEnumerateDeviceExtensionProperties(this->vk_physical_device, nullptr, &properties_count, nullptr);
        properties.resize(properties_count);
        vkEnumerateDeviceExtensionProperties(this->vk_physical_device, nullptr, &properties_count, properties.data());

        // Check for extension to query a 64-bit buffer device address value for a buffer
        if (mazorca::isExtensionAvailable(properties, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
            device_extensions.push_back("VK_KHR_buffer_device_address");
        }

        constexpr std::array<float, 1> queue_priority{1.0F};
        std::array<VkDeviceQueueCreateInfo, 1> queue_info{{{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                                            .queueFamilyIndex = this->vk_queue_family,
                                                            .queueCount = 1,
                                                            .pQueuePriorities = queue_priority.data()}}};

        VkDeviceCreateInfo const create_info{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
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

constexpr void FrameRender(mazorca::vulkan_data& vulkan_data, ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data) {
    VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkResult err = vkAcquireNextImageKHR(vulkan_data.vk_device, wd->Swapchain, UINT64_MAX, image_acquired_semaphore,
                                         VK_NULL_HANDLE, &wd->FrameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        vulkan_data.vk_swap_chain_rebuild = true;
    }
    if (err == VK_ERROR_OUT_OF_DATE_KHR) {
        return;
    }
    if (err != VK_SUBOPTIMAL_KHR) {
        mazorca::check_vk_result(err);
    }

    ImGui_ImplVulkanH_Frame const* fd = &wd->Frames[wd->FrameIndex];
    {
        err = vkWaitForFences(vulkan_data.vk_device, 1, &fd->Fence, VK_TRUE,
                              UINT64_MAX);  // wait indefinitely instead of periodically checking
        mazorca::check_vk_result(err);

        err = vkResetFences(vulkan_data.vk_device, 1, &fd->Fence);
        mazorca::check_vk_result(err);
    }
    {
        err = vkResetCommandPool(vulkan_data.vk_device, fd->CommandPool, 0);
        mazorca::check_vk_result(err);
        VkCommandBufferBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
        mazorca::check_vk_result(err);
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
        VkPipelineStageFlags const wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
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
        mazorca::check_vk_result(err);
        err = vkQueueSubmit(vulkan_data.vk_queue, 1, &info, fd->Fence);
        mazorca::check_vk_result(err);
    }
}

constexpr void FramePresent(mazorca::vulkan_data& vulkan_data, ImGui_ImplVulkanH_Window* wd) {
    if (vulkan_data.vk_swap_chain_rebuild) {
        return;
    }
    VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &render_complete_semaphore;
    info.swapchainCount = 1;
    info.pSwapchains = &wd->Swapchain;
    info.pImageIndices = &wd->FrameIndex;
    VkResult const err = vkQueuePresentKHR(vulkan_data.vk_queue, &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        vulkan_data.vk_swap_chain_rebuild = true;
    }
    if (err == VK_ERROR_OUT_OF_DATE_KHR) {
        return;
    }
    if (err != VK_SUBOPTIMAL_KHR) {
        mazorca::check_vk_result(err);
    }
    wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->SemaphoreCount;  // Now we can use the next set of semaphores
}
}  // namespace mazorca
