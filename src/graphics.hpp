#pragma once

#include <cstdint>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>

namespace mazorca {

struct vulkan_data {
    VkAllocationCallbacks* vk_allocator;
    VkInstance vk_instance;
    VkPhysicalDevice vk_physical_device;
    VkDevice vk_device;
    std::uint32_t vk_queue_family;
    VkQueue vk_queue;
    VkPipelineCache vk_pipeline_cache;
    VkDescriptorPool vk_descriptor_pool;
    ImGui_ImplVulkanH_Window vk_main_window_data;
    std::uint32_t vk_min_image_count;
    bool vk_swap_chain_rebuild;
};

void check_vk_result(VkResult err);
bool isExtensionAvailable(const ImVector<VkExtensionProperties>& properties, const char* extension);
void CleanupVulkan(mazorca::vulkan_data& vulkan_data);
void CleanupVulkanWindow(mazorca::vulkan_data& vulkan_data);
void SetupVulkan(vulkan_data& vulkan_data, ImVector<const char*> instance_extensions);
void SetupVulkanWindow(vulkan_data& vulkan_data, ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height);
void FrameRender(mazorca::vulkan_data& vulkan_data, ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data);
void FramePresent(mazorca::vulkan_data& vulkan_data, ImGui_ImplVulkanH_Window* wd);

}
