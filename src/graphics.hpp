#pragma once

#include "mazorca/mazorca.hpp"

#include <cstdint>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>

namespace mazorca {

struct vulkan_data {
  VkAllocationCallbacks* vk_allocator{};
  VkInstance vk_instance{};
  VkPhysicalDevice vk_physical_device{};
  VkDevice vk_device{};
  std::uint32_t vk_queue_family{};
  VkQueue vk_queue{};
  VkPipelineCache vk_pipeline_cache{};
  VkDescriptorPool vk_descriptor_pool{};
  ImGui_ImplVulkanH_Window vk_main_window_data;
  std::uint32_t vk_min_image_count{};
  bool vk_swap_chain_rebuild{};
  VkDescriptorSetLayout vk_descriptor_set_layout{};
};

constexpr void check_vk_result(VkResult err) {
  if (err != VK_SUCCESS) {
    std::println("[vulkan] Error: VkResult = {}", std::to_underlying(err));
  }
}

[[nodiscard]] constexpr auto isExtensionAvailable(
    const ImVector<VkExtensionProperties>& properties, const char* extension)
    -> bool {
  for (const VkExtensionProperties& property : properties) {
    if (strcmp(property.extensionName, extension) == 0) {
      return true;
    }
  }
  return false;
}

constexpr void CleanupVulkan(mazorca::vulkan_data& vulkan_data) {
  vkDestroyDescriptorPool(vulkan_data.vk_device, vulkan_data.vk_descriptor_pool,
                          vulkan_data.vk_allocator);
  vkDestroyDevice(vulkan_data.vk_device, vulkan_data.vk_allocator);
  vkDestroyInstance(vulkan_data.vk_instance, vulkan_data.vk_allocator);
}

constexpr void CleanupVulkanWindow(mazorca::vulkan_data& vulkan_data) {
  ImGui_ImplVulkanH_DestroyWindow(
      vulkan_data.vk_instance, vulkan_data.vk_device,
      &vulkan_data.vk_main_window_data, vulkan_data.vk_allocator);
}

constexpr void SetupVulkan(mazorca::vulkan_data& vulkan_data,
                           ImVector<const char*> instance_extensions) {
  VkResult err;
  {
    VkInstanceCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};

    // Enumerate available extensions
    std::uint32_t properties_count = 0;
    ImVector<VkExtensionProperties> properties;
    vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, nullptr);
    properties.resize(properties_count);
    err = vkEnumerateInstanceExtensionProperties(nullptr, &properties_count,
                                                 properties.Data);
    mazorca::check_vk_result(err);

    // Enable required extensions
    if (mazorca::isExtensionAvailable(
            properties,
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
      instance_extensions.push_back(
          VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }

    // Create Vulkan Instance
    create_info.enabledExtensionCount =
        static_cast<uint32_t>(instance_extensions.Size);
    create_info.ppEnabledExtensionNames = instance_extensions.Data;
    err = vkCreateInstance(&create_info, vulkan_data.vk_allocator,
                           &vulkan_data.vk_instance);
    mazorca::check_vk_result(err);
  }

  // Select Physical Device (GPU)
  vulkan_data.vk_physical_device =
      ImGui_ImplVulkanH_SelectPhysicalDevice(vulkan_data.vk_instance);
  if (vulkan_data.vk_physical_device == VK_NULL_HANDLE) {
    std::println(
        "[{}] [ERROR] Vulkan opaque handle to physical device object points to "
        "null!",
        mazorca::current_time());
  }

  // Select graphics queue family
  vulkan_data.vk_queue_family =
      ImGui_ImplVulkanH_SelectQueueFamilyIndex(vulkan_data.vk_physical_device);
  IM_ASSERT(std::cmp_not_equal(vulkan_data.vk_queue_family, -1));

  // Create Logical Device (with 1 queue)
  {
    ImVector<const char*> device_extensions;
    device_extensions.push_back("VK_KHR_swapchain");

    // Enumerate physical device extension
    std::uint32_t properties_count = 0;
    ImVector<VkExtensionProperties> properties;
    vkEnumerateDeviceExtensionProperties(vulkan_data.vk_physical_device,
                                         nullptr, &properties_count, nullptr);
    properties.resize(properties_count);
    vkEnumerateDeviceExtensionProperties(vulkan_data.vk_physical_device,
                                         nullptr, &properties_count,
                                         properties.Data);

    // Check for extension to query a 64-bit buffer device address value for a
    // buffer
    if (mazorca::isExtensionAvailable(
            properties, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
      device_extensions.push_back("VK_KHR_buffer_device_address");
    }

    const float queue_priority[] = {1.0F};
    VkDeviceQueueCreateInfo queue_info[1] = {};
    queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info[0].queueFamilyIndex = vulkan_data.vk_queue_family;
    queue_info[0].queueCount = 1;
    queue_info[0].pQueuePriorities = queue_priority;
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = std::size(queue_info);
    create_info.pQueueCreateInfos = queue_info;
    create_info.enabledExtensionCount =
        static_cast<std::uint32_t>(device_extensions.Size);
    create_info.ppEnabledExtensionNames = device_extensions.Data;
    err = vkCreateDevice(vulkan_data.vk_physical_device, &create_info,
                         vulkan_data.vk_allocator, &vulkan_data.vk_device);
    check_vk_result(err);
    vkGetDeviceQueue(vulkan_data.vk_device, vulkan_data.vk_queue_family, 0,
                     &vulkan_data.vk_queue);
  }

  // Create Descriptor Pool
  // If you wish to load e.g. additional textures you may need to alter pools
  // sizes and maxSets.
  {
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE},
    };
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 0;
    for (VkDescriptorPoolSize const& pool_size : pool_sizes) {
      pool_info.maxSets += pool_size.descriptorCount;
    }
    pool_info.poolSizeCount = std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    err = vkCreateDescriptorPool(vulkan_data.vk_device, &pool_info,
                                 vulkan_data.vk_allocator,
                                 &vulkan_data.vk_descriptor_pool);
    check_vk_result(err);
  }
}

// All the ImGui_ImplVulkanH_XXX structures/functions are optional helpers used by the demo.
// Your real engine/app may not use them.
constexpr void SetupVulkanWindow(mazorca::vulkan_data& vulkan_data,
                                 ImGui_ImplVulkanH_Window* wd,
                                 VkSurfaceKHR surface, int width, int height) {
  wd->Surface = surface;

  // Check for WSI support
  VkBool32 res = 0;
  vkGetPhysicalDeviceSurfaceSupportKHR(vulkan_data.vk_physical_device,
                                       vulkan_data.vk_queue_family, wd->Surface,
                                       &res);
  if (res != VK_TRUE) {
    std::println("[{}] [ERROR] no WSI support on physical device.",
                 mazorca::current_time());
    return;
  }

  // Select surface format
  const VkFormat requestSurfaceImageFormat[] = {
      VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
  const VkColorSpaceKHR requestSurfaceColorSpace =
      VK_COLORSPACE_SRGB_NONLINEAR_KHR;
  wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
      vulkan_data.vk_physical_device, wd->Surface, requestSurfaceImageFormat,
      std::size(requestSurfaceImageFormat), requestSurfaceColorSpace);

  VkPresentModeKHR const present_modes[] = {VK_PRESENT_MODE_FIFO_KHR};
  wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
      vulkan_data.vk_physical_device, wd->Surface, &present_modes[0],
      std::size(present_modes));

  IM_ASSERT(vulkan_data.vk_min_image_count >= 2);
  ImGui_ImplVulkanH_CreateOrResizeWindow(
      vulkan_data.vk_instance, vulkan_data.vk_physical_device,
      vulkan_data.vk_device, wd, vulkan_data.vk_queue_family,
      vulkan_data.vk_allocator, width, height, vulkan_data.vk_min_image_count,
      0);
}

constexpr void FrameRender(mazorca::vulkan_data& vulkan_data,
                           ImGui_ImplVulkanH_Window* wd,
                           ImDrawData* draw_data) {
  VkSemaphore image_acquired_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
  VkSemaphore render_complete_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
  VkResult err = vkAcquireNextImageKHR(vulkan_data.vk_device, wd->Swapchain,
                                       UINT64_MAX, image_acquired_semaphore,
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
    err = vkWaitForFences(
        vulkan_data.vk_device, 1, &fd->Fence, VK_TRUE,
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
    VkPipelineStageFlags const wait_stage =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
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

constexpr void FramePresent(mazorca::vulkan_data& vulkan_data,
                            ImGui_ImplVulkanH_Window* wd) {
  if (vulkan_data.vk_swap_chain_rebuild) {
    return;
  }
  VkSemaphore render_complete_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
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
  wd->SemaphoreIndex =
      (wd->SemaphoreIndex + 1) %
      wd->SemaphoreCount;  // Now we can use the next set of semaphores
}
} // namespace mazorca
