#pragma once

#define VK_USE_PLATFORM_WIN32_KHR

#include <SDL2/SDL.h>
#include <vulkan/vulkan.hpp>

#include "Camera.h"
#include "Object.h"

typedef struct {
	vk::Image image;
	vk::CommandBuffer cmd;
	vk::CommandBuffer graphics_to_present_cmd;
	vk::ImageView view;
	vk::Buffer uniform_buffer;
	vk::DeviceMemory uniform_memory;
	vk::Framebuffer framebuffer;
	vk::DescriptorSet descriptor_set;
} SwapchainImageResources;

// defines all objects, the camera, and the vulkan base
class Structure
{
public:
	Structure();
	~Structure();
private:
	std::vector<Camera> cameras;
	std::vector<Object> models;
	SDL_Window* window;
	vk::SurfaceKHR surface;
	vk::Instance instance;
	vk::PhysicalDevice physicalDevice; 
	uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
	uint32_t presentQueueFamilyIndex = UINT32_MAX;
	std::vector<vk::ExtensionProperties> deviceExtensions;
	vk::Device device;
	vk::CommandPool commandPool;
	vk::CommandBuffer commandBuffer;
	std::vector<vk::Queue> graphicsQueues;
	std::vector<vk::Queue> presentQueues;
	vk::SurfaceCapabilitiesKHR surfaceCapabilities;
	std::vector<vk::PresentModeKHR> presentModes;
	vk::SwapchainKHR sc;
	std::vector<SwapchainImageResources> swapchainImageResources;
	struct {
		vk::Format format;
		vk::Image image;
		vk::MemoryAllocateInfo mem_alloc;
		vk::DeviceMemory mem;
		vk::ImageView view;
	} depth;
	vk::PhysicalDeviceMemoryProperties memoryProperties;
	struct {
		vk::DeviceMemory memory;
		vk::Buffer buffer;
		vk::MemoryAllocateInfo mem_info;
		vk::DescriptorBufferInfo bufInfo;
	} uniformBuffer;
};

