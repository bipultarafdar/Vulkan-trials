/*
 * Vulkan Windowed Program
 *
 * Copyright (C) 2016, 2018 Valve Corporation
 * Copyright (C) 2016, 2018 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
Vulkan C++ Windowed Project Template
Create and destroy a Vulkan surface on an SDL window.
*/

// Enable the WSI extensions
#if defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#elif defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

// Tell SDL not to mess with main()
#define SDL_MAIN_HANDLED

#include <glm/glm.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.hpp>

#include <iostream>
#include <vector>
#include <cassert>

int main()
{
    // Create an SDL window that supports Vulkan rendering.
    if(SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cout << "Could not initialize SDL." << std::endl;
        return 1;
    }
    SDL_Window* window = SDL_CreateWindow("Vulkan Window", SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_VULKAN);
    if(window == NULL) {
        std::cout << "Could not create SDL window." << std::endl;
        return 1;
    }
    
    // Get WSI extensions from SDL (we can add more if we like - we just can't remove these)
    unsigned extension_count;
    if(!SDL_Vulkan_GetInstanceExtensions(window, &extension_count, NULL)) {
        std::cout << "Could not get the number of required instance extensions from SDL." << std::endl;
        return 1;
    }
    std::vector<const char*> extensions(extension_count);
    if(!SDL_Vulkan_GetInstanceExtensions(window, &extension_count, extensions.data())) {
        std::cout << "Could not get the names of required instance extensions from SDL." << std::endl;
        return 1;
    }

    // Use validation layers if this is a debug build
    std::vector<const char*> layers;
#if defined(_DEBUG)
    layers.push_back("VK_LAYER_LUNARG_standard_validation");
#endif

    // vk::ApplicationInfo allows the programmer to specifiy some basic information about the
    // program, which can be useful for layers and tools to provide more debug information.
    vk::ApplicationInfo appInfo = vk::ApplicationInfo()
        .setPApplicationName("Vulkan C++ Windowed Program Template")
        .setApplicationVersion(1)
        .setPEngineName("LunarG SDK")
        .setEngineVersion(1)
        .setApiVersion(VK_API_VERSION_1_0);

    // vk::InstanceCreateInfo is where the programmer specifies the layers and/or extensions that
    // are needed.
    vk::InstanceCreateInfo instInfo = vk::InstanceCreateInfo()
        .setFlags(vk::InstanceCreateFlags())
        .setPApplicationInfo(&appInfo)
        .setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()))
        .setPpEnabledExtensionNames(extensions.data())
        .setEnabledLayerCount(static_cast<uint32_t>(layers.size()))
        .setPpEnabledLayerNames(layers.data());

    // Create the Vulkan instance.
    vk::Instance instance;
    try {
        instance = vk::createInstance(instInfo);
    } catch(const std::exception& e) {
        std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
        return 1;
    }

	// 2. Enumerate Devices
	std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices(NULL);
	assert(physicalDevices.data() != NULL && physicalDevices.size() > 0);

	// Select the first device
	vk::PhysicalDevice physicalDevice = physicalDevices[0];
	
	// 3. Init the device

	vk::DeviceQueueCreateInfo queueInfo = vk::DeviceQueueCreateInfo()
		.setPQueuePriorities({ 0 })
		.setQueueCount(1);

	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties(NULL);
	assert(queueFamilyProperties.data() != NULL && queueFamilyProperties.size() > 0);
	bool found = false;
	for (int i = 0; i < queueFamilyProperties.size(); i++) {
		if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
			queueInfo.queueFamilyIndex = i;
			found = true;
			break;
		}
	}
	assert(found);

	vk::DeviceCreateInfo deviceInfo = vk::DeviceCreateInfo()
		.setQueueCreateInfoCount(1)
		.setPQueueCreateInfos(&queueInfo);

	vk::Device device = physicalDevice.createDevice(deviceInfo);

	// 4. Init Command Buffer
	vk::CommandPoolCreateInfo commandPoolInfo = vk::CommandPoolCreateInfo()
		.setQueueFamilyIndex(queueInfo.queueFamilyIndex);
	vk::CommandPool commandPool = device.createCommandPool(commandPoolInfo);

	vk::CommandBufferAllocateInfo cmd = vk::CommandBufferAllocateInfo()
		.setCommandBufferCount(1)
		.setCommandPool(commandPool);

	std::vector<vk::CommandBuffer> commandBuffers = device.allocateCommandBuffers(cmd);


    // Create a Vulkan surface for rendering
    VkSurfaceKHR c_surface;
    if(!SDL_Vulkan_CreateSurface(window, static_cast<VkInstance>(instance), &c_surface)) {
        std::cout << "Could not create a Vulkan surface." << std::endl;
        return 1;
    }
    vk::SurfaceKHR surface(c_surface);

	// 5. Init Swapchain

	uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
	uint32_t presentQueueFamilyIndex = UINT32_MAX;
	for (int i = 0; i < queueFamilyProperties.size(); i++) {
		if (!(queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
			if (graphicsQueueFamilyIndex == UINT32_MAX) graphicsQueueFamilyIndex = i;
			if (physicalDevice.getSurfaceSupportKHR(i, surface) == true) {
				graphicsQueueFamilyIndex = i;
				presentQueueFamilyIndex = i;
				break;
			}
		}
	}

	if (graphicsQueueFamilyIndex == UINT32_MAX || presentQueueFamilyIndex == UINT32_MAX) {
		std::cout << "Could not find a queues for graphics and "
			"present\n";
		exit(-1);
	}

	vk::Format format;
	std::vector<vk::SurfaceFormatKHR> surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface);
	if (surfaceFormats.size() == 1 && surfaceFormats[0].format == vk::Format::eUndefined) {
		format = vk::Format::eB8G8R8A8Unorm;
	}
	else {
		assert(surfaceFormats.size() > 0);
		format = surfaceFormats[0].format;
	}
	surfaceFormats.clear();

	vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	std::vector<vk::PresentModeKHR> presentModes = physicalDevice.getSurfacePresentModesKHR(surface);
	//vk::Extent2D swapchainExtent(surfaceCapabilities.currentExtent);
	vk::PresentModeKHR swapchainPresentMode = vk::PresentModeKHR::eFifo;
	uint32_t desiredSwapChainImages = surfaceCapabilities.minImageCount;
	vk::SurfaceTransformFlagBitsKHR preTransform;
	if (surfaceCapabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) {
		preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
	}
	else {
		preTransform = surfaceCapabilities.currentTransform;
	}
	vk::CompositeAlphaFlagBitsKHR compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	for (int i = 0; i < sizeof(vk::CompositeAlphaFlagBitsKHR); i++) {
		if (surfaceCapabilities.supportedCompositeAlpha & (vk::CompositeAlphaFlagBitsKHR)i) {
			compositeAlpha = (vk::CompositeAlphaFlagBitsKHR)i;
		}
	}

	vk::SwapchainCreateInfoKHR swapchainInfo = vk::SwapchainCreateInfoKHR()
		.setSurface(surface)
		.setMinImageCount(desiredSwapChainImages)
		.setImageFormat(format)
		.setImageExtent(surfaceCapabilities.currentExtent)
		.setPreTransform(preTransform)
		.setCompositeAlpha(compositeAlpha)
		.setImageArrayLayers(1)
		.setPresentMode(swapchainPresentMode)
		.setClipped(true)
		.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear)
		.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
		.setImageSharingMode(vk::SharingMode::eExclusive);
	if (graphicsQueueFamilyIndex != presentQueueFamilyIndex) {
		uint32_t queueFamilyIndices[2] = { (uint32_t)graphicsQueueFamilyIndex, (uint32_t)presentQueueFamilyIndex };
		swapchainInfo.setImageSharingMode(vk::SharingMode::eConcurrent)
			.setQueueFamilyIndexCount(2)
			.setPQueueFamilyIndices(queueFamilyIndices);
	}

	vk::SwapchainKHR swapchain = device.createSwapchainKHR(swapchainInfo);
	std::vector<vk::Image> swapchainImages = device.getSwapchainImagesKHR(swapchain);

    // This is where most initializtion for a program should be performed

    // Poll for user input.
    bool stillRunning = true;
    while(stillRunning) {

        SDL_Event event;
        while(SDL_PollEvent(&event)) {

            switch(event.type) {

            case SDL_QUIT:
                stillRunning = false;
                break;

            default:
                // Do nothing.
                break;
            }
        }

        SDL_Delay(10);
    }

	//Clean
	device.freeCommandBuffers(commandPool, commandBuffers);
	device.destroyCommandPool(commandPool);
	device.destroy();

    // Clean up.
    instance.destroySurfaceKHR(surface);
    SDL_DestroyWindow(window);
    SDL_Quit();
    instance.destroy();

    return 0;
}
