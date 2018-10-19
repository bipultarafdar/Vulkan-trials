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
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>

#define WIDTH 1280
#define HEIGHT 720
#define NUM_SAMPLES vk::SampleCountFlagBits::e1
#define XYZ1(_x_, _y_, _z_) (_x_), (_y_), (_z_), 1.f

struct Vertex {
	float posX, posY, posZ, posW;  // Position data
	float r, g, b, a;              // Color
};

struct VertexUV {
	float posX, posY, posZ, posW;  // Position data
	float u, v;                    // texture u,v
};

static const Vertex g_vb_solid_face_colors_Data[] = {
	// red face
	{ XYZ1(-1, -1, 1), XYZ1(1.f, 0.f, 0.f) },
	{ XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 0.f) },
	{ XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 0.f) },
	{ XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 0.f) },
	{ XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 0.f) },
	{ XYZ1(1, 1, 1), XYZ1(1.f, 0.f, 0.f) },
	// green face
	{ XYZ1(-1, -1, -1), XYZ1(0.f, 1.f, 0.f) },
	{ XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 0.f) },
	{ XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f) },
	{ XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f) },
	{ XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 0.f) },
	{ XYZ1(1, 1, -1), XYZ1(0.f, 1.f, 0.f) },
	// blue face
	{ XYZ1(-1, 1, 1), XYZ1(0.f, 0.f, 1.f) },
	{ XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f) },
	{ XYZ1(-1, 1, -1), XYZ1(0.f, 0.f, 1.f) },
	{ XYZ1(-1, 1, -1), XYZ1(0.f, 0.f, 1.f) },
	{ XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f) },
	{ XYZ1(-1, -1, -1), XYZ1(0.f, 0.f, 1.f) },
	// yellow face
	{ XYZ1(1, 1, 1), XYZ1(1.f, 1.f, 0.f) },
	{ XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f) },
	{ XYZ1(1, -1, 1), XYZ1(1.f, 1.f, 0.f) },
	{ XYZ1(1, -1, 1), XYZ1(1.f, 1.f, 0.f) },
	{ XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f) },
	{ XYZ1(1, -1, -1), XYZ1(1.f, 1.f, 0.f) },
	// magenta face
	{ XYZ1(1, 1, 1), XYZ1(1.f, 0.f, 1.f) },
	{ XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 1.f) },
	{ XYZ1(1, 1, -1), XYZ1(1.f, 0.f, 1.f) },
	{ XYZ1(1, 1, -1), XYZ1(1.f, 0.f, 1.f) },
	{ XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 1.f) },
	{ XYZ1(-1, 1, -1), XYZ1(1.f, 0.f, 1.f) },
	// cyan face
	{ XYZ1(1, -1, 1), XYZ1(0.f, 1.f, 1.f) },
	{ XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 1.f) },
	{ XYZ1(-1, -1, 1), XYZ1(0.f, 1.f, 1.f) },
	{ XYZ1(-1, -1, 1), XYZ1(0.f, 1.f, 1.f) },
	{ XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 1.f) },
	{ XYZ1(-1, -1, -1), XYZ1(0.f, 1.f, 1.f) },
};

static std::vector<uint32_t> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<uint32_t> buffer(fileSize);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
	file.close();

	return buffer;
}

int main()
{
	// Create an SDL window that supports Vulkan rendering.
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		std::cout << "Could not initialize SDL." << std::endl;
		return 1;
	}
	SDL_Window* window = SDL_CreateWindow("Vulkan Window", SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_VULKAN);
	if (window == NULL) {
		std::cout << "Could not create SDL window." << std::endl;
		return 1;
	}

	// Get WSI extensions from SDL (we can add more if we like - we just can't remove these)
	unsigned extension_count;
	if (!SDL_Vulkan_GetInstanceExtensions(window, &extension_count, NULL)) {
		std::cout << "Could not get the number of required instance extensions from SDL." << std::endl;
		return 1;
	}
	std::vector<const char*> extensions(extension_count);
	if (!SDL_Vulkan_GetInstanceExtensions(window, &extension_count, extensions.data())) {
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
	}
	catch (const std::exception& e) {
		std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
		return 1;
	}

	// 2. Enumerate Devices
	std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
	assert(physicalDevices.data() != NULL && physicalDevices.size() > 0);

	// Select the first device
	vk::PhysicalDevice physicalDevice = physicalDevices[0];

	// 3. Init the device
	float queuePrios[1] = { 0.0 };
	vk::DeviceQueueCreateInfo queueInfo = vk::DeviceQueueCreateInfo()
		.setPQueuePriorities(queuePrios)
		.setQueueCount(1);

	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
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

	// Create a Vulkan surface for rendering
	VkSurfaceKHR c_surface;
	if (!SDL_Vulkan_CreateSurface(window, static_cast<VkInstance>(instance), &c_surface)) {
		std::cout << "Could not create a Vulkan surface." << std::endl;
		return 1;
	}
	vk::SurfaceKHR surface(c_surface);

	// 5. Init Swapchain

	uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
	uint32_t presentQueueFamilyIndex = UINT32_MAX;
	for (int i = 0; i < queueFamilyProperties.size(); i++) {
		if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
			if (graphicsQueueFamilyIndex == UINT32_MAX) graphicsQueueFamilyIndex = i;
			if (physicalDevice.getSurfaceSupportKHR(i, surface)) {
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

	std::vector<vk::ExtensionProperties> deviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
	std::vector<char*> dvcExtNames;
	for (auto d : deviceExtensions) {
		dvcExtNames.push_back(d.extensionName);
	}
	dvcExtNames.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

	vk::DeviceCreateInfo deviceInfo = vk::DeviceCreateInfo()
		.setQueueCreateInfoCount(1)
		.setPQueueCreateInfos(&queueInfo)
		.setEnabledExtensionCount(dvcExtNames.size())
		.setPpEnabledExtensionNames(dvcExtNames.data());

	vk::Device device = physicalDevice.createDevice(deviceInfo);

	// 4. Init Command Buffer
	vk::CommandPoolCreateInfo commandPoolInfo = vk::CommandPoolCreateInfo()
		.setQueueFamilyIndex(queueInfo.queueFamilyIndex);
	vk::CommandPool commandPool = device.createCommandPool(commandPoolInfo);

	vk::CommandBufferAllocateInfo cmd = vk::CommandBufferAllocateInfo()
		.setCommandBufferCount(1)
		.setCommandPool(commandPool);

	vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(cmd)[0];

	vk::CommandBufferBeginInfo cmdBeginInfo = vk::CommandBufferBeginInfo();
	commandBuffer.begin(cmdBeginInfo);

	vk::Queue graphicsQueue = device.getQueue(graphicsQueueFamilyIndex, 0);
	vk::Queue presentQueue = graphicsQueue;


	//5. Init Swap Chain
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
			break;
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
	std::vector<vk::ImageView> bufferImageViews(swapchainImages.size());

	vk::ImageSubresourceRange subresourceRange = vk::ImageSubresourceRange()
		.setAspectMask(vk::ImageAspectFlagBits::eColor)
		.setLevelCount(1)
		.setLayerCount(1);

	for (int i = 0; i < swapchainImages.size(); i++) {
		vk::ImageViewCreateInfo colorImageView = vk::ImageViewCreateInfo()
			.setFlags((vk::ImageViewCreateFlagBits)0)
			.setImage(swapchainImages[i])
			.setViewType(vk::ImageViewType::e2D)
			.setFormat(format)
			.setComponents(vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA))
			.setSubresourceRange(subresourceRange);

		bufferImageViews[i] = device.createImageView(colorImageView);
	}

	uint32_t currentBuffer = 0;

	// 6. Init Depth Buffer
	vk::Format depthFormat = vk::Format::eD16Unorm;
	vk::ImageCreateInfo imageInfo = vk::ImageCreateInfo()
		.setImageType(vk::ImageType::e2D)
		.setFormat(depthFormat)
		.setExtent(vk::Extent3D(WIDTH, HEIGHT, 1))
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(NUM_SAMPLES)
		.setInitialLayout(vk::ImageLayout::eUndefined)
		.setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
		.setSharingMode(vk::SharingMode::eExclusive);
	//.setFlags(vk::ImageCreateFlags());
	vk::FormatProperties formatProps = physicalDevice.getFormatProperties(depthFormat);
	if (formatProps.linearTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
		imageInfo.setTiling(vk::ImageTiling::eLinear);
	}
	else if (formatProps.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
		imageInfo.setTiling(vk::ImageTiling::eOptimal);
	}
	else {
		std::cout << "VK_FORMAT_D16_UNORM Unsupported.\n";
		exit(-1);
	}

	vk::ImageSubresourceRange depthSubresourceRange = vk::ImageSubresourceRange()
		.setAspectMask(vk::ImageAspectFlagBits::eDepth)
		.setLevelCount(1)
		.setLayerCount(1);
	vk::ImageViewCreateInfo viewInfo = vk::ImageViewCreateInfo()
		.setImage(nullptr)
		.setFormat(depthFormat)
		.setComponents(vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA))
		.setViewType(vk::ImageViewType::e2D)
		.setSubresourceRange(depthSubresourceRange);

	vk::Image depthImage = device.createImage(imageInfo);
	vk::MemoryRequirements memReqs = device.getImageMemoryRequirements(depthImage);
	vk::MemoryAllocateInfo memAlloc1 = vk::MemoryAllocateInfo()
		.setAllocationSize(memReqs.size);
	vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();

	uint32_t typeBits = memReqs.memoryTypeBits;
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
		if ((typeBits & 1) == 1) {
			if (memoryProperties.memoryTypes[i].propertyFlags == vk::MemoryPropertyFlagBits::eDeviceLocal) {
				memAlloc1.setMemoryTypeIndex(i);
				break;
			}
		}
		typeBits >>= 1;
	}

	vk::DeviceMemory depthMem = device.allocateMemory(memAlloc1);
	device.bindImageMemory(depthImage, depthMem, 0);

	viewInfo.setImage(depthImage);
	vk::ImageView depthImageView = device.createImageView(viewInfo);

	// 7. Create Uniform Buffer
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
	glm::mat4 view = glm::lookAt(glm::vec3(-5, 3, -10),  // Camera is at (-5,3,-10), in World Space
		glm::vec3(0, 0, 0),     // and looks at the origin
		glm::vec3(0, -1, 0)     // Head is up (set to 0,-1,0 to look upside-down)
	);
	glm::mat4 model = glm::mat4(1.0f);

	// Vulkan clip space has inverted Y and half Z.
	// clang-format off
	glm::mat4 clip = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, -1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.5f, 0.0f,
		0.0f, 0.0f, 0.5f, 1.0f);
	// clang-format on
	glm::mat4 mvp = clip * projection * view * model;

	vk::BufferCreateInfo bufferCInfo = vk::BufferCreateInfo()
		.setUsage(vk::BufferUsageFlagBits::eUniformBuffer)
		.setSize(sizeof(mvp))
		.setSharingMode(vk::SharingMode::eExclusive);

	vk::Buffer buffer = device.createBuffer(bufferCInfo);

	memReqs = device.getBufferMemoryRequirements(buffer);
	vk::MemoryAllocateInfo memAlloc2 = vk::MemoryAllocateInfo()
		.setAllocationSize(memReqs.size);

	typeBits = memReqs.memoryTypeBits;
	for (int i = 0; i < memoryProperties.memoryTypeCount; i++) {
		if (typeBits & 1 == 1) {
			if (memoryProperties.memoryTypes[i].propertyFlags == (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)) {
				memAlloc2.setMemoryTypeIndex(i);
				break;
			}
		}
		typeBits >>= 1;
	}

	vk::DeviceMemory bufferMem = device.allocateMemory(memAlloc2);

	void * memPtr = device.mapMemory(bufferMem, 0, memReqs.size);
	memcpy(memPtr, &mvp, sizeof(mvp));
	device.unmapMemory(bufferMem);

	device.bindBufferMemory(buffer, bufferMem, 0);

	vk::DescriptorBufferInfo bufferInfo = vk::DescriptorBufferInfo()
		.setBuffer(buffer)
		.setOffset(0)
		.setRange(sizeof(mvp));

	// 8. Init Pipeline Layout
	vk::DescriptorSetLayoutBinding layoutBinding = vk::DescriptorSetLayoutBinding()
		.setDescriptorCount(1)
		.setStageFlags(vk::ShaderStageFlagBits::eVertex);
	vk::DescriptorSetLayoutCreateInfo descriptorLayoutInfo = vk::DescriptorSetLayoutCreateInfo()
		.setBindingCount(1)
		.setPBindings(&layoutBinding);

	vk::DescriptorSetLayout descriptorLayout = device.createDescriptorSetLayout(descriptorLayoutInfo);

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo = vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 1, &descriptorLayout);

	vk::PipelineLayout pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

	// 10. Init Render Pass
	vk::SemaphoreCreateInfo imageSemaphoreInfo = vk::SemaphoreCreateInfo();
	vk::Semaphore imageAcquiredSemaphore = device.createSemaphore(imageSemaphoreInfo);
	device.acquireNextImageKHR(swapchain, UINT64_MAX, imageAcquiredSemaphore, nullptr, &currentBuffer);

	vk::AttachmentDescription attachments[2];
	attachments[0] = vk::AttachmentDescription()
		.setFormat(format)
		.setSamples(vk::SampleCountFlagBits::e1)
		.setLoadOp(vk::AttachmentLoadOp::eClear)
		.setStoreOp(vk::AttachmentStoreOp::eStore)
		.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
		.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
		.setInitialLayout(vk::ImageLayout::eUndefined)
		.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

	attachments[1] = vk::AttachmentDescription()
		.setFormat(depthFormat)
		.setSamples(vk::SampleCountFlagBits::e1)
		.setLoadOp(vk::AttachmentLoadOp::eClear)
		.setStoreOp(vk::AttachmentStoreOp::eStore)
		.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
		.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
		.setInitialLayout(vk::ImageLayout::eUndefined)
		.setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

	vk::SubpassDescription subpass = vk::SubpassDescription()
		.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
		.setColorAttachmentCount(1)
		.setPColorAttachments(new vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal))
		.setPDepthStencilAttachment(new vk::AttachmentReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal));

	vk::RenderPassCreateInfo rpInfo = vk::RenderPassCreateInfo()
		.setAttachmentCount(2)
		.setPAttachments(attachments)
		.setSubpassCount(1)
		.setPSubpasses(&subpass);

	vk::RenderPass renderPass = device.createRenderPass(rpInfo);

	// 11. Init Shaders


	auto vertShaderCode = readFile("shaders/vert.spv");
	auto fragShaderCode = readFile("shaders/frag.spv");

	vk::PipelineShaderStageCreateInfo shaderStages[2];
	shaderStages[0] = vk::PipelineShaderStageCreateInfo()
		.setStage(vk::ShaderStageFlagBits::eVertex)
		.setPName("main");
	shaderStages[1] = vk::PipelineShaderStageCreateInfo()
		.setStage(vk::ShaderStageFlagBits::eFragment)
		.setPName("main");

	vk::ShaderModuleCreateInfo moduleInfo = vk::ShaderModuleCreateInfo()
		.setCodeSize(vertShaderCode.size() * sizeof(unsigned int))
		.setPCode(vertShaderCode.data());
	shaderStages[0].setModule(device.createShaderModule(moduleInfo));

	moduleInfo.setCodeSize(fragShaderCode.size() * sizeof(unsigned int))
		.setPCode(fragShaderCode.data());
	shaderStages[1].setModule(device.createShaderModule(moduleInfo));

	//12. Init Frame Buffers
	vk::ImageView fb_attachments[2];
	fb_attachments[1] = depthImageView;

	vk::FramebufferCreateInfo framebufferInfo = vk::FramebufferCreateInfo()
		.setRenderPass(renderPass)
		.setAttachmentCount(2)
		.setPAttachments(fb_attachments)
		.setWidth(WIDTH)
		.setHeight(HEIGHT)
		.setLayers(1);

	std::vector<vk::Framebuffer> framebuffers(swapchainImages.size());
	for (uint32_t i = 0; i < swapchainImages.size(); i++) {
		fb_attachments[0] = bufferImageViews[i];
		framebuffers[i] = device.createFramebuffer(framebufferInfo);
	}

	// 13. Init Vertex Buffer
	vk::BufferCreateInfo vBufferInfo = vk::BufferCreateInfo()
		.setUsage(vk::BufferUsageFlagBits::eVertexBuffer)
		.setSize(sizeof(g_vb_solid_face_colors_Data))
		.setSharingMode(vk::SharingMode::eExclusive);

	vk::Buffer vertexBuffer = device.createBuffer(vBufferInfo);
	memReqs = device.getBufferMemoryRequirements(vertexBuffer);
	vk::MemoryAllocateInfo memAlloc3 = vk::MemoryAllocateInfo()
		.setAllocationSize(memReqs.size);

	typeBits = memReqs.memoryTypeBits;
	for (int i = 0; i < memoryProperties.memoryTypeCount; i++) {
		if (typeBits & 1 == 1) {
			if (memoryProperties.memoryTypes[i].propertyFlags == (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)) {
				memAlloc3.setMemoryTypeIndex(i);
				break;
			}
		}
		typeBits >>= 1;
	}

	bufferMem = device.allocateMemory(memAlloc3);

	memPtr = device.mapMemory(bufferMem, 0, memReqs.size);
	memcpy(memPtr, &g_vb_solid_face_colors_Data, sizeof(g_vb_solid_face_colors_Data));
	device.unmapMemory(bufferMem);

	device.bindBufferMemory(vertexBuffer, bufferMem, 0);

	vk::VertexInputBindingDescription viBinding = vk::VertexInputBindingDescription()
		.setBinding(0)
		.setInputRate(vk::VertexInputRate::eVertex)
		.setStride(sizeof(g_vb_solid_face_colors_Data[0]));

	vk::VertexInputAttributeDescription viAttributes[2];
	viAttributes[0] = vk::VertexInputAttributeDescription()
		.setFormat(vk::Format::eR32G32B32A32Sfloat);
	viAttributes[0] = vk::VertexInputAttributeDescription()
		.setFormat(vk::Format::eR32G32B32A32Sfloat)
		.setLocation(1)
		.setOffset(16);


	// 9. Init Descriptor Set
	vk::DescriptorPoolSize typeCount[1] = { vk::DescriptorPoolSize()
		.setType(vk::DescriptorType::eUniformBuffer)
		.setDescriptorCount(1) };
	vk::DescriptorPoolCreateInfo desciptorPoolInfo = vk::DescriptorPoolCreateInfo()
		.setMaxSets(1)
		.setPoolSizeCount(1)
		.setPPoolSizes(typeCount);

	vk::DescriptorPool descriptorPool = device.createDescriptorPool(desciptorPoolInfo);

	vk::DescriptorSetAllocateInfo allocInfo = vk::DescriptorSetAllocateInfo()
		.setDescriptorPool(descriptorPool)
		.setDescriptorSetCount(1)
		.setPSetLayouts(&descriptorLayout);

	std::vector<vk::DescriptorSet> descriptorSets = device.allocateDescriptorSets(allocInfo);

	std::vector<vk::WriteDescriptorSet> writes(1);
	writes[0] = vk::WriteDescriptorSet()
		.setDstSet(descriptorSets[0])
		.setDescriptorCount(1)
		.setDescriptorType(vk::DescriptorType::eUniformBuffer)
		.setPBufferInfo(&bufferInfo);

	device.updateDescriptorSets(writes, NULL);

	// Init Pipeline Cache
	vk::PipelineCacheCreateInfo pipelineCacheInfo = vk::PipelineCacheCreateInfo();
	vk::PipelineCache pipelineCache = device.createPipelineCache(pipelineCacheInfo);

	// 14. Init Pipeline
	vk::DynamicState dynamicStateEnables[VK_DYNAMIC_STATE_RANGE_SIZE];
	memset(dynamicStateEnables, 0, sizeof dynamicStateEnables);
	vk::PipelineDynamicStateCreateInfo dynamicState = vk::PipelineDynamicStateCreateInfo()
		.setPDynamicStates(dynamicStateEnables);

	vk::PipelineVertexInputStateCreateInfo vi = vk::PipelineVertexInputStateCreateInfo()
		.setVertexBindingDescriptionCount(1)
		.setPVertexBindingDescriptions(&viBinding)
		.setVertexAttributeDescriptionCount(2)
		.setPVertexAttributeDescriptions(viAttributes);

	vk::PipelineInputAssemblyStateCreateInfo ia = vk::PipelineInputAssemblyStateCreateInfo()
		.setTopology(vk::PrimitiveTopology::eTriangleList);

	vk::PipelineRasterizationStateCreateInfo rs = vk::PipelineRasterizationStateCreateInfo()
		.setCullMode(vk::CullModeFlagBits::eBack)
		.setFrontFace(vk::FrontFace::eClockwise)
		.setLineWidth(1);

	vk::PipelineColorBlendAttachmentState attState = vk::PipelineColorBlendAttachmentState();
	vk::PipelineColorBlendStateCreateInfo cb = vk::PipelineColorBlendStateCreateInfo()
		.setAttachmentCount(1)
		.setPAttachments(&attState)
		.setLogicOp(vk::LogicOp::eNoOp)
		.setBlendConstants({ {1, 1, 1, 1} });

	vk::PipelineViewportStateCreateInfo vp = vk::PipelineViewportStateCreateInfo()
		.setViewportCount(1)
		.setScissorCount(1);
	dynamicStateEnables[dynamicState.dynamicStateCount++] = vk::DynamicState::eViewport;
	dynamicStateEnables[dynamicState.dynamicStateCount++] = vk::DynamicState::eScissor;
	
	vk::StencilOpState back = vk::StencilOpState()
		.setCompareOp(vk::CompareOp::eAlways);
	vk::PipelineDepthStencilStateCreateInfo ds = vk::PipelineDepthStencilStateCreateInfo()
		.setDepthTestEnable(true)
		.setDepthWriteEnable(true)
		.setDepthCompareOp(vk::CompareOp::eLessOrEqual)
		.setBack(back)
		.setFront(back);

	vk::PipelineMultisampleStateCreateInfo ms = vk::PipelineMultisampleStateCreateInfo();
	
	vk::GraphicsPipelineCreateInfo pipelineInfo = vk::GraphicsPipelineCreateInfo()
		.setLayout(pipelineLayout)
		.setPVertexInputState(&vi)
		.setPInputAssemblyState(&ia)
		.setPRasterizationState(&rs)
		.setPColorBlendState(&cb)
		.setPMultisampleState(&ms)
		.setPDynamicState(&dynamicState)
		.setPViewportState(&vp)
		.setPDepthStencilState(&ds)
		.setPStages(shaderStages)
		.setStageCount(2)
		.setRenderPass(renderPass)
		.setSubpass(0);

	vk::Pipeline pipeline = device.createGraphicsPipeline(pipelineCache, pipelineInfo);

	vk::ClearValue clearValues[2];
	std::array<float, 4> colorVal = { 0.2f, 0.2f, 0.2f, 0.2f };
	clearValues[0] = vk::ClearValue()
		.setColor(vk::ClearColorValue(colorVal));
	clearValues[1] = vk::ClearValue()
		.setDepthStencil(vk::ClearDepthStencilValue(1.0f, 0));

	//currentBuffer = device.acquireNextImageKHR(swapchain, UINT64_MAX, imageAcquiredSemaphore, {}).value;
	device.acquireNextImageKHR(swapchain, UINT64_MAX, imageAcquiredSemaphore, nullptr, &currentBuffer);

	vk::RenderPassBeginInfo rp_begin = vk::RenderPassBeginInfo()
		.setRenderPass(renderPass)
		.setFramebuffer(framebuffers[currentBuffer])
		.setRenderArea(vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D(WIDTH, HEIGHT)))
		.setClearValueCount(2)
		.setPClearValues(clearValues);

	commandBuffer.beginRenderPass(rp_begin, vk::SubpassContents::eInline);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, descriptorSets.data(), 0, NULL);

	const vk::DeviceSize offsets[1] = { 0 };
	commandBuffer.bindVertexBuffers(0, 1, &vertexBuffer, offsets);

	//Init Viewports
	vk::Viewport viewport = vk::Viewport(0, 0, WIDTH, HEIGHT, 0, 1);
	commandBuffer.setViewport(0, viewport);

	//Init Scissors
	vk::Rect2D scissor = vk::Rect2D(vk::Offset2D(), vk::Extent2D(WIDTH, HEIGHT));

	//L:149

	commandBuffer.draw(12 * 3, 1, 0, 0);
	commandBuffer.endRenderPass();
	commandBuffer.end();

	vk::FenceCreateInfo fenceInfo = vk::FenceCreateInfo();
	vk::Fence drawFence = device.createFence(fenceInfo);

	vk::PipelineStageFlags pipeStageFlags = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	std::vector<vk::SubmitInfo> submitInfo(1);
	submitInfo[0] = vk::SubmitInfo()
		.setPWaitDstStageMask(&pipeStageFlags)
		.setCommandBufferCount(1)
		.setPCommandBuffers(&commandBuffer);

	vk::PresentInfoKHR present = vk::PresentInfoKHR()
		.setSwapchainCount(1)
		.setPSwapchains(&swapchain)
		.setPImageIndices(&currentBuffer);

	vk::Result res = vk::Result::eSuccess;
	do {
		res = device.waitForFences(1, &drawFence, true, 10000000);
	} while (res == vk::Result::eTimeout);

	presentQueue.presentKHR(present);


	// This is where most initializtion for a program should be performed

	// Poll for user input.
	bool stillRunning = true;
	while (stillRunning) {

		SDL_Event event;
		while (SDL_PollEvent(&event)) {

			switch (event.type) {

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
	device.destroyPipeline(pipeline);
	device.destroyBuffer(vertexBuffer);
	device.destroyShaderModule(shaderStages[0].module);
	device.destroyShaderModule(shaderStages[1].module);
	device.destroyRenderPass(renderPass);
	device.destroySemaphore(imageAcquiredSemaphore);
	device.destroyDescriptorPool(descriptorPool);
	device.destroyDescriptorSetLayout(descriptorLayout);
	device.destroyBuffer(buffer);
	device.freeMemory(bufferMem);
	device.destroyImageView(depthImageView);
	device.destroyImage(depthImage);
	device.freeMemory(depthMem);
	for (int i = 0; i < swapchainImages.size(); i++) {
		device.destroyImageView(bufferImageViews[i]);
	}
	device.destroySwapchainKHR(swapchain);
	device.freeCommandBuffers(commandPool, commandBuffer);
	device.destroyCommandPool(commandPool);
	device.destroy();

	// Clean up.
	instance.destroySurfaceKHR(surface);
	SDL_DestroyWindow(window);
	SDL_Quit();
	instance.destroy();

	return 0;
}
