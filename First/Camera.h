#pragma once

#include <vector>
#include <glm/glm.hpp>

class Camera
{
public:
	Camera(
		glm::vec3 eye,
		glm::vec3 origin,
		glm::vec3 head
	);
	Camera();
	~Camera();
private:
	glm::vec3 eye;
	glm::vec3 origin;
	glm::vec3 head;

	glm::mat4 projection;
	glm::mat4 view;
};

