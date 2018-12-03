#include "Camera.h"

Camera::Camera(
	glm::vec3 eye,
	glm::vec3 origin,
	glm::vec3 head
):eye(eye), origin(origin), head(head) {}

Camera::Camera()
{
}


Camera::~Camera()
{
}
