#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>

struct Vertex {
public:
	Vertex(glm::vec4 pos, glm::vec4 color) :
		pos(pos), color(color) {}
private:
	// v
	glm::vec4 pos;
	glm::vec4 color;
	// vt
	glm::vec2 uv;
	// vn
	glm::vec3 normals;
	//vp
	glm::vec3 parametricSpaceVertices;
};

class Object
{
public:
	Object();
	~Object();
	void read_object(std::string filename);
	void load_sample();
private:
	std::vector<Vertex> vertices;
	glm::mat4 model;
};

