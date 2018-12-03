#include "Object.h"
#include <fstream>

#define vert(x, y, z, r, g, b) vertices.push_back \
		(Vertex(glm::vec4(x, y, z, 1.0f), glm::vec4(r, g, b, 1.0f)));

void Object::read_object(std::string filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
}

void Object::load_sample() {
	// red face
	vert(-1, -1, 1, 1.f, 0.f, 0.f);
	vert(-1, 1, 1, 1.f, 0.f, 0.f);
	vert(1, -1, 1, 1.f, 0.f, 0.f);
	vert(1, -1, 1, 1.f, 0.f, 0.f);
	vert(-1, 1, 1, 1.f, 0.f, 0.f);
	vert(1, 1, 1, 1.f, 0.f, 0.f);
	// green face
	vert(-1, -1, -1, 0.f, 1.f, 0.f);
	vert(1, -1, -1, 0.f, 1.f, 0.f);
	vert(-1, 1, -1, 0.f, 1.f, 0.f);
	vert(-1, 1, -1, 0.f, 1.f, 0.f);
	vert(1, -1, -1, 0.f, 1.f, 0.f);
	vert(1, 1, -1, 0.f, 1.f, 0.f);
	// blue face
	vert(-1, 1, 1, 0.f, 0.f, 1.f);
	vert(-1, -1, 1, 0.f, 0.f, 1.f);
	vert(-1, 1, -1, 0.f, 0.f, 1.f);
	vert(-1, 1, -1, 0.f, 0.f, 1.f);
	vert(-1, -1, 1, 0.f, 0.f, 1.f);
	vert(-1, -1, -1, 0.f, 0.f, 1.f);
	// yellow face
	vert(1, 1, 1, 1.f, 1.f, 0.f);
	vert(1, 1, -1, 1.f, 1.f, 0.f);
	vert(1, -1, 1, 1.f, 1.f, 0.f);
	vert(1, -1, 1, 1.f, 1.f, 0.f);
	vert(1, 1, -1, 1.f, 1.f, 0.f);
	vert(1, -1, -1, 1.f, 1.f, 0.f);
	// magenta face
	vert(1, 1, 1, 1.f, 0.f, 1.f);
	vert(-1, 1, 1, 1.f, 0.f, 1.f);
	vert(1, 1, -1, 1.f, 0.f, 1.f);
	vert(1, 1, -1, 1.f, 0.f, 1.f);
	vert(-1, 1, 1, 1.f, 0.f, 1.f);
	vert(-1, 1, -1, 1.f, 0.f, 1.f);
	// cyan face
	vert(1, -1, 1, 0.f, 1.f, 1.f);
	vert(1, -1, -1, 0.f, 1.f, 1.f);
	vert(-1, -1, 1, 0.f, 1.f, 1.f);
	vert(-1, -1, 1, 0.f, 1.f, 1.f);
	vert(1, -1, -1, 0.f, 1.f, 1.f);
	vert(-1, -1, -1, 0.f, 1.f, 1.f);
}

Object::Object()
{
}


Object::~Object()
{
}
