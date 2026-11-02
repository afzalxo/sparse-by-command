#ifndef UTILS_HPP
#define UTILS_HPP

#include "globals.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <limits>
#include <string>

template <typename T>
void load_bin(const std::string& filename, std::vector<T>& buffer) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        exit(1);
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    buffer.resize(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
}

void wait_for_enter(const std::string &msg);

std::vector<LayerConfig> load_model_specs(std::string path);

#endif
