#include "utils.hpp"
#include <iostream>
#include <cstdint>
#include <cmath>
#include <vector>
#include <fstream>
#include <limits>

void wait_for_enter(const std::string &msg) {
    std::cout << msg << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

std::vector<LayerConfig> load_model_specs(std::string path) {
    std::vector<LayerConfig> layers;
    std::ifstream fs(path);
    LayerConfig l;
    int relu_int;

    while (fs >> l.name >> l.in_w >> l.in_h >> l.in_c >> l.out_c >> l.stride 
              >> l.quant_shift >> relu_int >> l.opcode 
	      >> l.is_sparse >> l.oc_tile_mask_lo >> l.oc_tile_mask_hi
	      >> l.weight_file >> l.bias_file >> l.golden_file) {
        l.relu = (relu_int != 0);
        layers.push_back(l);
    }
    std::cout << "Loaded " << layers.size() << " layers from model metadata file: " << path << std::endl;
    return layers;
}

