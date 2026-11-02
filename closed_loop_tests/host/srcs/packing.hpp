#ifndef PACKING_HPP
#define PACKING_HPP

#include "globals.hpp"

#include <vector>
#include <cstring>
#include <cstdint>

// Pack Feature Map (Host -> Hardware Layout)
template <typename T>
void pack_feature_map(const T* input, T* output, int H, int W, int C, int H_tile, int IC_PAR_HW) {
    int num_h_tiles = H / H_tile;
    int num_c_slots = (C + IC_PAR_HW - 1) / IC_PAR_HW;
    int out_idx = 0;

    for (int ht = 0; ht < num_h_tiles; ++ht) {
        for (int cs = 0; cs < num_c_slots; ++cs) {
            for (int h_loc = 0; h_loc < H_tile; ++h_loc) {
                int h_abs = (ht * H_tile) + h_loc;
                for (int w = 0; w < W; ++w) {
                    for (int c_loc = 0; c_loc < IC_PAR_HW; ++c_loc) {
                        int c_abs = (cs * IC_PAR_HW) + c_loc;
                        if (c_abs >= C) {
                            output[out_idx++] = 0; // Padding
                        } else {
                            // Planar Input [C][H][W]
                            int in_idx = (c_abs * H * W) + (h_abs * W) + w;
                            output[out_idx++] = input[in_idx];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void pack_weights(const T* src, T* dst, int OC, int IC, int OC_PAR_HW, int IC_PAR_HW, bool is_1x1) {
    int num_ic_tiles = (IC + IC_PAR_HW - 1) / IC_PAR_HW;
    int num_oc_tiles = (OC + OC_PAR_HW - 1) / OC_PAR_HW;
    int K = is_1x1 ? 1 : 3;
    int dst_idx = 0;

    for (int t_oc = 0; t_oc < num_oc_tiles; ++t_oc) {
        for (int t_ic = 0; t_ic < num_ic_tiles; ++t_ic) {
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    for (int p_oc = 0; p_oc < OC_PAR_HW; ++p_oc) {
                        for (int p_ic = 0; p_ic < IC_PAR_HW; ++p_ic) {
                            int global_oc = (t_oc * OC_PAR_HW) + p_oc;
                            int global_ic = (t_ic * IC_PAR_HW) + p_ic;
                            
                            if (global_oc >= OC || global_ic >= IC) {
                                dst[dst_idx++] = 0;
                            } else {
                                // Source: [OC][IC][3][3] or [OC][IC][1][1]
				int src_idx;
				if (is_1x1) {
				    src_idx = (global_oc * IC) + global_ic;
				} else {
                                    src_idx = (global_oc * IC * 9) + (global_ic * 9) + (ky * 3 + kx);
				}
                                dst[dst_idx++] = src[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T1, typename T2>
void pack_biases(const T1* src, T2* dst, int OC, int OC_PAR_HW) {
    int num_oc_tiles = (OC + OC_PAR_HW - 1) / OC_PAR_HW;
    int dst_idx = 0;

    size_t src_type_size = sizeof(T1);
    size_t dst_type_size = sizeof(T2);
    size_t conversion_factor = src_type_size / dst_type_size;
    // Nominally, conversion_factor would be 4. I need to pack 1 int32_t bias into 4 int8_t slots. All 4 slots are utilized.
    for (int t_oc = 0; t_oc < num_oc_tiles; ++t_oc) {
	for (int p_oc = 0; p_oc < OC_PAR_HW; ++p_oc) {
	    int global_oc = (t_oc * OC_PAR_HW) + p_oc;
	    if (global_oc >= OC) {
		dst[dst_idx++] = 0;
	    } else {
		// dst[dst_idx++] = src[global_oc];
		T1 bias_value = src[global_oc];
		for (size_t i = 0; i < conversion_factor; ++i) {
		    dst[dst_idx++] = (bias_value >> (8 * i)) & 0xFF;
		}
	    }
	}
    }
}

std::vector<int8_t> pad_vector_for_hw(const std::vector<int8_t>& input) {
    // We assume input size is a multiple of 16 (IC_PAR)
    int num_units = input.size() / 16;
    
    // Each unit becomes a 128-byte strip (16 valid + 112 zero)
    std::vector<int8_t> padded(num_units * 64, 0);
    
    for (int i = 0; i < num_units; ++i) {
        // Copy 16 bytes of valid data
        // Dest Index: i * 128
        // Src Index: i * 16
        memcpy(&padded[i * 64], &input[i * 16], 16);
        
        // The remaining 112 bytes are already 0 initialized
    }
    
    return padded;
}

#endif // PACKING_HPP

