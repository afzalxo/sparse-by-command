#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "utils.hpp"
#include "globals.hpp"


int verify_output(const std::vector<int8_t>& hw_output, const std::string& golden_file, 
                  int H, int W, int C, int stride, int flatten, bool is_sparse, uint64_t oc_mask_lo, uint64_t oc_mask_hi) {
    
    std::vector<uint8_t> file_golden_raw;
    load_bin(golden_file, file_golden_raw);
    
    // Cast to signed for comparison
    std::vector<int8_t> golden(file_golden_raw.begin(), file_golden_raw.end());

    int num_h_tiles = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int num_w_strips = (W + PP_PAR - 1) / PP_PAR;

    int eff_pp_par = PP_PAR / stride;

    int out_h = H / stride;
    int out_w = W / stride;
    
    int max_diff = 0;
    int errors = 0;
    int linear_idx = 0;

    // Iterate in Hardware Tiled Order
    for (int ht = 0; ht < num_h_tiles; ++ht) {
	int rows_remaining = H - (ht * TILE_HEIGHT);
	int active_tile_h_in = (rows_remaining < TILE_HEIGHT) ? rows_remaining : TILE_HEIGHT;
	int active_tile_h_out = active_tile_h_in / stride;
	int num_oc_tiles = (C + OC_PAR - 1) / OC_PAR;

        for (int ot = 0; ot < num_oc_tiles; ++ot) {
	    bool is_tile_active = !is_sparse || (is_sparse && ((oc_mask_lo & (1ULL << ot)) != 0));
            for (int r = 0; r < active_tile_h_out; ++r) {
                for (int w_strip = 0; w_strip < num_w_strips; ++w_strip) {
                    for (int p = 0; p < eff_pp_par; ++p) {
                        for (int o = 0; o < OC_PAR; ++o) {
                            
                            int global_h = (ht * (TILE_HEIGHT/stride)) + r;
                            int global_w = (w_strip * eff_pp_par) + p;
                            int global_oc = (ot * OC_PAR) + o;

			    bool is_padding = (global_w >= out_w) || (global_oc >= C);

			    if (!is_padding) {
                                // Planar Index for Golden
                                size_t gold_idx = (global_oc * out_h * out_w) + (global_h * out_w) + global_w;

				if (gold_idx < golden.size()) {
				    if (is_tile_active) {
                                        int8_t exp = golden[gold_idx];
                                        int8_t got = hw_output[linear_idx];
			                // std::cout << "Idx[" << linear_idx << "] H=" << global_h << " W=" << global_w << " C=" << global_oc << " Exp=" << (int)exp << " Got=" << (int)got << std::endl;
                                        int diff = std::abs((int)got - (int)exp);
                                        if (diff > 0) {
                                            // if (errors < 50) {
                                                std::cout << "Mismatch [H=" << global_h << " W=" << global_w << " C=" << global_oc 
                                                          << "] Exp: " << (int)exp << " Got: " << (int)got 
                                                          << " (Diff: " << diff << ")" << std::endl;
                                            // }
                                            errors++;
			                    if (diff > max_diff) {
			                        max_diff = diff;
			                    }
				        }
                                    } else {
				        // std::cout << "Output Match [H=" << global_h << " W=" << global_w << " C=" << global_oc 
				    //	      << "] Exp: " << (int)exp << " Got: " << (int)got << std::endl;
				    }
				}
			    }
                            linear_idx++;
                        }
			if (flatten) {
                            linear_idx += (PP_PAR-1) * OC_PAR;
                        }
                    }
                }
            }
        }
    }
    if (errors > 0) {
	std::cout << "Max Difference: " << max_diff << std::endl;
    }
    return errors;
}

int verify_gap_output(const std::vector<int8_t>& hw_output, const std::string& golden_file, int total_channels) {
    std::vector<uint8_t> file_golden_raw;
    load_bin(golden_file, file_golden_raw);
    std::vector<int8_t> golden(file_golden_raw.begin(), file_golden_raw.end());

    int errors = 0;
    int max_diff = 0;
    const int CHANNELS_PER_TILE = OC_PAR;
    const int BYTES_PER_HW_STRIP = 4 * CHANNELS_PER_TILE; //PP_PAR * CHANNELS_PER_TILE;

    int num_tiles = (total_channels + CHANNELS_PER_TILE - 1) / CHANNELS_PER_TILE;

    std::cout << "INFO: Verifying GAP Output for " << total_channels << " channels (" << num_tiles << " tiles)..." << std::endl;

    for (int t = 0; t < num_tiles; t++) {
        for (int c = 0; c < CHANNELS_PER_TILE; c++) {
            
            int global_channel = t * CHANNELS_PER_TILE + c;
            
            // Only verify if we haven't exceeded the logical channel count
            if (global_channel < total_channels) {
                
                // Hardware Index: Each tile starts every 128 bytes
                size_t hw_idx = (t * BYTES_PER_HW_STRIP) + c;
                
                // Golden Index: Flat array
                int gold_idx = global_channel;

                if (hw_idx >= hw_output.size()) {
                    std::cerr << "Error: Hardware buffer too small!" << std::endl;
                    return -1;
                }

                int8_t got = hw_output[hw_idx];
                int8_t exp = golden[gold_idx];

                int diff = std::abs((int)got - (int)exp);
		if (diff > 0) {
                    std::cout << "GAP Mismatch [Ch=" << global_channel << "] "
                              << "Exp: " << (int)exp << " Got: " << (int)got 
                              << " (Diff: " << diff << ")" << std::endl;
                    errors++;
		}
                if (diff > max_diff) max_diff = diff;
		std::cout << "Exp: " << (int)exp << " Got: " << (int)got << std::endl;
            }
        }
    }

    if (errors > 0) {
        std::cout << "GAP Verification Failed! Total Errors: " << errors 
                  << " | Max Diff: " << max_diff << std::endl;
    } else {
        std::cout << "GAP Verification Passed!" << std::endl;
    }

    return errors;
}


