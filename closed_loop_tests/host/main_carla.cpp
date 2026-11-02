#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <string>
#include <fstream>
#include <iomanip>
#include <cstring>

#include "srcs/globals.hpp"
#include "srcs/packing.hpp"
#include "srcs/verify_output.hpp"
#include "srcs/utils.hpp"

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl2.hpp>
#include "xcl2.hpp"


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    size_t vision_features = 1024;
    std::vector<LayerConfig> layers = load_model_specs("model_spec_temp.txt");

    // A. Input Feature Map
    std::vector<uint8_t> raw_input;
    load_bin<uint8_t>("fpga_data_carla/input_vision.bin", raw_input);
    std::vector<uint8_t> extra_features;
    load_bin<uint8_t>("fpga_data_carla/extra_features.bin", extra_features);
    
    std::vector<int8_t> extra_features_signed(extra_features.begin(), extra_features.end());
    std::vector<int8_t> packed_extra_features = pad_vector_for_hw(extra_features_signed);

    size_t L0_INP_FMAP_SIZE = layers[0].in_w * layers[0].in_h * 
                                  ((layers[0].in_c + IC_PAR - 1) / IC_PAR) * IC_PAR;
    size_t heap_size = INSTR_SECTION_SIZE + L0_INP_FMAP_SIZE + 
		       packed_extra_features.size();
    std::vector<uint8_t, aligned_allocator<uint8_t>> host_heap(heap_size);

    // Pack Input into Heap
    pack_feature_map<uint8_t>(raw_input.data(), host_heap.data() + INSTR_SECTION_SIZE, 
                              layers[0].in_h, layers[0].in_w, layers[0].in_c, 
                              TILE_HEIGHT, IC_PAR);
    for (size_t i = 0; i < packed_extra_features.size(); ++i) {
	host_heap[INSTR_SECTION_SIZE + L0_INP_FMAP_SIZE + i] = packed_extra_features[i];
    }
    // B. Weights (Stack all layers)
    size_t total_weight_size = 0;
    size_t total_bias_size = 0;
    for (const auto& layer : layers) {
	if (layer.opcode == OP_MEMCPY || layer.opcode == OP_GAP) {
	    continue;
	}
	bool is_1x1 = layer.opcode == OP_GEMM;
	uint32_t kernel_size = is_1x1 ? 1 : 9;
        // Align to IC_PAR/OC_PAR
        int ic_pad = (layer.in_c + IC_PAR - 1) / IC_PAR * IC_PAR;
        int oc_pad = (layer.out_c + OC_PAR - 1) / OC_PAR * OC_PAR;
        total_weight_size += oc_pad * ic_pad * kernel_size;
	total_bias_size += oc_pad * sizeof(BIAS_T);
    }
    std::vector<uint8_t, aligned_allocator<uint8_t>> host_weights(total_weight_size + total_bias_size);
    
    // Pack Weights
    size_t current_weight_offset = 0;
    size_t current_bias_offset = total_weight_size;
    std::vector<size_t> layer_weight_offsets;
    std::vector<size_t> layer_bias_offsets;

    for (const auto& layer : layers) {
	if (layer.opcode == OP_MEMCPY) { continue; }
	if (layer.opcode == OP_GAP) { continue; }
        std::vector<uint8_t> w_raw;
        load_bin<uint8_t>(layer.weight_file, w_raw);
	std::vector<uint32_t> b_raw;
	load_bin<uint32_t>(layer.bias_file, b_raw);
        
        layer_weight_offsets.push_back(current_weight_offset);
	layer_bias_offsets.push_back(current_bias_offset);

	bool is_1x1 = (layer.stride == 1 && layer.in_h == 1);
	uint32_t kernel_size = is_1x1 ? 1 : 9;
        
        pack_weights<uint8_t>(w_raw.data(), host_weights.data() + current_weight_offset, 
                              layer.out_c, layer.in_c, OC_PAR, IC_PAR, is_1x1);
	pack_biases<uint32_t, uint8_t>(b_raw.data(), host_weights.data() + current_bias_offset, 
				   layer.out_c, OC_PAR);
        
        int ic_pad = (layer.in_c + IC_PAR - 1) / IC_PAR * IC_PAR;
        int oc_pad = (layer.out_c + OC_PAR - 1) / OC_PAR * OC_PAR;
        current_weight_offset += oc_pad * ic_pad * kernel_size;
	current_bias_offset += oc_pad * sizeof(BIAS_T);
    }

    // C. Output Buffers (Max size needed)
    // Just allocate enough for the largest feature map
    size_t max_fmap_size = 256 * 256 * 64;
    std::vector<int8_t, aligned_allocator<int8_t>> host_buf_a(max_fmap_size);
    std::vector<int8_t, aligned_allocator<int8_t>> host_buf_b(max_fmap_size);

    // Gen instructions
    std::vector<NISA_Instruction> instr_list(layers.size() + 1); // +1 for HALT

    // Bank Mapping: 0=Heap, 1=BufA, 2=BufB
    int prev_bank = 0; // Input starts at Heap
    int prev_stride = 1; // Virtual stride of "input"

    for (size_t i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];
        auto& instr = instr_list[i];

	instr.aux_flags = 0;
        // Determine Banks (Ping-Pong)
        // Input is prev_bank. Output toggles between 1 and 2.
        int in_bank = layer.opcode == OP_MEMCPY ? 0 : prev_bank;
        int out_bank = layer.opcode == OP_MEMCPY ? 2 :   // TODO: Deal with this for the memcpy case
		       (in_bank == 1) ? 2 : 1; 
        
        // Bank Sel Byte: [Output(3:2) | Input(1:0)]
        instr.bank_sel = (out_bank << 2) | in_bank;

        // Offsets
        instr.input_offset = (i == 0) ? INSTR_SECTION_SIZE : 0; // L0 reads from Heap offset
        instr.output_offset = 0;
	if (layer.opcode == OP_CONV) {
            instr.weight_offset = layer_weight_offsets[i];   // TODO: Deal with this for layers that come after memcpy/gap
	    instr.aux_flags |= (1 << 3);  // Bias en flag
	    instr.bias_offset = layer_bias_offsets[i];
	} else if (layer.opcode == OP_GEMM) {
	    instr.weight_offset = layer_weight_offsets[i-1];
	    instr.aux_flags |= (1 << 3);  // Bias flag
	    instr.bias_offset = layer_bias_offsets[i-1];
	} else { 
	    instr.weight_offset = 0;
	    instr.bias_offset = 0;
	}

	int hw_width = (layer.in_w < 4) ? 4 : layer.in_w;
	int hw_height = layer.in_h;
        // Dimensions
        instr.width = hw_width;
        instr.height = hw_height;
        // Pad Input Channels for Hardware Loop
        instr.in_channels = (layer.in_c + IC_PAR - 1) / IC_PAR * IC_PAR;
        instr.out_channels = (layer.out_c + OC_PAR - 1) / OC_PAR * OC_PAR;

	if (layer.opcode == OP_CONV) {
            instr.opcode = OP_CONV;
            int mem_tile_h = (i == 0) ? TILE_HEIGHT : (TILE_HEIGHT / prev_stride);
            // FIX: tiny inputs (in_h < mem_tile_h, e.g. conv_7 2x2) -> prev layer output is
            // tiled with only in_h rows; reading mem_tile_h rows/tile misaligns IC-tiles.
            if ((int)layer.in_h < mem_tile_h && layer.in_h > 0) mem_tile_h = layer.in_h;
            instr.log2_mem_tile_height = (int)std::log2(mem_tile_h);
	} else if (layer.opcode == OP_GEMM) {
	    instr.opcode = OP_GEMM;
	    instr.log2_mem_tile_height = 0;
	} else if (layer.opcode == OP_GAP) {
	    instr.opcode = OP_GAP;
	    int rows_per_block = TILE_HEIGHT / prev_stride;
	    instr.log2_mem_tile_height = (int)std::log2(rows_per_block);
	} else if (layer.opcode == OP_MEMCPY) {
	    // Memcpy from INSTR_SECTION_SIZE+L0_FMAP_SIZE 128 bytes to BufA output offset
	    instr.opcode = OP_MEMCPY;
	    instr.input_offset = INSTR_SECTION_SIZE + L0_INP_FMAP_SIZE;
	    instr.output_offset = vision_features * (PP_PAR / 2); //PP_PAR;
	}
        instr.quant_shift = layer.quant_shift;
        instr.stride = layer.stride;
	if (layer.relu)    instr.aux_flags |= (1 << 0);
	// if (layer.flatten) instr.aux_flags |= (1 << 1);
	if (layer.is_sparse) instr.aux_flags |= (1 << 2);
	// if (true) instr.aux_flags |= (1 << 3);  // Bias enabled always for now

	instr.oc_tile_mask_hi = layer.oc_tile_mask_hi;
	instr.oc_tile_mask_lo = layer.oc_tile_mask_lo;
        // Update state for next layer
        prev_bank = out_bank;
        prev_stride = layer.stride;
    }

    instr_list[layers.size()].opcode = OP_HALT;

    std::cout << "Size of NISA instruction struct (in bytes): " << sizeof(NISA_Instruction) << std::endl;
    std::cout << "Length of instruction list: " << instr_list.size() << std::endl;
    std::cout << "Length of layers list: " << layers.size() << std::endl;
    // Copy to Heap
    memcpy(host_heap.data(), instr_list.data(), sizeof(NISA_Instruction) * instr_list.size());

    // OpenCL routines
    cl_int err;
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    cl::Program::Binaries bins = xcl::import_binary_file(argv[1]);
    OCL_CHECK(err, cl::Program program(context, {device}, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel kernel(program, "vla_accel_top:{vla_accel_top_0}", &err));
    // wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");
    OCL_CHECK(err, cl::Buffer d_heap(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, heap_size, host_heap.data(), &err));
    OCL_CHECK(err, cl::Buffer d_buf_a(context, CL_MEM_READ_WRITE, max_fmap_size, NULL, &err));
    OCL_CHECK(err, cl::Buffer d_buf_b(context, CL_MEM_READ_WRITE, max_fmap_size, NULL, &err));
    OCL_CHECK(err, cl::Buffer d_wgt(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, total_weight_size + total_bias_size, host_weights.data(), &err));

    kernel.setArg(0, d_heap);
    kernel.setArg(1, d_wgt);
    kernel.setArg(2, d_buf_a);
    kernel.setArg(3, d_buf_b);

    cl::Event kernel_event;
    std::cout << "INFO: Running Kernel..." << std::endl;
    q.enqueueTask(kernel, NULL, &kernel_event);
    q.finish();

    uint64_t start_time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    uint64_t end_time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double execution_time_ms = (end_time - start_time) / 1000000.0;
    std::cout << " FPGA Kernel Execution Time: " << std::fixed << std::setprecision(3) << execution_time_ms << " ms" << std::endl;

    std::cout << "INFO: Verifying Last Layer Output..." << std::endl;
    
    // Determine where the final output is
    // Buffer alternates each layer EXCEPT memcpy (opcode 3) which doesn't swap.
    // Subtract 1 for the memcpy instruction that doesn't change output buffer.
    cl::Buffer* final_buf = ((layers.size() - 1) % 2 != 0) ? &d_buf_a : &d_buf_b;
    
    auto& last_layer = layers.back();
    size_t out_bytes = 0;
    if (last_layer.opcode == OP_CONV || last_layer.opcode == OP_GEMM) {
	int hw_in_w = (last_layer.in_w < 4) ? 4 : last_layer.in_w;
	int num_strips = (hw_in_w + PP_PAR - 1) / PP_PAR;
	int eff_pp_par = (last_layer.stride == 2) ? (PP_PAR / 2) : PP_PAR;

        int out_w_padded = num_strips * eff_pp_par; //((last_layer.in_w / last_layer.stride) + PP_PAR - 1) / PP_PAR * PP_PAR;
        int out_ch_padded = ((last_layer.out_c + OC_PAR - 1) / OC_PAR) * OC_PAR;
        out_bytes = out_w_padded * 
                    (last_layer.in_h / last_layer.stride) * 
                    out_ch_padded;
    }
    if (last_layer.opcode == OP_MEMCPY) {
	out_bytes = (vision_features + 128) * PP_PAR;
    }
    if (last_layer.opcode == OP_GAP) {
	out_bytes = ((last_layer.out_c + OC_PAR - 1) / OC_PAR) * OC_PAR * PP_PAR;
    }
    std::cout << "Output Bytes to Read: " << out_bytes << std::endl;

    std::vector<int8_t> hw_results(out_bytes);
    q.enqueueReadBuffer(*final_buf, CL_TRUE, 0, out_bytes, hw_results.data());

    std::vector<uint32_t> perf_counters(16, 0);
    q.enqueueReadBuffer(d_buf_a, CL_TRUE, max_fmap_size - 64, 64, perf_counters.data());
    if (perf_counters[3] == 0xDEADBEEF) {
        std::cout << "\n=== N-ISA Hardware Performance Counters ===" << std::endl;
        std::cout << "Total Execution Cycles: " << perf_counters[0] << std::endl;
        std::cout << "Compute Engine Cycles:  " << perf_counters[1] << std::endl;
        std::cout << "Memory Wait Cycles:     " << perf_counters[2] << std::endl;
        std::cout << "---" << std::endl;
        std::cout << "Stall Cycles (idle):    " << perf_counters[4] << std::endl;
        std::cout << "Overlap Cycles (C+M):   " << perf_counters[5] << std::endl;
        std::cout << "Output DMA Cycles:      " << perf_counters[6] << std::endl;
        std::cout << "Mem Stall Cycles (wait): " << perf_counters[7] << std::endl;
        std::cout << "---" << std::endl;
        std::cout << "Stall Breakdown:" << std::endl;
        std::cout << "  Prologue (fetch):     " << perf_counters[8] << std::endl;
        std::cout << "  Barrier (DMA wait):   " << perf_counters[9] << std::endl;
        std::cout << "  Output DMA wait:      " << perf_counters[10] << std::endl;
        uint32_t accounted = perf_counters[8] + perf_counters[9] + perf_counters[10];
        std::cout << "  Other (FSM/init):     " << (perf_counters[4] - accounted) << std::endl;
        std::cout << "===========================================\n" << std::endl;
    } else {
        std::cout << "Performance counters missing or overwritten. Magic: 0x" << std::hex << perf_counters[3] << std::dec << std::endl;
    }

    int errors = 0;
    if (last_layer.opcode == OP_CONV) {
        errors = verify_output(hw_results, last_layer.golden_file, 
                                   last_layer.in_h, last_layer.in_w, last_layer.out_c, 
                                   last_layer.stride, 0, last_layer.is_sparse, last_layer.oc_tile_mask_lo, last_layer.oc_tile_mask_hi);
    } else if (last_layer.opcode == OP_MEMCPY) {
	errors = verify_gap_output(hw_results, last_layer.golden_file, vision_features + 128);
    } else {
    	errors = verify_gap_output(hw_results, last_layer.golden_file, last_layer.out_c);
    }

    std::cout << "Raw output values: " << (int32_t)(hw_results[0]) << ", " << (int32_t)(hw_results[1]) << std::endl;

    if (errors == 0) std::cout << "TEST PASSED!" << std::endl;
    else std::cout << "TEST FAILED with " << errors << " errors." << std::endl;

    return (errors == 0) ? 0 : 1;
}
