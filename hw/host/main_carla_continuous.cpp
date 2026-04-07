#include <iostream>
#include <vector>
#include <map>
#include <cstdint>
#include <cmath>
#include <string>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

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

    int branch_start_instr_idx = 9;  // 6 Convs + 1 GAP + 1 Memcpy + 1 Fusion
    // Total instructions: 8 (backbone) + 3 (active branch) + 1 (halt)
    std::map<int, BranchMetadata> branch_registry;

    size_t vision_features = 1024;
    auto layers = load_model_specs("fpga_data_carla/model_spec.txt");

    size_t L0_INP_FMAP_SIZE = layers[0].in_w * layers[0].in_h * 
                                  ((layers[0].in_c + IC_PAR - 1) / IC_PAR) * IC_PAR;
    // Weights (Stack all layers)
    size_t total_weight_size = 0;
    size_t total_bias_size = 0;
    for (const auto& layer : layers) {
	if (layer.opcode == OP_MEMCPY || layer.opcode == OP_GAP) {
	    continue;
	}
	uint32_t kernel_size = (layer.opcode == OP_GEMM) ? 1 : 9;
        // Align to IC_PAR/OC_PAR
        int ic_pad = (layer.in_c + IC_PAR - 1) / IC_PAR * IC_PAR;
        int oc_pad = (layer.out_c + OC_PAR - 1) / OC_PAR * OC_PAR;
        total_weight_size += oc_pad * ic_pad * kernel_size;
	total_bias_size += oc_pad * sizeof(BIAS_T);
    }
    std::cout << "Total weight size (bytes): " << total_weight_size << std::endl;
    std::cout << "Total bias size (bytes): " << total_bias_size << std::endl;
    std::vector<uint8_t, aligned_allocator<uint8_t>> host_parameters(total_weight_size + total_bias_size);
    
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
        
        pack_weights<uint8_t>(w_raw.data(), host_parameters.data() + current_weight_offset, 
                              layer.out_c, layer.in_c, OC_PAR, IC_PAR, is_1x1);
	pack_biases<uint32_t, uint8_t>(b_raw.data(), host_parameters.data() + current_bias_offset,
			      layer.out_c, OC_PAR);
        
        int ic_pad = (layer.in_c + IC_PAR - 1) / IC_PAR * IC_PAR;
        int oc_pad = (layer.out_c + OC_PAR - 1) / OC_PAR * OC_PAR;
	std::cout << "Placed weights for " << layer.name << " at offset " << current_weight_offset << " (size " << (oc_pad * ic_pad * kernel_size) << " bytes)" << std::endl;
	std::cout << "Placed biases for " << layer.name << " at offset " << current_bias_offset << " (size " << (oc_pad * sizeof(BIAS_T)) << " bytes)" << std::endl;
        current_weight_offset += oc_pad * ic_pad * kernel_size;
	current_bias_offset += oc_pad * sizeof(BIAS_T);
    }

    // C. Output Buffers (Max size needed)
    // Just allocate enough for the largest feature map
    size_t max_fmap_size = 256 * 256 * 64;
    std::vector<int8_t, aligned_allocator<int8_t>> host_buf_a(max_fmap_size);
    std::vector<int8_t, aligned_allocator<int8_t>> host_buf_b(max_fmap_size);

    // Gen instructions
    // std::vector<NISA_Instruction> instr_list(layers.size() + 1); // +1 for HALT
    const int num_layers = 12;
    const int num_branches = 6;
    // Populate branch_registry with offsets of weights and biases corresponding to the branches
    for (int b = 0; b < num_branches; b++) {
	BranchMetadata meta;
	// layer_weight_offsets index 7 and beyond are for the active branch layers
	meta.weights[0] = layer_weight_offsets[branch_start_instr_idx - 2 + b * 3 + 0]; // 3 dense layers per branch
	meta.weights[1] = layer_weight_offsets[branch_start_instr_idx - 2 + b * 3 + 1]; // - 2 since there are two auxiliary layers (GAP + Memcpy) before the first branch layer
	meta.weights[2] = layer_weight_offsets[branch_start_instr_idx - 2 + b * 3 + 2]; 
	meta.biases[0] = layer_bias_offsets[branch_start_instr_idx - 2 + b * 3 + 0];
	meta.biases[1] = layer_bias_offsets[branch_start_instr_idx - 2 + b * 3 + 1];
	meta.biases[2] = layer_bias_offsets[branch_start_instr_idx - 2 + b * 3 + 2];
	meta.shifts[0] = layers[branch_start_instr_idx + b * 3 + 0].quant_shift;
	meta.shifts[1] = layers[branch_start_instr_idx + b * 3 + 1].quant_shift;
	meta.shifts[2] = layers[branch_start_instr_idx + b * 3 + 2].quant_shift;
	// std::cout << "Branch " << b << " w0 offset: " << meta.weights[0] << ", b0 offset: " << meta.biases[0] << ", shift0: " << (int)meta.shifts[0] << " w1 offset: " << meta.weights[1] << ", b1 offset: " << meta.biases[1] << ", shift1: " << (int)meta.shifts[1] << " w2 offset: " << meta.weights[2] << ", b2 offset: " << meta.biases[2] << ", shift2: " << (int)meta.shifts[2] << std::endl;
	branch_registry[b] = meta;
    }
    std::vector<NISA_Instruction> instr_list(num_layers + 1); // 6 (conv) + 2 (gap + memcpy) + 1 (fusion) + 3 (active branch) + 1 (halt)

    std::cout << "Populating NISA instruction list..." << std::endl;
    // Bank Mapping: 0=Heap, 1=BufA, 2=BufB
    int prev_bank = 0; // Input starts at Heap
    int prev_stride = 1; // Virtual stride of "input"

    for (size_t i = 0; i < num_layers; ++i) {
        auto& layer = layers[i];
        auto& instr = instr_list[i];

        // Determine Banks (Ping-Pong)
        // Input is prev_bank. Output toggles between 1 and 2.
        int in_bank = layer.opcode == OP_MEMCPY ? 0 : prev_bank;
        int out_bank = layer.opcode == OP_MEMCPY ? 1 :   // TODO: Deal with this for the memcpy case
		       (in_bank == 1) ? 2 : 1; 
        
        // Bank Sel Byte: [Output(3:2) | Input(1:0)]
        instr.bank_sel = (out_bank << 2) | in_bank;

        // Offsets
        instr.input_offset = (i == 0) ? INSTR_SECTION_SIZE : 0; // L0 reads from Heap offset
        instr.output_offset = 0;
	if (layer.opcode == OP_CONV) {
            instr.weight_offset = layer_weight_offsets[i];   // TODO: Deal with this for layers that come after memcpy/gap
	    instr.bias_en = 1;
	    instr.bias_offset = layer_bias_offsets[i];
	} else if (layer.opcode == OP_GEMM) {
	    instr.weight_offset = layer_weight_offsets[i-2];
	    instr.bias_en = 1;
	    instr.bias_offset = layer_bias_offsets[i-2];
	} else { 
	    instr.weight_offset = 0;
	    instr.bias_en = 0;
	}

        // Dimensions
        instr.width = layer.in_w;
        instr.height = layer.in_h;
        // Pad Input Channels for Hardware Loop
        instr.in_channels = (layer.in_c + IC_PAR - 1) / IC_PAR * IC_PAR;
        instr.out_channels = (layer.out_c + OC_PAR - 1) / OC_PAR * OC_PAR;

	if (layer.opcode == OP_CONV) {
            instr.opcode = OP_CONV;
            int mem_tile_h = (i == 0) ? TILE_HEIGHT : (TILE_HEIGHT / prev_stride);
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
	    instr.output_offset = vision_features * PP_PAR;
	}
        instr.quant_shift = layer.quant_shift;
        instr.relu_en = layer.relu ? 1 : 0;
        instr.stride = layer.stride;

	instr.flatten = 0;
	instr.is_sparse = 0;
	instr.ic_tile_mask = 0xFFFFFFFF;
	instr.oc_tile_mask = 0xFFFFFFFF;
        // Update state for next layer
        prev_bank = out_bank;
        prev_stride = layer.stride;
    }

    // instr_list[layers.size()].opcode = OP_HALT;
    instr_list[num_layers].opcode = OP_HALT;

    size_t heap_size = INSTR_SECTION_SIZE + L0_INP_FMAP_SIZE + 
		       EXTRA_FEATURES_BYTES * 128;
    std::vector<uint8_t, aligned_allocator<uint8_t>> host_heap(heap_size);
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
    OCL_CHECK(err, cl::Buffer d_buf_a(context, CL_MEM_READ_WRITE, max_fmap_size, NULL, &err));
    OCL_CHECK(err, cl::Buffer d_buf_b(context, CL_MEM_READ_WRITE, max_fmap_size, NULL, &err));
    OCL_CHECK(err, cl::Buffer d_wgt(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, total_weight_size + total_bias_size, host_parameters.data(), &err));

    kernel.setArg(1, d_wgt);
    kernel.setArg(2, d_buf_a);
    kernel.setArg(3, d_buf_b);

    int server_fd, client_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    std::vector<uint8_t> image_buffer(IMG_SIZE_BYTES);
    std::vector<uint8_t> extra_features(EXTRA_FEATURES_BYTES, 0);
    InferenceRequest req;
    InferenceResponse res;

    // 1. Create Socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(COMM_PORT);

    // Place image and extra features recv code here
    bind(server_fd, (struct sockaddr *)&address, sizeof(address));
    listen(server_fd, 3);
    std::cout << "N-ISA Server listening on port " << COMM_PORT << "..." << std::endl;

    client_fd = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
    std::cout << "CARLA Client connected!" << std::endl;

    int32_t active_cmd_id = -1;
    bool conn_closed = false;
    while (!conn_closed) {

        recv(client_fd, &req, sizeof(InferenceRequest), 0);
	if (req.terminate == 1) {
	    conn_closed = true;
	    std::cout << "Termination signal received. Closing connection." << std::endl;
	    close(client_fd);
	    close(server_fd);
	    continue;
	}

        size_t bytes_received = 0;
        while (bytes_received < IMG_SIZE_BYTES) {
                int r = recv(client_fd, image_buffer.data() + bytes_received, IMG_SIZE_BYTES - bytes_received, 0);
                if (r <= 0) {
                    close(client_fd);
            	    close(server_fd);
            	    std::cout << "Connection closed while receiving image data." << std::endl;
		    conn_closed = true;
		    break;
                }
                bytes_received += r;
        }
        while (bytes_received < IMG_SIZE_BYTES + EXTRA_FEATURES_BYTES) {
                int r = recv(client_fd, extra_features.data() + (bytes_received - IMG_SIZE_BYTES), 
            		 EXTRA_FEATURES_BYTES - (bytes_received - IMG_SIZE_BYTES), 0);
                if (r <= 0) {
            	close(client_fd);
            	close(server_fd);
            	std::cout << "Connection closed while receiving extra features." << std::endl;
		conn_closed = true;
		break;
                }
                bytes_received += r;
        }
	// If command has changed, we update the instruction list here
	if (req.command_id != active_cmd_id) {
	    std::cout << "Received new command ID: " << req.command_id << std::endl;
	    active_cmd_id = req.command_id;
	    // Update the instruction list's branch metadata based on the command ID
	    if (branch_registry.find(req.command_id) != branch_registry.end()) {
		BranchMetadata meta = branch_registry[req.command_id];
		// Update the 3 active branch layers in the instruction list
		int instr_idx = branch_start_instr_idx;
		for (int i = 0; i < 3; i++) {
		    instr_list[instr_idx+i].weight_offset = meta.weights[i];
		    instr_list[instr_idx+i].bias_offset = meta.biases[i];
		    instr_list[instr_idx+i].quant_shift = meta.shifts[i];
		    std::cout << "Updating instruction at index " << instr_idx+i << " with weight offset: " << meta.weights[i] << ", bias offset: " << meta.biases[i] << ", quant shift: " << (int)meta.shifts[i] << std::endl;
		}
	    } else {
		std::cerr << "Warning: Received unknown command ID " << req.command_id << ". No update to instructions." << std::endl;
	    }
    	    memcpy(host_heap.data(), instr_list.data(), sizeof(NISA_Instruction) * instr_list.size());
	}

        std::vector<int8_t> extra_features_signed(extra_features.begin(), extra_features.end());
        std::vector<int8_t> packed_extra_features = pad_vector_for_hw(extra_features_signed);

        // Pack Input into Heap
        pack_feature_map<uint8_t>(image_buffer.data(), host_heap.data() + INSTR_SECTION_SIZE, 
                                  layers[0].in_h, layers[0].in_w, layers[0].in_c, 
                                  TILE_HEIGHT, IC_PAR);
        for (size_t i = 0; i < packed_extra_features.size(); ++i) {
            host_heap[INSTR_SECTION_SIZE + L0_INP_FMAP_SIZE + i] = packed_extra_features[i];
        }

        OCL_CHECK(err, cl::Buffer d_heap(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, heap_size, host_heap.data(), &err));

        kernel.setArg(0, d_heap);
        // std::cout << "INFO: Running Kernel..." << std::endl;
        q.enqueueTask(kernel);
        q.finish();
        // std::cout << "INFO: Verifying Last Layer Output..." << std::endl;
        
        cl::Buffer* final_buf = ((instr_list.size() - 2) % 2 != 0) ? &d_buf_a : &d_buf_b;
        auto& last_instr = instr_list[instr_list.size() - 2];
        size_t out_bytes = 0;
        if (last_instr.opcode == OP_CONV || last_instr.opcode == OP_GEMM) {
            int out_w_padded = ((last_instr.width / last_instr.stride) + PP_PAR - 1) / PP_PAR * PP_PAR;
            int out_ch_padded = ((last_instr.out_channels + OC_PAR - 1) / OC_PAR) * OC_PAR;
            out_bytes = out_w_padded * 
                        (last_instr.height / last_instr.stride) * 
                        out_ch_padded;
        }
        if (last_instr.opcode == OP_MEMCPY) {
            out_bytes = (vision_features + 128) * PP_PAR;
        }
        if (last_instr.opcode == OP_GAP) {
            out_bytes = ((last_instr.out_channels + OC_PAR - 1) / OC_PAR) * OC_PAR * PP_PAR;
        }
        // std::cout << "Output Bytes to Read: " << out_bytes << std::endl;

        std::vector<int8_t> hw_results(out_bytes);
        q.enqueueReadBuffer(*final_buf, CL_TRUE, 0, out_bytes, hw_results.data());

        std::cout << "Raw output values: " << (int32_t)(hw_results[0]) << ", " << (int32_t)(hw_results[1]) << std::endl;
        res.steer = std::tanh(hw_results[0] / OUTPUT_SCALES[active_cmd_id]);
        res.accel = std::tanh(hw_results[1] / OUTPUT_SCALES[active_cmd_id]);

        send(client_fd, &res, sizeof(InferenceResponse), 0);
    }

    // if client_fd and server_fd have not been closed in the recv loop, close them here
    if (client_fd > 0) {
	close(client_fd);
	std::cout << "Client connection closed." << std::endl;
    }
    if (server_fd > 0) {
	close(server_fd);
	std::cout << "Server socket closed." << std::endl;
    }
}
