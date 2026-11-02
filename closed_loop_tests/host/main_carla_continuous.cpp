#include <iostream>
#include <vector>
#include <map>
#include <cstdint>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cstdio>
#include <chrono>
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

// Read FPGA board power via xbutil
float read_fpga_power() {
    FILE* pipe = popen("/opt/xilinx/xrt/bin/xbutil examine -d 0000:01:00.1 --report electrical 2>/dev/null", "r");
    if (!pipe) return 0.0f;
    char line[256];
    float power = 0.0f;
    while (fgets(line, sizeof(line), pipe)) {
        // Look for "  Power                  : XX.XX Watts"
        if (strstr(line, "Power") && strstr(line, "Watts") && !strstr(line, "Max") && !strstr(line, "Warning") && !strstr(line, "Rails")) {
            char* colon = strchr(line, ':');
            if (colon) power = atof(colon + 1);
        }
    }
    pclose(pipe);
    return power;
}

// Load per-task tile masks from task_masks.txt
// Format: 6 lines, each with 8 space-separated uint64 values (oc_tile_mask_lo per conv layer)
void load_task_masks(const std::string& path, uint64_t task_masks[NUM_MLP_HEADS][NUM_CONV_LAYERS]) {
    std::ifstream fs(path);
    if (!fs.is_open()) {
        std::cerr << "WARNING: Could not open " << path << " - using all-ones masks" << std::endl;
        for (int t = 0; t < NUM_MLP_HEADS; t++)
            for (int l = 0; l < NUM_CONV_LAYERS; l++)
                task_masks[t][l] = 0xFFFFFFFFFFFFFFFF;
        return;
    }
    for (int t = 0; t < NUM_MLP_HEADS; t++) {
        for (int l = 0; l < NUM_CONV_LAYERS; l++) {
            fs >> task_masks[t][l];
        }
    }
    std::cout << "Loaded per-task tile masks from " << path << std::endl;
    for (int t = 0; t < NUM_MLP_HEADS; t++) {
        std::cout << "  Task " << t << ": ";
        for (int l = 0; l < NUM_CONV_LAYERS; l++)
            std::cout << "0x" << std::hex << task_masks[t][l] << " ";
        std::cout << std::dec << std::endl;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    // ── Load model spec and task masks ───────────────────────────────────
    int branch_start_instr_idx = 10;  // 8 Convs + 1 Memcpy + 1 Fusion + offset
    int num_instr = 13;               // 8 conv + memcpy + fusion + 3 branch layers
    size_t vision_features = 1024;

    auto layers = load_model_specs("fpga_data_carla/model_spec.txt");

    uint64_t task_masks[NUM_MLP_HEADS][NUM_CONV_LAYERS];
    load_task_masks("fpga_data_carla/task_masks.txt", task_masks);

    // ── Build branch registry ────────────────────────────────────────────
    std::map<int, BranchMetadata> branch_registry;

    size_t L0_INP_FMAP_SIZE = layers[0].in_w * layers[0].in_h *
                              ((layers[0].in_c + IC_PAR - 1) / IC_PAR) * IC_PAR;

    // Pack weights for all layers
    size_t total_weight_size = 0;
    size_t total_bias_size = 0;
    for (const auto& layer : layers) {
        if (layer.opcode == OP_MEMCPY || layer.opcode == OP_GAP) continue;
        bool is_1x1 = layer.opcode == OP_GEMM;
        uint32_t kernel_size = is_1x1 ? 1 : 9;
        int ic_pad = (layer.in_c + IC_PAR - 1) / IC_PAR * IC_PAR;
        int oc_pad = (layer.out_c + OC_PAR - 1) / OC_PAR * OC_PAR;
        total_weight_size += oc_pad * ic_pad * kernel_size;
        total_bias_size += oc_pad * sizeof(BIAS_T);
    }
    std::cout << "Total weight size: " << total_weight_size << " bytes, bias size: " << total_bias_size << " bytes" << std::endl;

    std::vector<uint8_t, aligned_allocator<uint8_t>> host_parameters(total_weight_size + total_bias_size);
    size_t current_weight_offset = 0;
    size_t current_bias_offset = total_weight_size;
    std::vector<size_t> layer_weight_offsets;
    std::vector<size_t> layer_bias_offsets;

    for (const auto& layer : layers) {
        if (layer.opcode == OP_MEMCPY || layer.opcode == OP_GAP) continue;
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
        current_weight_offset += oc_pad * ic_pad * kernel_size;
        current_bias_offset += oc_pad * sizeof(BIAS_T);
    }

    // Populate branch_registry
    for (int b = 0; b < NUM_MLP_HEADS; b++) {
        BranchMetadata meta;
        meta.weights[0] = layer_weight_offsets[branch_start_instr_idx - 1 + b * 3 + 0];
        meta.weights[1] = layer_weight_offsets[branch_start_instr_idx - 1 + b * 3 + 1];
        meta.weights[2] = layer_weight_offsets[branch_start_instr_idx - 1 + b * 3 + 2];
        meta.biases[0] = layer_bias_offsets[branch_start_instr_idx - 1 + b * 3 + 0];
        meta.biases[1] = layer_bias_offsets[branch_start_instr_idx - 1 + b * 3 + 1];
        meta.biases[2] = layer_bias_offsets[branch_start_instr_idx - 1 + b * 3 + 2];
        meta.shifts[0] = layers[branch_start_instr_idx + b * 3 + 0].quant_shift;
        meta.shifts[1] = layers[branch_start_instr_idx + b * 3 + 1].quant_shift;
        meta.shifts[2] = layers[branch_start_instr_idx + b * 3 + 2].quant_shift;
        branch_registry[b] = meta;
    }

    // ── Build instruction list (backbone + memcpy + fusion + 3 branch slots + HALT) ──
    std::vector<NISA_Instruction> instr_list(num_instr + 1);

    int prev_bank = 0;
    int prev_stride = 1;

    for (size_t i = 0; i < (size_t)num_instr; ++i) {
        auto& layer = layers[i];
        auto& instr = instr_list[i];

        instr.aux_flags = 0;

        int in_bank = layer.opcode == OP_MEMCPY ? 0 : prev_bank;
        int out_bank = layer.opcode == OP_MEMCPY ? 2 : (in_bank == 1) ? 2 : 1;
        instr.bank_sel = (out_bank << 2) | in_bank;

        instr.input_offset = (i == 0) ? INSTR_SECTION_SIZE : 0;
        instr.output_offset = 0;

        if (layer.opcode == OP_CONV) {
            instr.weight_offset = layer_weight_offsets[i];
            instr.aux_flags |= (1 << 3);  // bias_en
            instr.bias_offset = layer_bias_offsets[i];
        } else if (layer.opcode == OP_GEMM) {
            instr.weight_offset = layer_weight_offsets[i - 1]; // skip memcpy in offset index
            instr.aux_flags |= (1 << 3);
            instr.bias_offset = layer_bias_offsets[i - 1];
        } else {
            instr.weight_offset = 0;
            instr.bias_offset = 0;
        }

        int hw_width = (layer.in_w < 4) ? 4 : layer.in_w;
        instr.width = hw_width;
        instr.height = layer.in_h;
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
            instr.opcode = OP_MEMCPY;
            instr.input_offset = INSTR_SECTION_SIZE + L0_INP_FMAP_SIZE;
            instr.output_offset = vision_features * (PP_PAR / 2);
        }

        instr.quant_shift = layer.quant_shift;
        instr.stride = layer.stride;
        if (layer.relu) instr.aux_flags |= (1 << 0);
        // Sparsity is set dynamically per command below
        // Start with all-ones (no sparsity) until first command arrives
        instr.oc_tile_mask_lo = 0xFFFFFFFFFFFFFFFF;
        instr.oc_tile_mask_hi = 0xFFFFFFFFFFFFFFFF;

        prev_bank = out_bank;
        prev_stride = layer.stride;
    }
    instr_list[num_instr].opcode = OP_HALT;

    // ── Allocate buffers ─────────────────────────────────────────────────
    size_t max_fmap_size = 256 * 256 * 64;
    std::vector<int8_t, aligned_allocator<int8_t>> host_buf_a(max_fmap_size);
    std::vector<int8_t, aligned_allocator<int8_t>> host_buf_b(max_fmap_size);

    size_t heap_size = INSTR_SECTION_SIZE + L0_INP_FMAP_SIZE + EXTRA_FEATURES_BYTES * 128;
    std::vector<uint8_t, aligned_allocator<uint8_t>> host_heap(heap_size);
    memcpy(host_heap.data(), instr_list.data(), sizeof(NISA_Instruction) * instr_list.size());

    // ── OpenCL setup ─────────────────────────────────────────────────────
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
    OCL_CHECK(err, cl::Buffer d_wgt(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    total_weight_size + total_bias_size, host_parameters.data(), &err));
    kernel.setArg(1, d_wgt);
    kernel.setArg(2, d_buf_a);
    kernel.setArg(3, d_buf_b);

    // ── TCP Server ───────────────────────────────────────────────────────
    int server_fd, client_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    std::vector<uint8_t> image_buffer(IMG_SIZE_BYTES);
    std::vector<uint8_t> extra_features(EXTRA_FEATURES_BYTES, 0);
    InferenceRequest req;
    InferenceResponse res;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(COMM_PORT);
    bind(server_fd, (struct sockaddr *)&address, sizeof(address));
    listen(server_fd, 3);
    std::cout << "N-ISA Server listening on port " << COMM_PORT << "..." << std::endl;

    client_fd = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
    std::cout << "CARLA Client connected!" << std::endl;

    int32_t active_cmd_id = -1;
    bool conn_closed = false;
    int frame_count = 0;

    while (!conn_closed) {
        // Receive request header
        ssize_t r = recv(client_fd, &req, sizeof(InferenceRequest), MSG_WAITALL);
        if (r <= 0) { conn_closed = true; break; }
        if (req.terminate == 1) {
            std::cout << "Terminate signal received." << std::endl;
            conn_closed = true;
            break;
        }

        // Receive image
        size_t bytes_received = 0;
        while (bytes_received < IMG_SIZE_BYTES) {
            int n = recv(client_fd, image_buffer.data() + bytes_received,
                         IMG_SIZE_BYTES - bytes_received, 0);
            if (n <= 0) { conn_closed = true; break; }
            bytes_received += n;
        }
        if (conn_closed) break;

        // Receive extra features
        bytes_received = 0;
        while (bytes_received < EXTRA_FEATURES_BYTES) {
            int n = recv(client_fd, extra_features.data() + bytes_received,
                         EXTRA_FEATURES_BYTES - bytes_received, 0);
            if (n <= 0) { conn_closed = true; break; }
            bytes_received += n;
        }
        if (conn_closed) break;

        // ── Update instructions if command changed ───────────────────────
        if (req.command_id != active_cmd_id) {
            active_cmd_id = req.command_id;
            std::cout << "Command changed to " << active_cmd_id << std::endl;

            // Update conv tile masks (instructions 0-7)
            for (int l = 0; l < NUM_CONV_LAYERS; l++) {
                instr_list[l].oc_tile_mask_lo = task_masks[active_cmd_id][l];
                instr_list[l].oc_tile_mask_hi = 0xFFFFFFFFFFFFFFFF;
                instr_list[l].aux_flags |= (1 << 2);  // is_sparse = 1
            }

            // Update branch MLP layers (instructions branch_start_instr_idx .. +2)
            if (branch_registry.find(active_cmd_id) != branch_registry.end()) {
                BranchMetadata& meta = branch_registry[active_cmd_id];
                for (int i = 0; i < 3; i++) {
                    auto& bi = instr_list[branch_start_instr_idx + i];
                    bi.weight_offset = meta.weights[i];
                    bi.bias_offset = meta.biases[i];
                    bi.quant_shift = meta.shifts[i];
                }
            }
            // Re-copy instruction section to heap
            memcpy(host_heap.data(), instr_list.data(), sizeof(NISA_Instruction) * instr_list.size());
        }

        // ── Pack input ───────────────────────────────────────────────────
        std::vector<int8_t> extra_signed(extra_features.begin(), extra_features.end());
        std::vector<int8_t> packed_extra = pad_vector_for_hw(extra_signed);

        pack_feature_map<uint8_t>(image_buffer.data(), host_heap.data() + INSTR_SECTION_SIZE,
                                  layers[0].in_h, layers[0].in_w, layers[0].in_c,
                                  TILE_HEIGHT, IC_PAR);
        for (size_t i = 0; i < packed_extra.size(); ++i) {
            host_heap[INSTR_SECTION_SIZE + L0_INP_FMAP_SIZE + i] = packed_extra[i];
        }

        // ── Run kernel with timing ───────────────────────────────────────
        OCL_CHECK(err, cl::Buffer d_heap(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         heap_size, host_heap.data(), &err));
        kernel.setArg(0, d_heap);

        cl::Event kernel_event;
        auto t_start = std::chrono::high_resolution_clock::now();
        q.enqueueTask(kernel, NULL, &kernel_event);
        q.finish();
        auto t_end = std::chrono::high_resolution_clock::now();
        float latency_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

        // ── Read output ──────────────────────────────────────────────────
        cl::Buffer* final_buf = ((instr_list.size() - 2) % 2 != 0) ? &d_buf_a : &d_buf_b;
        auto& last_instr = instr_list[instr_list.size() - 2];
        size_t out_bytes = 0;
        if (last_instr.opcode == OP_CONV || last_instr.opcode == OP_GEMM) {
            int hw_w = (last_instr.width < 4) ? 4 : last_instr.width;
            int num_strips = (hw_w + PP_PAR - 1) / PP_PAR;
            int eff_pp = (last_instr.stride == 2) ? (PP_PAR / 2) : PP_PAR;
            int out_w_padded = num_strips * eff_pp;
            int out_ch_padded = ((last_instr.out_channels + OC_PAR - 1) / OC_PAR) * OC_PAR;
            out_bytes = out_w_padded * (last_instr.height / last_instr.stride) * out_ch_padded;
        }
        if (last_instr.opcode == OP_MEMCPY) out_bytes = (vision_features + 128) * PP_PAR;
        if (last_instr.opcode == OP_GAP) out_bytes = ((last_instr.out_channels + OC_PAR - 1) / OC_PAR) * OC_PAR * PP_PAR;

        std::vector<int8_t> hw_results(out_bytes);
        q.enqueueReadBuffer(*final_buf, CL_TRUE, 0, out_bytes, hw_results.data());

        // ── Post-process output ──────────────────────────────────────────
        res.steer = std::tanh(hw_results[0] / OUTPUT_SCALES[active_cmd_id]);
        res.accel = std::tanh(hw_results[1] / OUTPUT_SCALES[active_cmd_id]);
        res.latency_ms = latency_ms;

        // Read power every 50 frames to avoid subprocess overhead
        if (frame_count % 50 == 0) {
            res.power_w = read_fpga_power();
        }
        // (power_w persists from last reading for intermediate frames)

        send(client_fd, &res, sizeof(InferenceResponse), 0);
        frame_count++;

        if (frame_count % 100 == 0) {
            std::cout << "Frame " << frame_count << " | cmd=" << active_cmd_id
                      << " | lat=" << std::fixed << std::setprecision(2) << latency_ms
                      << "ms | pwr=" << res.power_w << "W" << std::endl;
        }
    }

    close(client_fd);
    close(server_fd);
    std::cout << "Server shut down after " << frame_count << " frames." << std::endl;
    return 0;
}
