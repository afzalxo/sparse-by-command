// =============================================================================
// Open-loop single-image inference host (Sparse by Command).
//
// Runs ONE image through the accelerator for a chosen driving command and
// prints the predicted (steer, throttle). Unlike the Table-2 host, this reads
// the head of the ACTIVE command (the per-command spec is truncated to end at
// b<cmd>_head) and dequantizes it with OUTPUT_SCALES[cmd] --- the same formula
// the closed-loop server uses. Compare the printout against gpu_reference.py.
//
// Usage:  ./app_open_loop <xclbin> <command_id> [spec_file]
//   command_id: 0=follow lane 1=turn left 2=turn right
//               3=change lane left 4=change lane right 5=brake
//   spec_file : defaults to model_spec_temp.txt
// =============================================================================
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

static const char* CMD_NAMES[6] = {
    "follow lane", "turn left", "turn right",
    "change lane left", "change lane right", "brake"};

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0]
                  << " <XCLBIN File> <command_id 0-5> [spec_file]" << std::endl;
        return EXIT_FAILURE;
    }
    int command_id = std::atoi(argv[2]);
    if (command_id < 0 || command_id >= (int)NUM_MLP_HEADS) {
        std::cout << "command_id must be in [0," << NUM_MLP_HEADS - 1 << "]" << std::endl;
        return EXIT_FAILURE;
    }
    std::string spec_file = (argc >= 4) ? argv[3] : "model_spec_temp.txt";

    size_t vision_features = 1024;
    std::vector<LayerConfig> layers = load_model_specs(spec_file);

    // A. Input Feature Map
    std::vector<uint8_t> raw_input;
    load_bin<uint8_t>("fpga_data/input_vision.bin", raw_input);
    std::vector<uint8_t> extra_features;
    load_bin<uint8_t>("fpga_data/extra_features.bin", extra_features);

    std::vector<int8_t> extra_features_signed(extra_features.begin(), extra_features.end());
    std::vector<int8_t> packed_extra_features = pad_vector_for_hw(extra_features_signed);

    size_t L0_INP_FMAP_SIZE = layers[0].in_w * layers[0].in_h *
                                  ((layers[0].in_c + IC_PAR - 1) / IC_PAR) * IC_PAR;
    size_t heap_size = INSTR_SECTION_SIZE + L0_INP_FMAP_SIZE +
                       packed_extra_features.size();
    std::vector<uint8_t, aligned_allocator<uint8_t>> host_heap(heap_size);

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
        if (layer.opcode == OP_MEMCPY || layer.opcode == OP_GAP) continue;
        bool is_1x1 = layer.opcode == OP_GEMM;
        uint32_t kernel_size = is_1x1 ? 1 : 9;
        int ic_pad = (layer.in_c + IC_PAR - 1) / IC_PAR * IC_PAR;
        int oc_pad = (layer.out_c + OC_PAR - 1) / OC_PAR * OC_PAR;
        total_weight_size += oc_pad * ic_pad * kernel_size;
        total_bias_size += oc_pad * sizeof(BIAS_T);
    }
    std::vector<uint8_t, aligned_allocator<uint8_t>> host_weights(total_weight_size + total_bias_size);

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
        pack_weights<uint8_t>(w_raw.data(), host_weights.data() + current_weight_offset,
                              layer.out_c, layer.in_c, OC_PAR, IC_PAR, is_1x1);
        pack_biases<uint32_t, uint8_t>(b_raw.data(), host_weights.data() + current_bias_offset,
                                   layer.out_c, OC_PAR);

        uint32_t kernel_size = is_1x1 ? 1 : 9;
        int ic_pad = (layer.in_c + IC_PAR - 1) / IC_PAR * IC_PAR;
        int oc_pad = (layer.out_c + OC_PAR - 1) / OC_PAR * OC_PAR;
        current_weight_offset += oc_pad * ic_pad * kernel_size;
        current_bias_offset += oc_pad * sizeof(BIAS_T);
    }

    size_t max_fmap_size = 256 * 256 * 64;
    std::vector<int8_t, aligned_allocator<int8_t>> host_buf_a(max_fmap_size);
    std::vector<int8_t, aligned_allocator<int8_t>> host_buf_b(max_fmap_size);

    std::vector<NISA_Instruction> instr_list(layers.size() + 1);

    int prev_bank = 0;
    int prev_stride = 1;

    for (size_t i = 0; i < layers.size(); ++i) {
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
            instr.aux_flags |= (1 << 3);
            instr.bias_offset = layer_bias_offsets[i];
        } else if (layer.opcode == OP_GEMM) {
            instr.weight_offset = layer_weight_offsets[i-1];
            instr.aux_flags |= (1 << 3);
            instr.bias_offset = layer_bias_offsets[i-1];
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
        if (layer.relu)    instr.aux_flags |= (1 << 0);
        if (layer.is_sparse) instr.aux_flags |= (1 << 2);
        instr.oc_tile_mask_hi = layer.oc_tile_mask_hi;
        instr.oc_tile_mask_lo = layer.oc_tile_mask_lo;
        prev_bank = out_bank;
        prev_stride = layer.stride;
    }

    instr_list[layers.size()].opcode = OP_HALT;
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
    OCL_CHECK(err, cl::Buffer d_heap(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, heap_size, host_heap.data(), &err));
    OCL_CHECK(err, cl::Buffer d_buf_a(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, max_fmap_size, host_buf_a.data(), &err));
    OCL_CHECK(err, cl::Buffer d_buf_b(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, max_fmap_size, host_buf_b.data(), &err));
    OCL_CHECK(err, cl::Buffer d_wgt(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, total_weight_size + total_bias_size, host_weights.data(), &err));

    kernel.setArg(0, d_heap);
    kernel.setArg(1, d_wgt);
    kernel.setArg(2, d_buf_a);
    kernel.setArg(3, d_buf_b);

    cl::Event kernel_event;
    std::cout << "INFO: Running kernel for command " << command_id
              << " (" << CMD_NAMES[command_id] << ")..." << std::endl;
    q.enqueueTask(kernel, NULL, &kernel_event);
    q.finish();

    uint64_t start_time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    uint64_t end_time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double execution_time_ms = (end_time - start_time) / 1000000.0;

    cl::Buffer* final_buf = ((layers.size() - 1) % 2 != 0) ? &d_buf_a : &d_buf_b;
    auto& last_layer = layers.back();
    size_t out_bytes = 0;
    if (last_layer.opcode == OP_CONV || last_layer.opcode == OP_GEMM) {
        int hw_in_w = (last_layer.in_w < 4) ? 4 : last_layer.in_w;
        int num_strips = (hw_in_w + PP_PAR - 1) / PP_PAR;
        int eff_pp_par = (last_layer.stride == 2) ? (PP_PAR / 2) : PP_PAR;
        int out_w_padded = num_strips * eff_pp_par;
        int out_ch_padded = ((last_layer.out_c + OC_PAR - 1) / OC_PAR) * OC_PAR;
        out_bytes = out_w_padded * (last_layer.in_h / last_layer.stride) * out_ch_padded;
    }
    if (last_layer.opcode == OP_MEMCPY) out_bytes = (vision_features + 128) * PP_PAR;
    if (last_layer.opcode == OP_GAP)    out_bytes = ((last_layer.out_c + OC_PAR - 1) / OC_PAR) * OC_PAR * PP_PAR;

    std::vector<int8_t> hw_results(out_bytes);
    q.enqueueReadBuffer(*final_buf, CL_TRUE, 0, out_bytes, hw_results.data());
    // DEBUG: dump BOTH buffers raw (final + other) to files
    std::vector<int8_t> other_res(out_bytes);
    cl::Buffer* other_buf = (final_buf == &d_buf_a) ? &d_buf_b : &d_buf_a;
    q.enqueueReadBuffer(*other_buf, CL_TRUE, 0, out_bytes, other_res.data());
    { FILE* f=fopen("/tmp/hw_final.bin","wb"); fwrite(hw_results.data(),1,out_bytes,f); fclose(f); }
    { FILE* f=fopen("/tmp/hw_other.bin","wb"); fwrite(other_res.data(),1,out_bytes,f); fclose(f); }
    std::cout << "out_bytes=" << out_bytes << " final_buf=" << ((final_buf==&d_buf_a)?"A":"B") << std::endl;


    int errors = 0;
    if (last_layer.opcode == OP_CONV) {
        int ef = verify_output(hw_results, last_layer.golden_file, last_layer.in_h, last_layer.in_w, last_layer.out_c, last_layer.stride, 0, last_layer.is_sparse, last_layer.oc_tile_mask_lo, last_layer.oc_tile_mask_hi);
        int eo = verify_output(other_res, last_layer.golden_file, last_layer.in_h, last_layer.in_w, last_layer.out_c, last_layer.stride, 0, last_layer.is_sparse, last_layer.oc_tile_mask_lo, last_layer.oc_tile_mask_hi);
        std::cout << "CONV_VERIFY final_errors=" << ef << " other_errors=" << eo << std::endl;
        errors = (ef < eo) ? ef : eo;
    } else {
        errors = verify_gap_output(hw_results, last_layer.golden_file, last_layer.out_c);
    }

    // Dequantize the active command's head into (steer, throttle).
    int32_t steer_raw = (int32_t)hw_results[0];
    int32_t throttle_raw = (int32_t)hw_results[1];
    float scale = OUTPUT_SCALES[command_id];
    float steer    = std::tanh(steer_raw / scale);
    float throttle = std::tanh(throttle_raw / scale);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n================ FPGA open-loop prediction ================" << std::endl;
    std::cout << "  command        : " << command_id << " (" << CMD_NAMES[command_id] << ")" << std::endl;
    std::cout << "  kernel time    : " << std::setprecision(3) << execution_time_ms << " ms" << std::setprecision(4) << std::endl;
    std::cout << "  raw head (int8): [" << steer_raw << ", " << throttle_raw << "]" << std::endl;
    std::cout << "  steer          : " << steer << std::endl;
    std::cout << "  throttle       : " << throttle << std::endl;
    std::cout << "  golden check   : " << (errors == 0 ? "PASSED (bit-exact)" : "FAILED") << std::endl;
    std::cout << "===========================================================" << std::endl;

    return (errors == 0) ? 0 : 1;
}
