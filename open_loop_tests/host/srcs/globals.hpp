#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <cstdint>
#include <string>

constexpr uint16_t COMM_PORT = 12345;
constexpr uint32_t NUM_MLP_HEADS = 6; 
constexpr uint32_t NUM_CONV_LAYERS = 8;

constexpr uint16_t IMG_WIDTH  = 256;
constexpr uint16_t IMG_HEIGHT = 256;
constexpr uint16_t IMG_CHANNELS = 3;
constexpr uint32_t IMG_SIZE_BYTES = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS;
constexpr uint32_t EXTRA_FEATURES_BYTES = 128;
constexpr float OUTPUT_SCALES[NUM_MLP_HEADS] = {115.88358565620587f, 48.785212214480566f, 71.12760893340649f, 95.04763828343076f, 97.91779153171946f, 14.295189688054489f};

constexpr uint16_t IC_PAR = 16;
constexpr uint16_t OC_PAR = 16;
constexpr uint16_t PP_PAR = 8;
constexpr uint16_t TILE_HEIGHT = 8;
constexpr size_t INSTR_SECTION_SIZE = 64 * 1024;

typedef int32_t BIAS_T;

constexpr uint8_t OP_CONV   = 1;
constexpr uint8_t OP_GEMM   = 2;
constexpr uint8_t OP_MEMCPY = 3;
constexpr uint8_t OP_GAP    = 4;
constexpr uint8_t OP_HALT   = 255;

struct LayerConfig {
    std::string name;
    int in_w, in_h, in_c;
    int out_c;
    int stride;
    int quant_shift;
    bool relu;
    int opcode;
    bool is_sparse;
    uint64_t oc_tile_mask_lo;
    uint64_t oc_tile_mask_hi;
    std::string weight_file;
    std::string bias_file;
    std::string golden_file;
};

struct alignas(64) NISA_Instruction {
    uint64_t input_offset;
    uint64_t output_offset;
    uint64_t weight_offset;
    uint64_t bias_offset;
    uint16_t width;
    uint16_t height;
    uint16_t in_channels;
    uint16_t out_channels;
    uint8_t  opcode;
    uint8_t  quant_shift;
    uint8_t  bank_sel;
    uint8_t  stride;
    uint8_t  log2_mem_tile_height;
    uint8_t  aux_flags;   // bit0: relu, bit1: flatten, bit2: is_sparse, bit3: bias_en
    uint8_t  padding[2];
    uint64_t oc_tile_mask_lo;
    uint64_t oc_tile_mask_hi;
};

struct BranchMetadata {
    size_t weights[3];
    size_t biases[3];
    int shifts[3];
};

#pragma pack(push, 1)
struct InferenceRequest {
    int32_t command_id;
    int32_t terminate;
};

struct InferenceResponse {
    float steer;
    float accel;
    float latency_ms;
    float power_w;
};
#pragma pack(pop)

#endif
