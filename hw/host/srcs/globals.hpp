#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <cstdint>
#include <string>

constexpr uint16_t COMM_PORT = 12345;
constexpr uint32_t NUM_MLP_HEADS = 6; 

constexpr uint16_t IMG_WIDTH  = 256;
constexpr uint16_t IMG_HEIGHT = 256;
constexpr uint16_t IMG_CHANNELS = 3;
constexpr uint32_t IMG_SIZE_BYTES = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS;
constexpr uint32_t EXTRA_FEATURES_BYTES = 128;
constexpr float OUTPUT_SCALES[NUM_MLP_HEADS] = {128.03760724312042, 173.26145216853368, 104.04061515443053, 116.55602761658783, 175.09492976766535, 17.65202029986622};

constexpr uint16_t IC_PAR = 16;
constexpr uint16_t OC_PAR = 16;
constexpr uint16_t PP_PAR = 8;
constexpr uint16_t TILE_HEIGHT = 4;
constexpr size_t INSTR_SECTION_SIZE = 64 * 1024;

typedef int32_t BIAS_T;

constexpr uint8_t OP_CONV   = 1;
constexpr uint8_t OP_GEMM   = 2;
constexpr uint8_t OP_MEMCPY = 3;
constexpr uint8_t OP_GAP    = 4;
constexpr uint8_t OP_HALT   = 255;

// User-defined Layer Configuration
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
    std::string golden_file; // For verification
};

struct alignas(64) NISA_Instruction {
    uint64_t input_offset;     // [63:0]
    uint64_t output_offset;    // [127:64]
    uint64_t weight_offset;    // [191:128]
    uint64_t bias_offset;      // [255:192]
    uint16_t width;            // [271:256]
    uint16_t height;           // [287:272]
    uint16_t in_channels;      // [303:288]
    uint16_t out_channels;     // [319:304]

    uint8_t  opcode;           // [327:320]
    uint8_t  quant_shift;      // [335:328]
    uint8_t  bank_sel;         // [343:336]
    uint8_t  stride;           // [351:344]
    uint8_t  log2_mem_tile_height;  // [359:352]
    // PACKED FLAGS:[bit0: relu, bit1: flatten, bit2: is_sparse, bit3: bias_en]
    uint8_t  aux_flags;        // [367:360]
    uint8_t  padding[2];           // [383:368] - Padding to make 64 bytes
		
    uint64_t oc_tile_mask_lo;          // [447:384]
    uint64_t oc_tile_mask_hi;          // [511:448]
};

struct BranchMetadata {
    size_t weights[3]; // FC1, FC2, Head
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
};
#pragma pack(pop)


#endif
