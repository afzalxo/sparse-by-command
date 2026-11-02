`timescale 1ns / 1ps

module output_dma #(
    parameter OC_PAR = 16,
    parameter PP_PAR = 8,
    parameter DATA_WIDTH = 8,
    parameter ACCUM_WIDTH = 28,
    parameter TILE_HEIGHT = 4,
    parameter HBM_DATA_WIDTH = 512,
    parameter ADDR_WIDTH = 64,
    parameter BIAS_WIDTH = 32,
    parameter BIAS_BITS_PER_BLOCK = OC_PAR * BIAS_WIDTH,
    parameter URAM_WIDTH = PP_PAR * OC_PAR * ACCUM_WIDTH
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire zero_fill,   // FIX: when 1, write all-zero tile (masked oc-tile slot)

    // Configuration
    input wire [15:0] img_width,
    input wire [15:0] oc_channels,
    input wire [15:0] tile_y_index,
    input wire [15:0] tile_oc_index,
    input wire [15:0] active_height,
    input wire [4:0] quant_shift,
    input wire relu_en,
    input wire [1:0] stride,
    input wire flatten,
    
    // URAM Interface
    output reg [15:0] uram_addr,
    input wire [URAM_WIDTH-1:0] uram_rdata,
    output reg uram_ren,

    // BIAS Interface
    input wire [BIAS_BITS_PER_BLOCK-1:0] bias_rdata,
    
    // Interface to burst_axi_write_master
    output reg start_write,
    output reg [ADDR_WIDTH-1:0] write_base_addr,
    output reg [15:0] write_count,
    output reg [HBM_DATA_WIDTH-1:0] write_data,
    output reg write_valid,
    input wire write_ready,
    input wire write_done,
    output reg done
);

    localparam BITS_PER_PIXEL_BLOCK = OC_PAR * DATA_WIDTH; 
    localparam QUANTIZED_BLOCK_WIDTH = PP_PAR * OC_PAR * DATA_WIDTH;
    localparam CHUNKS_PER_URAM_WORD = (QUANTIZED_BLOCK_WIDTH + HBM_DATA_WIDTH - 1) / HBM_DATA_WIDTH;

    localparam S_IDLE        = 0;
    localparam S_CALC_1      = 1;
    localparam S_CALC_2      = 2;
    localparam S_CALC_3      = 3;
    localparam S_CALC_4      = 4;
    localparam S_START_AXI   = 5;
    localparam S_READ_URAM   = 6;
    localparam S_WAIT_URAM   = 7;
    localparam S_PROCESS     = 8;
    localparam S_FLUSH       = 9;
    localparam S_WAIT_DONE   = 10;
    localparam S_DONE        = 11;
    localparam S_QUANTIZE    = 12;
    localparam S_LATCH_RAW   = 13;
    localparam S_PIPE_WAIT_1   = 14;
    localparam S_PIPE_WAIT_2   = 15;
    localparam S_PIPE_WAIT_3   = 16;
    localparam S_PIPE_WAIT_4   = 17;
    
    reg [4:0] output_dma_state;

    reg [15:0] r_out_width;
    reg [15:0] r_num_w_strips;
    reg [15:0] r_valid_pixels_per_strip;
    reg [15:0] r_active_bits_per_strip;
    reg [31:0] r_global_tile_index;
    
    reg [15:0] r_num_oc_tiles;
    reg [31:0] r_tile_y_stride;
    reg [31:0] r_words_per_row;
    reg [15:0] r_total_uram_words;
    
    reg [31:0] r_stride_oc_tile_words;
    reg [15:0] r_total_uram_words_minus_1;
    
    (* max_fanout = 32 *) reg [URAM_WIDTH-1:0] r_raw_uram_data;
    (* max_fanout = 32 *) reg [BIAS_BITS_PER_BLOCK-1:0] r_bias_data;
    reg [QUANTIZED_BLOCK_WIDTH-1:0] quantized_data;
    reg [QUANTIZED_BLOCK_WIDTH-1:0] r_quantized_data;
    reg r_zero_fill;   // FIX: latched zero_fill for masked oc-tile

    (* max_fanout = 16 *) reg [4:0] quant_shift_repl;
    (* max_fanout = 32 *) reg relu_en_repl;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            quant_shift_repl <= 0;
            relu_en_repl <= 0;
        end else begin
            quant_shift_repl <= quant_shift;
            relu_en_repl <= relu_en;
        end
    end

    // Stage 1: Post-Model-Bias Sum
    (* max_fanout = 32 *) reg signed [ACCUM_WIDTH-1:0] s1_acc_plus_mb [0:PP_PAR-1][0:OC_PAR-1];
    
    // Stage 2: Post-Rounding-Bias Sum (ReLU applied here)
    (* max_fanout = 32 *) reg signed [ACCUM_WIDTH:0] s2_biased_acc [0:PP_PAR-1][0:OC_PAR-1];

    (* max_fanout = 32 *) reg signed [ACCUM_WIDTH-1:0] s3_shifted_acc [0:PP_PAR-1][0:OC_PAR-1];
    
    // Helper for Rounding Bias
    wire signed [ACCUM_WIDTH-1:0] quant_bias = (quant_shift_repl > 0) ? (1 << (quant_shift_repl - 1)) : 0;

    integer pi, oi;
    (* max_fanout = 20 *) wire raw_data_en = (output_dma_state == S_LATCH_RAW);

    always @ (posedge clk or negedge rst_n) begin
	if (!rst_n) begin
	    for (pi = 0; pi < PP_PAR; pi = pi + 1) begin
		for (oi = 0; oi < OC_PAR; oi = oi + 1) begin
		    s1_acc_plus_mb[pi][oi] <= 0;
		    s2_biased_acc[pi][oi] <= 0;
		    s3_shifted_acc[pi][oi] <= 0;
		end
	    end
	    r_quantized_data <= 0;
	    r_raw_uram_data <= 0;
	    r_bias_data <= 0;
	end else begin
	    if (raw_data_en) begin
		r_raw_uram_data <= uram_rdata;
		r_bias_data <= bias_rdata;
		r_quantized_data <= 0;
	    end
	    for (pi = 0; pi < PP_PAR; pi = pi + 1) begin
		for (oi = 0; oi < OC_PAR; oi = oi + 1) begin
		    logic signed [BIAS_WIDTH-1:0] model_bias;
		    model_bias = $signed(r_bias_data[oi * BIAS_WIDTH +: BIAS_WIDTH]);
		    s1_acc_plus_mb[pi][oi] <= $signed(r_raw_uram_data[((pi*OC_PAR + oi)*ACCUM_WIDTH) +: ACCUM_WIDTH]) 
					    + $signed(model_bias[ACCUM_WIDTH-1:0]);
		end
	    end
	
	    for (pi = 0; pi < PP_PAR; pi = pi + 1) begin
		for (oi = 0; oi < OC_PAR; oi = oi + 1) begin
		    automatic logic signed [ACCUM_WIDTH-1:0] relu_val;
		    relu_val = (relu_en_repl && s1_acc_plus_mb[pi][oi][ACCUM_WIDTH-1]) ? 0 : s1_acc_plus_mb[pi][oi];
		    
		    s2_biased_acc[pi][oi] <= $signed(relu_val) + $signed({1'b0, quant_bias});
		end
	    end

	    for (pi = 0; pi < PP_PAR; pi = pi + 1) begin
		for (oi = 0; oi < OC_PAR; oi = oi + 1) begin
		    s3_shifted_acc[pi][oi] <= s2_biased_acc[pi][oi] >>> quant_shift_repl;
		end
	    end

	    for (pi = 0; pi < PP_PAR; pi = pi + 1) begin
		for (oi = 0; oi < OC_PAR; oi = oi + 1) begin
		    automatic logic signed [ACCUM_WIDTH-1:0] shifted;
		    automatic logic signed [DATA_WIDTH-1:0] clamped;
		    
		    shifted = s3_shifted_acc[pi][oi];
		    
		    if (shifted > 127)      clamped = 8'd127;
		    else if (shifted < -128) clamped = -8'd128;
		    else                     clamped = shifted[DATA_WIDTH-1:0];
		    
		    r_quantized_data[((pi*OC_PAR + oi)*DATA_WIDTH) +: DATA_WIDTH] <= r_zero_fill ? {DATA_WIDTH{1'b0}} : clamped;
		end
	    end
	end
    end

    reg [QUANTIZED_BLOCK_WIDTH + HBM_DATA_WIDTH - 1:0] gearbox_buffer;
    reg [15:0] gearbox_fill_level;

    reg [15:0] word_cnt;      
    reg [7:0] input_pixel_cnt;
    reg [3:0] flat_chunk_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_dma_state <= S_IDLE;
            r_zero_fill <= 0;
            done <= 0;
            start_write <= 0;
            write_valid <= 0;
            uram_addr <= 0;
            uram_ren <= 0;
            write_base_addr <= 0;
            write_count <= 0;
            word_cnt <= 0;
            input_pixel_cnt <= 0;
            flat_chunk_idx <= 0;
            write_data <= 0;
            gearbox_buffer <= 0;
            gearbox_fill_level <= 0;
            
            r_out_width <= 0;
            r_num_w_strips <= 0;
            r_valid_pixels_per_strip <= 0;
            r_active_bits_per_strip <= 0;
            r_global_tile_index <= 0;
            r_words_per_row <= 0;
            r_total_uram_words <= 0;
	    r_num_oc_tiles <= 0;
            r_stride_oc_tile_words <= 0;
        end else begin
            start_write <= 0;
            done <= 0;
            
            case (output_dma_state)
                S_IDLE: begin
                    write_valid <= 0;
                    if (start) begin
                        output_dma_state <= S_CALC_1;
			r_zero_fill <= zero_fill;   // FIX: latch for this tile
			r_out_width <= 0;
			r_num_w_strips <= 0;
			r_valid_pixels_per_strip <= 0;
			r_active_bits_per_strip <= 0;
			r_words_per_row <= 0;
			r_total_uram_words <= 0;
			r_num_oc_tiles <= 0;
			r_stride_oc_tile_words <= 0;
                    end
                end

		// --- PIPELINE STAGE 1: Basic Dimensions ---
                S_CALC_1: begin
                    r_out_width <= (stride == 2) ? (img_width >> 1) : img_width;
                    r_num_w_strips <= (img_width + PP_PAR - 1) / PP_PAR;
                    // r_valid_pixels_per_strip <= (stride == 2) ? (PP_PAR >> 1) : PP_PAR;
		    if (stride == 2) begin
			r_valid_pixels_per_strip <= (PP_PAR >> 1);
		    end else begin
			r_valid_pixels_per_strip <= (img_width < PP_PAR) ? img_width : PP_PAR;
		    end
                    output_dma_state <= S_CALC_2;
                end
                
                // --- PIPELINE STAGE 2: Derived Counts ---
                S_CALC_2: begin
		    r_total_uram_words_minus_1 <= ((stride == 2) ? r_num_w_strips * (active_height >> 1) : r_num_w_strips * active_height) - 1;
                    r_active_bits_per_strip <= r_valid_pixels_per_strip * BITS_PER_PIXEL_BLOCK;
                    if (flatten) begin
                        r_words_per_row <= r_out_width * CHUNKS_PER_URAM_WORD;
                    end else begin
                        // r_words_per_row <= (r_out_width * BITS_PER_PIXEL_BLOCK + HBM_DATA_WIDTH - 1) / HBM_DATA_WIDTH;
			r_words_per_row <= (r_num_w_strips * r_valid_pixels_per_strip * BITS_PER_PIXEL_BLOCK + HBM_DATA_WIDTH - 1) / HBM_DATA_WIDTH;
                    end
                    r_total_uram_words <= (stride == 2) ? r_num_w_strips * (active_height >> 1) : r_num_w_strips * active_height;
		    r_num_oc_tiles <= oc_channels / OC_PAR;
                    output_dma_state <= S_CALC_3;
                end
                
                // --- PIPELINE STAGE 3: Strides & Beats ---
                S_CALC_3: begin
                    // 2. Stride OC Tile (Words)
		    if (stride == 2) begin
                        r_stride_oc_tile_words <= r_words_per_row * (active_height >> 1);
		    end
		    else begin
                        r_stride_oc_tile_words <= r_words_per_row * active_height;
		    end
                    
                    // 3. Write Count
                    if (flatten) begin
                        write_count <= r_total_uram_words * r_valid_pixels_per_strip * CHUNKS_PER_URAM_WORD;
                    end else begin
                        write_count <= (r_total_uram_words * r_active_bits_per_strip + HBM_DATA_WIDTH - 1) / HBM_DATA_WIDTH;
                    end
		    r_tile_y_stride <= tile_y_index * r_num_oc_tiles;

                    output_dma_state <= S_CALC_4;
                end

		S_CALC_4: begin
		    write_base_addr <= (r_tile_y_stride + tile_oc_index) * r_stride_oc_tile_words;
		    uram_addr <= 0;
		    word_cnt <= 0;
		    flat_chunk_idx <= 0;
		    gearbox_fill_level <= 0;
		    output_dma_state <= S_START_AXI;
		end 

                S_START_AXI: begin
                    start_write <= 1;
                    uram_ren <= 1;
                    output_dma_state <= S_READ_URAM;
                end
                
                S_READ_URAM: begin
                    uram_ren <= 0;
                    output_dma_state <= S_LATCH_RAW;
                end

		S_LATCH_RAW: begin
		    output_dma_state <= S_PIPE_WAIT_1;
		end

		S_PIPE_WAIT_1: begin
		    output_dma_state <= S_PIPE_WAIT_2;
		end
		S_PIPE_WAIT_2: begin
		    output_dma_state <= S_PIPE_WAIT_3;
		end
		S_PIPE_WAIT_3: begin
		    output_dma_state <= S_PIPE_WAIT_4;
		end

		S_PIPE_WAIT_4: begin
		    output_dma_state <= S_WAIT_URAM;
		end

                S_WAIT_URAM: begin
                    if (flatten) begin
                        input_pixel_cnt <= 0;
                        flat_chunk_idx <= 0;
                        output_dma_state <= S_PROCESS;
                    end else begin
                        // STANDARD MODE: Load Gearbox
                        // gearbox_buffer[gearbox_fill_level +: QUANTIZED_BLOCK_WIDTH] <= r_quantized_data;
			if (gearbox_fill_level == 0) begin
                            gearbox_buffer[0 +: QUANTIZED_BLOCK_WIDTH] <= r_quantized_data;
                        end else begin // Fill level is exactly 512
                            gearbox_buffer[512 +: QUANTIZED_BLOCK_WIDTH] <= r_quantized_data;
                        end	
                        gearbox_fill_level <= gearbox_fill_level + r_active_bits_per_strip;
                        output_dma_state <= S_PROCESS;
                    end
                end
                
                S_PROCESS: begin
		    if (write_valid && !write_ready) begin
                         output_dma_state <= S_PROCESS;
                    end else begin
                    if (flatten) begin
                            if (input_pixel_cnt < r_valid_pixels_per_strip) begin
                                if (flat_chunk_idx == 0) begin
                                    write_data <= {{(HBM_DATA_WIDTH - BITS_PER_PIXEL_BLOCK){1'b0}}, 
                                                   r_quantized_data[(input_pixel_cnt) * BITS_PER_PIXEL_BLOCK +: BITS_PER_PIXEL_BLOCK]};
                                end else begin
                                    write_data <= 0;
                                end
                                write_valid <= 1;

                                if (flat_chunk_idx == CHUNKS_PER_URAM_WORD - 1) begin
                                    flat_chunk_idx <= 0;
                                    input_pixel_cnt <= input_pixel_cnt + 1;
                                end else begin
                                    flat_chunk_idx <= flat_chunk_idx + 1;
                                end
                            end else begin
                                input_pixel_cnt <= 0;
                                flat_chunk_idx <= 0;
				if (word_cnt == r_total_uram_words_minus_1) begin // r_total_uram_words - 1) begin
                                    write_valid <= 0;
                                    output_dma_state <= S_WAIT_DONE;
                                end else begin
                                    word_cnt <= word_cnt + 1;
                                    uram_addr <= uram_addr + 1;
                                    uram_ren <= 1;
                                    write_valid <= 0; 
                                    output_dma_state <= S_READ_URAM;
                                end
                            end
                    end else begin
                        // --- STANDARD MODE (Gearbox) ---
                        if (gearbox_fill_level >= HBM_DATA_WIDTH) begin
                                write_data <= gearbox_buffer[0 +: HBM_DATA_WIDTH];
                                write_valid <= 1;
                                gearbox_buffer <= (gearbox_buffer >> HBM_DATA_WIDTH);
                                gearbox_fill_level <= gearbox_fill_level - HBM_DATA_WIDTH;
                        end else begin
			    write_valid <= 0;
                            if (word_cnt == r_total_uram_words_minus_1) begin
                                if (gearbox_fill_level > 0) begin
                                    output_dma_state <= S_FLUSH;
                                end else begin
                                    output_dma_state <= S_WAIT_DONE;
                                end
                            end else begin
                                word_cnt <= word_cnt + 1;
                                uram_addr <= uram_addr + 1;
                                uram_ren <= 1;
                                output_dma_state <= S_READ_URAM;
                            end
                        end
		    end
	    	    end
                end
                
                S_FLUSH: begin
		    if (write_valid && !write_ready) begin
			output_dma_state <= S_FLUSH;
		    end else begin
                        write_data <= gearbox_buffer[0 +: HBM_DATA_WIDTH];
                        write_valid <= 1;
                        gearbox_fill_level <= 0;
                        output_dma_state <= S_WAIT_DONE;
		    end
                end
                
                S_WAIT_DONE: begin
		    if (write_valid && write_ready) begin
                        write_valid <= 1'b0;
                    end
		    if (write_done) begin
			write_data <= 0;
                        write_valid <= 1'b0;
                        output_dma_state <= S_DONE;
                    end
                end
                
                S_DONE: begin
                    done <= 1;
                    output_dma_state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
